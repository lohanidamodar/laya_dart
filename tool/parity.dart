// End-to-end parity against laya-serve on real weights (Task 17).
//
// Usage:
//   dart run tool/parity.dart [fixtures] [bench]
//
// `fixtures` runs tool/reference/serve_parity.py once per checkpoint and writes
// test/fixtures/parity_<checkpoint>.json: laya-serve (torch, CPU) on the upstream
// weights, read offline from the local Hugging Face cache (HF_HUB_OFFLINE=1).
//
// `bench` measures one representative request per checkpoint on both sides:
// the median of 20 after a warm-up, with LAYA_THREADS (torch) and the ONNX
// Runtime intra-op threads at the physical core count, plus each process's peak
// working set. It writes .tooling/logs/parity_latency.json.
//
// With no argument it does both. Logs go to .tooling/logs/. Only english and
// multilingual are run: typed-decisions' weights are not downloaded.
//
// The ONNX models must be installed under models/ (dart run laya_server:fetch).
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/laya_server.dart';
import 'package:shelf/shelf.dart';

const _checkpoints = ['english', 'multilingual'];

/// The request (an index into cases.json's `requests`) timed for each
/// checkpoint: three questions of all three types on an email state.
const _benchRequest = 23;
const _benchRuns = 20;

String get _python => File(
  Platform.isWindows
      ? '.tooling/venv/Scripts/python.exe'
      : '.tooling/venv/bin/python',
).absolute.path;

/// The physical core count, or `LAYA_THREADS` when set.
Future<int> _cores() async {
  final env = int.tryParse(Platform.environment['LAYA_THREADS'] ?? '');
  if (env != null && env > 0) return env;
  if (Platform.isWindows) {
    final r = await Process.run('powershell', [
      '-NoProfile',
      '-Command',
      '(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum',
    ]);
    final n = int.tryParse('${r.stdout}'.trim());
    if (n != null && n > 0) return n;
  }
  return Platform.numberOfProcessors;
}

Future<String> _py(List<String> args, String log, {int? threads}) async {
  final result = await Process.run(
    _python,
    args,
    workingDirectory: 'tool/reference',
    environment: {
      'PYTHONUTF8': '1',
      'PYTHONDONTWRITEBYTECODE': '1',
      'HF_HUB_OFFLINE': '1',
      'LAYA_DEVICE': 'cpu',
      'LAYA_THREADS': ?threads?.toString(),
    },
    stdoutEncoding: utf8,
    stderrEncoding: utf8,
  );
  File(log)
    ..createSync(recursive: true)
    ..writeAsStringSync('${result.stderr}');
  if (result.exitCode != 0) {
    throw ProcessException(
      _python,
      args,
      'exit ${result.exitCode}; see $log',
      result.exitCode,
    );
  }
  return result.stdout as String;
}

Future<void> _fixtures(int threads) async {
  for (final name in _checkpoints) {
    final out = await _py(
      ['serve_parity.py', 'fixture', name],
      '.tooling/logs/parity_ref_$name.txt',
      threads: threads,
    );
    final path = 'test/fixtures/parity_$name.json';
    File(path).writeAsStringSync(out);
    final decoded = jsonDecode(out) as Map<String, Object?>;
    stdout.writeln(
      'wrote $path: ${(decoded['cases'] as List).length} cases, '
      '${(decoded['skipped'] as List).length} skipped',
    );
  }
}

/// The body laya-serve was sent for [_benchRequest] pinned to [name], from its
/// fixture, so both sides time identical bytes.
String _benchBody(String name) {
  final fixture =
      jsonDecode(File('test/fixtures/parity_$name.json').readAsStringSync())
          as Map<String, Object?>;
  final hit = (fixture['cases'] as List)
      .cast<Map<String, Object?>>()
      .firstWhere(
        (c) => c['request'] == _benchRequest && c['variant'] == 'model',
      );
  return hit['body'] as String;
}

double _median(List<double> xs) {
  final s = [...xs]..sort();
  final m = s.length ~/ 2;
  return s.length.isOdd ? s[m] : (s[m - 1] + s[m]) / 2;
}

/// Times the Dart server on [name] in this process; run in a child per
/// checkpoint so the peak working set is that checkpoint's alone.
Future<Map<String, Object?>> _benchDart(String name, int threads) async {
  final engine = LayaEngine(
    loader: LayaEngine.fileLoader('models', threads: threads),
  );
  final handler = createHandler(engine, const ServerConfig());
  final body = _benchBody(name);
  Future<void> post() async {
    final resp = await handler(
      Request(
        'POST',
        Uri.parse('http://localhost/v1/systemone'),
        body: body,
        headers: {'content-type': 'application/json'},
      ),
    );
    final text = await resp.readAsString();
    if (resp.statusCode != 200) throw StateError('$name: $text');
  }

  await post();
  final samples = <double>[];
  for (var i = 0; i < _benchRuns; i++) {
    final sw = Stopwatch()..start();
    await post();
    samples.add(sw.elapsedMicroseconds / 1000);
  }
  await engine.close();
  return {
    'checkpoint': name,
    'request': _benchRequest,
    'n': _benchRuns,
    'threads': threads,
    'median_ms': _median(samples),
    'samples_ms': samples,
    'peak_rss_bytes': ProcessInfo.maxRss,
  };
}

Future<void> _bench(int threads) async {
  final results = <String, Object?>{'threads': threads};
  for (final name in _checkpoints) {
    final py =
        (jsonDecode(
              await _py(
                [
                  'serve_parity.py',
                  'bench',
                  name,
                  '$_benchRequest',
                  '$_benchRuns',
                ],
                '.tooling/logs/parity_bench_py_$name.txt',
                threads: threads,
              ),
            )
            as Map<String, Object?>);
    final child = await Process.run(Platform.resolvedExecutable, [
      'run',
      'tool/parity.dart',
      'bench-dart',
      name,
      '$threads',
    ], stdoutEncoding: utf8);
    File('.tooling/logs/parity_bench_dart_$name.txt')
      ..createSync(recursive: true)
      ..writeAsStringSync('${child.stdout}${child.stderr}');
    if (child.exitCode != 0) {
      throw StateError('Dart bench for $name failed: exit ${child.exitCode}');
    }
    final dart =
        jsonDecode(
              LineSplitter.split(
                child.stdout as String,
              ).lastWhere((l) => l.isNotEmpty),
            )
            as Map<String, Object?>;
    results[name] = {'laya_serve': py, 'laya_server': dart};
    String mb(Object? b) => '${((b as num) / (1 << 20)).round()} MiB';
    stdout.writeln(
      '$name: laya-serve ${(py['median_ms'] as num).toStringAsFixed(1)} ms '
      '(${mb(py['peak_rss_bytes'])}), laya_server '
      '${(dart['median_ms'] as num).toStringAsFixed(1)} ms '
      '(${mb(dart['peak_rss_bytes'])})',
    );
  }
  File('.tooling/logs/parity_latency.json')
    ..createSync(recursive: true)
    ..writeAsStringSync(const JsonEncoder.withIndent('  ').convert(results));
}

Future<void> main(List<String> args) async {
  if (args.isNotEmpty && args.first == 'bench-dart') {
    final result = await _benchDart(args[1], int.parse(args[2]));
    stdout.writeln(jsonEncode(result));
    return;
  }
  final threads = await _cores();
  stdout.writeln('threads: $threads');
  final all = args.isEmpty;
  if (all || args.contains('fixtures')) await _fixtures(threads);
  if (all || args.contains('bench')) await _bench(threads);
}
