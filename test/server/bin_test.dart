// bin/laya_server.dart as a subprocess, against a checkpoint installed with
// the stub graph (the layout ModelStore installs).
@Tags(['ort'])
@Timeout(Duration(minutes: 4))
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import '../support/checkpoints.dart';

Future<int> _freePort() async {
  final socket = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
  final port = socket.port;
  await socket.close();
  return port;
}

/// The ONNX Runtime library the build hook extracted, under .dart_tool/.
String _bundledLibrary() {
  final name = Platform.isWindows
      ? 'onnxruntime.dll'
      : Platform.isMacOS
      ? 'libonnxruntime.dylib'
      : 'libonnxruntime.so';
  return Directory('.dart_tool')
      .listSync(recursive: true, followLinks: false)
      .whereType<File>()
      .firstWhere((f) => p.basename(f.path) == name)
      .absolute
      .path;
}

/// Starts the server; its output is collected into [log].
///
/// Not `dart run`: that re-bundles the native assets, and on Windows the
/// library is locked by the test isolates already using it. The child gets
/// the bundled library through LAYA_ORT_LIB instead, as ort_session_test does.
Future<Process> _start(Map<String, String> env, StringBuffer log) async {
  final process = await Process.start(
    Platform.resolvedExecutable,
    [
      '--packages=${p.absolute('.dart_tool/package_config.json')}',
      'bin/laya_server.dart',
    ],
    environment: {'LAYA_ORT_LIB': _bundledLibrary(), ...env},
  );
  process.stdout.transform(utf8.decoder).listen(log.write);
  process.stderr.transform(utf8.decoder).listen(log.write);
  return process;
}

void main() {
  group('bin/laya_server.dart', skip: tokenizerSkip('english'), () {
    late Directory models;
    setUpAll(() {
      models = Directory.systemTemp.createTempSync('laya_bin_');
      final dir = p.join(models.path, 'english');
      final configs =
          (jsonDecode(File('test/fixtures/decode.json').readAsStringSync())
                  as Map)['configs']
              as Map;
      Directory(p.join(dir, 'onnx')).createSync(recursive: true);
      File(
        p.join(dir, 'tokenizer.json'),
      ).writeAsBytesSync(File(tokenizerPath('english')).readAsBytesSync());
      File(
        p.join(dir, 'config.json'),
      ).writeAsStringSync(jsonEncode(configs['english']));
      File(p.join(dir, 'onnx', 'model.onnx')).writeAsBytesSync(
        File('test/fixtures/ort/laya_stub.onnx').readAsBytesSync(),
      );
      File(p.join(dir, 'onnx', 'model.onnx_data')).writeAsBytesSync(const []);
    });
    tearDownAll(() => models.deleteSync(recursive: true));

    test('serves a request, then exits on a signal', () async {
      final port = await _freePort();
      final log = StringBuffer();
      final process = await _start({
        'LAYA_HOST': '127.0.0.1',
        'LAYA_PORT': '$port',
        'LAYA_MODEL_DIR': models.path,
        'LAYA_MODELS': 'english',
        'LAYA_THREADS': '1',
      }, log);
      var exited = false;
      final exit = process.exitCode.then((code) {
        exited = true;
        return code;
      });
      addTearDown(() {
        if (!exited) process.kill(ProcessSignal.sigkill);
      });

      final base = Uri.parse('http://127.0.0.1:$port');
      final client = HttpClient();
      addTearDown(client.close);
      Future<(int, String)> call(
        String method,
        String path, [
        Object? body,
      ]) async {
        final request = await client.openUrl(method, base.resolve(path));
        if (body != null) {
          request.headers.contentType = ContentType.json;
          request.write(jsonEncode(body));
        }
        final response = await request.close();
        return (
          response.statusCode,
          await response.transform(utf8.decoder).join(),
        );
      }

      // Preloading happens before the port opens.
      final deadline = DateTime.now().add(const Duration(minutes: 3));
      (int, String)? health;
      while (health == null) {
        if (exited) fail('server exited early:\n$log');
        if (DateTime.now().isAfter(deadline)) fail('never listened:\n$log');
        try {
          health = await call('GET', '/health');
        } on SocketException {
          await Future<void>.delayed(const Duration(milliseconds: 250));
        }
      }
      expect(health.$1, 200);
      expect((jsonDecode(health.$2) as Map)['loaded'], ['english']);

      final (status, text) = await call('POST', '/v1/systemone', {
        'state': 'I was charged twice for my March invoice.',
        'questions': {
          'dept': {
            'type': 'choice',
            'instructions': 'Which department?',
            'criteria': ['support', 'sales', 'billing'],
          },
        },
      });
      expect(status, 200, reason: text);
      final answer =
          ((jsonDecode(text) as Map)['answers'] as Map)['dept'] as Map;
      expect(answer['choice'], 'billing');

      // Windows has no SIGTERM to send another process: kill() terminates it.
      process.kill(
        Platform.isWindows ? ProcessSignal.sigkill : ProcessSignal.sigterm,
      );
      final code = await exit.timeout(const Duration(seconds: 30));
      if (!Platform.isWindows) expect(code, 0, reason: '$log');
    });

    test(
      'a missing checkpoint fails at startup with the fetch command',
      () async {
        final log = StringBuffer();
        final process = await _start({
          'LAYA_PORT': '${await _freePort()}',
          'LAYA_MODEL_DIR': models.path,
          'LAYA_MODELS': 'multilingual',
        }, log);
        final code = await process.exitCode.timeout(const Duration(minutes: 3));
        expect(code, 2, reason: '$log');
        expect('$log', contains('dart run laya_server:fetch multilingual'));
      },
    );

    test('a bad LAYA_PORT fails at startup in one line', () async {
      final log = StringBuffer();
      final process = await _start({'LAYA_PORT': 'eighty'}, log);
      final code = await process.exitCode.timeout(const Duration(minutes: 3));
      expect(code, 2);
      expect(
        '$log'.trim(),
        endsWith("invalid LAYA_PORT 'eighty': must be an integer 1-65535"),
      );
    });
  });
}
