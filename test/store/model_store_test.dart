import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:laya_server/laya_server.dart' show LayaEngine, PredictOptions;
import 'package:laya_server/src/store/fetch_command.dart';
import 'package:laya_server/src/store/model_store.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import '../support/checkpoints.dart';

const _repo = 'o/r';
const _rev = 'abc123';

final _contents = <String, List<int>>{
  'tokenizer.json': utf8.encode('{"tok":1}'),
  'config.json': utf8.encode('{"cfg":2}'),
  'onnx/model.onnx': utf8.encode('graph32'),
  'onnx/model.onnx_data': List<int>.generate(1000, (i) => i % 251),
  'onnx/model_fp16.onnx': utf8.encode('graph16'),
  'onnx/model_fp16.onnx_data': List<int>.generate(400, (i) => i % 97),
};

PinnedFile _pin(String path) => PinnedFile(
  _repo,
  _rev,
  path,
  _contents[path]!.length,
  sha256.convert(_contents[path]!).toString(),
);

final _pins = {
  'english': [for (final path in _contents.keys) _pin(path)],
};

/// What the fake hub saw.
class _Seen {
  _Seen(this.path, this.range, this.auth);
  final String path;
  final String? range;
  final String? auth;
}

/// A fake hub serving [_contents], with switches for each behaviour under test.
class _Hub {
  late HttpServer server;
  final seen = <_Seen>[];
  bool ignoreRange = false;
  Map<String, List<int>> override = {};

  /// When set, `onnx/model.onnx_data` redirects to this URL.
  Uri? redirectData;

  Uri get base => Uri.parse('http://127.0.0.1:${server.port}');

  Future<void> start() async {
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((req) async {
      final prefix = '/$_repo/resolve/$_rev/';
      final path = req.uri.path.startsWith(prefix)
          ? req.uri.path.substring(prefix.length)
          : req.uri.path.substring(1);
      seen.add(
        _Seen(
          path,
          req.headers.value('range'),
          req.headers.value('authorization'),
        ),
      );
      final redirect = redirectData;
      if (redirect != null && path == 'onnx/model.onnx_data') {
        req.response
          ..statusCode = 302
          ..headers.set('location', redirect.toString());
        await req.response.close();
        return;
      }
      final body = override[path] ?? _contents[path];
      if (body == null) {
        req.response.statusCode = 404;
        await req.response.close();
        return;
      }
      final range = req.headers.value('range');
      if (range != null && !ignoreRange) {
        final start = int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
        req.response
          ..statusCode = 206
          ..headers.set(
            'content-range',
            'bytes $start-${body.length - 1}/${body.length}',
          )
          ..add(body.sublist(start));
      } else {
        req.response.add(body);
      }
      await req.response.close();
    });
  }

  Future<void> stop() => server.close(force: true);
}

void main() {
  late Directory tmp;
  late _Hub hub;

  setUp(() async {
    tmp = Directory.systemTemp.createTempSync('model_store_test');
    hub = _Hub();
    await hub.start();
  });
  tearDown(() async {
    await hub.stop();
    tmp.deleteSync(recursive: true);
  });

  ModelStore store({String? token, Map<String, String>? env}) => ModelStore(
    tmp.path,
    hubBase: hub.base,
    token: token,
    environment: env ?? const {},
    pins: _pins,
  );

  String local(String path) =>
      p.joinAll([tmp.path, 'english', ...path.split('/')]);

  test('filesFor picks the variant pair plus tokenizer and config', () {
    final s = store();
    expect(s.filesFor('english').map((f) => f.path), [
      'tokenizer.json',
      'config.json',
      'onnx/model.onnx',
      'onnx/model.onnx_data',
    ]);
    expect(s.filesFor('english', variant: 'fp16').map((f) => f.path), [
      'tokenizer.json',
      'config.json',
      'onnx/model_fp16.onnx',
      'onnx/model_fp16.onnx_data',
    ]);
    expect(() => s.filesFor('nope'), throwsArgumentError);
    expect(() => s.filesFor('english', variant: 'int8'), throwsArgumentError);
  });

  test('the built-in pins cover every checkpoint and variant', () {
    final s = ModelStore(tmp.path);
    for (final name in ['english', 'multilingual', 'typed-decisions']) {
      for (final v in ['fp32', 'fp16']) {
        final files = s.filesFor(name, variant: v);
        expect(files, hasLength(4));
        for (final f in files) {
          expect(f.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
          expect(f.size, greaterThan(0));
        }
      }
    }
  });

  test('a fresh install downloads, verifies and reports progress', () async {
    final s = store();
    expect(s.isInstalled('english'), isFalse);
    final progress = <String, int>{};
    await s.install(
      'english',
      onProgress: (path, done, total) {
        progress[path] = done;
        expect(done, lessThanOrEqualTo(total));
      },
    );
    expect(s.isInstalled('english'), isTrue);
    expect(s.isInstalled('english', variant: 'fp16'), isFalse);
    for (final path in [
      'tokenizer.json',
      'config.json',
      'onnx/model.onnx',
      'onnx/model.onnx_data',
    ]) {
      expect(File(local(path)).readAsBytesSync(), _contents[path]);
      expect(File('${local(path)}.part').existsSync(), isFalse);
      expect(progress[path], _contents[path]!.length);
    }
    expect(await s.verify('english'), isEmpty);
    expect(hub.seen.every((r) => r.range == null), isTrue);
    // A second install downloads nothing.
    hub.seen.clear();
    await s.install('english');
    expect(hub.seen, isEmpty);
  });

  test('the fp16 variant installs its own pair', () async {
    await store().install('english', variant: 'fp16');
    expect(File(local('onnx/model_fp16.onnx_data')).existsSync(), isTrue);
    expect(File(local('onnx/model.onnx_data')).existsSync(), isFalse);
  });

  test('a partial file resumes with a Range request answered by 206', () async {
    final data = _contents['onnx/model.onnx_data']!;
    File('${local('onnx/model.onnx_data')}.part')
      ..createSync(recursive: true)
      ..writeAsBytesSync(data.sublist(0, 300));
    await store().install('english');
    expect(File(local('onnx/model.onnx_data')).readAsBytesSync(), data);
    final r = hub.seen.singleWhere((r) => r.path == 'onnx/model.onnx_data');
    expect(r.range, 'bytes=300-');
  });

  test('a 200 to a Range request restarts from zero', () async {
    hub.ignoreRange = true;
    final data = _contents['onnx/model.onnx_data']!;
    File('${local('onnx/model.onnx_data')}.part')
      ..createSync(recursive: true)
      ..writeAsBytesSync(data.sublist(0, 300));
    await store().install('english');
    expect(File(local('onnx/model.onnx_data')).readAsBytesSync(), data);
    expect(
      hub.seen.singleWhere((r) => r.path == 'onnx/model.onnx_data').range,
      'bytes=300-',
    );
  });

  test('a sha mismatch throws a StateError naming the path and leaves '
      'no file', () async {
    final bad = List<int>.of(_contents['onnx/model.onnx_data']!)..[5] ^= 0xff;
    hub.override['onnx/model.onnx_data'] = bad;
    final target = local('onnx/model.onnx_data');
    await expectLater(
      store().install('english'),
      throwsA(
        isA<StateError>().having((e) => e.message, 'message', contains(target)),
      ),
    );
    expect(File(target).existsSync(), isFalse);
    expect(File('$target.part').existsSync(), isFalse);
    expect(store().isInstalled('english'), isFalse);
  });

  test('an HTTP error throws and leaves no final file', () async {
    final pins = {
      'english': [
        PinnedFile(
          _repo,
          _rev,
          'missing.json',
          3,
          sha256.convert([1]).toString(),
        ),
      ],
    };
    final s = ModelStore(tmp.path, hubBase: hub.base, pins: pins);
    await expectLater(s.install('english'), throwsA(isA<HttpException>()));
    expect(File(local('missing.json')).existsSync(), isFalse);
  });

  test(
    'a redirect to another host keeps Range and drops Authorization',
    () async {
      final cdn = _Hub();
      await cdn.start();
      addTearDown(cdn.stop);
      // 127.0.0.1 (the hub) and localhost (the CDN) are different hosts.
      hub.redirectData = Uri.parse(
        'http://localhost:${cdn.server.port}/cdn/blob',
      );
      final data = _contents['onnx/model.onnx_data']!;
      cdn.override['cdn/blob'] = data;
      File('${local('onnx/model.onnx_data')}.part')
        ..createSync(recursive: true)
        ..writeAsBytesSync(data.sublist(0, 300));
      await store(token: 'secret-token').install('english');
      expect(File(local('onnx/model.onnx_data')).readAsBytesSync(), data);
      final first = hub.seen.singleWhere(
        (r) => r.path == 'onnx/model.onnx_data',
      );
      expect(first.auth, 'Bearer secret-token');
      expect(first.range, 'bytes=300-');
      final hop = cdn.seen.single;
      expect(hop.path, 'cdn/blob');
      expect(hop.range, 'bytes=300-');
      expect(hop.auth, isNull);
      // The token goes to the hub host for ordinary files too.
      expect(
        hub.seen.where((r) => r.path == 'config.json').single.auth,
        'Bearer secret-token',
      );
    },
  );

  test('no token means no Authorization header', () async {
    await store().install('english');
    expect(hub.seen.every((r) => r.auth == null), isTrue);
  });

  test(
    'isInstalled checks sizes and verify reports a byte-flipped file',
    () async {
      final s = store();
      await s.install('english');
      final target = File(local('onnx/model.onnx_data'));
      final bytes = target.readAsBytesSync()..[10] ^= 0xff;
      target.writeAsBytesSync(bytes);
      expect(s.isInstalled('english'), isTrue); // same size: the fast check
      expect(await s.verify('english'), [target.path]);
      target.writeAsBytesSync(bytes.sublist(0, 999));
      expect(s.isInstalled('english'), isFalse);
      target.deleteSync();
      expect(await s.verify('english'), [target.path]);
    },
  );

  test('HF_TOKEN from the environment is the fallback token', () async {
    await store(env: {'HF_TOKEN': 'from-env'}).install('english');
    expect(hub.seen.every((r) => r.auth == 'Bearer from-env'), isTrue);
  });

  test('an explicit token beats the environment', () async {
    await store(
      token: 'explicit',
      env: {'HF_TOKEN': 'from-env'},
    ).install('english');
    expect(hub.seen.every((r) => r.auth == 'Bearer explicit'), isTrue);
  });

  group('fetch command', () {
    Future<(int, String, String)> fetch(List<String> args) async {
      final out = _Sink();
      final err = _Sink();
      final code = await runFetch(
        args,
        out: out,
        err: err,
        storeFor: (dir) =>
            ModelStore(dir, hubBase: hub.base, environment: {}, pins: _pins),
      );
      return (code, out.text, err.text);
    }

    test('installs, verifies and exits 0', () async {
      final (code, out, err) = await fetch(['--dir', tmp.path, 'english']);
      expect(code, 0, reason: err);
      expect(out, contains('english: installed and verified.'));
      expect(File(local('onnx/model.onnx_data')).existsSync(), isTrue);
    });

    test('a same-size corrupt file fails verification and names it', () async {
      final target = File(local('onnx/model.onnx_data'))
        ..createSync(recursive: true);
      final bytes = List<int>.of(_contents['onnx/model.onnx_data']!)..[7] ^= 1;
      target.writeAsBytesSync(bytes);
      final (code, _, err) = await fetch(['--dir', tmp.path, 'english']);
      expect(code, 1);
      expect(err, contains(target.path));
      expect(err, contains('sha256'));
    });

    test('--no-verify skips the hash', () async {
      File(local('onnx/model.onnx_data'))
        ..createSync(recursive: true)
        ..writeAsBytesSync(
          List<int>.of(_contents['onnx/model.onnx_data']!)..[7] ^= 1,
        );
      final (code, _, _) = await fetch([
        '--dir',
        tmp.path,
        '--no-verify',
        'english',
      ]);
      expect(code, 0);
    });

    test('an unknown checkpoint exits 64 and lists the valid names', () async {
      final (code, _, err) = await fetch(['bogus']);
      expect(code, 64);
      expect(err, contains('english, multilingual, typed-decisions'));
    });
  });

  group(
    'layout matches LayaEngine.fileLoader',
    skip: tokenizerSkip('english'),
    () {
      // The store installs real files (the stub graph, a real tokenizer and a
      // Laya config) and the engine's own loader reads them back.
      Map<String, List<int>> realFiles() {
        final configs =
            (jsonDecode(File('test/fixtures/decode.json').readAsStringSync())
                    as Map)['configs']
                as Map;
        final stub = File('test/fixtures/ort/laya_stub.onnx').readAsBytesSync();
        return {
          'tokenizer.json': File(tokenizerPath('english')).readAsBytesSync(),
          'config.json': utf8.encode(jsonEncode(configs['english'])),
          'onnx/model.onnx': stub,
          'onnx/model.onnx_data': const [],
          'onnx/model_fp16.onnx': stub,
          'onnx/model_fp16.onnx_data': const [],
        };
      }

      ModelStore realStore(Map<String, List<int>> files) {
        hub.override.addAll(files);
        return ModelStore(
          tmp.path,
          hubBase: hub.base,
          environment: {},
          pins: {
            'english': [
              for (final e in files.entries)
                PinnedFile(
                  _repo,
                  _rev,
                  e.key,
                  e.value.length,
                  sha256.convert(e.value).toString(),
                ),
            ],
          },
        );
      }

      test('the four files fileLoader checks exist after install', () async {
        await realStore(realFiles()).install('english');
        for (final path in [
          'tokenizer.json',
          'config.json',
          'onnx/model.onnx',
          'onnx/model.onnx_data',
        ]) {
          expect(File(local(path)).existsSync(), isTrue, reason: path);
        }
      });

      test(
        'an installed checkpoint loads and predicts through ONNX Runtime',
        tags: ['ort'],
        () async {
          final s = realStore(realFiles());
          await s.install('english');
          await s.install('english', variant: 'fp16');
          for (final variant in ['fp32', 'fp16']) {
            final engine = LayaEngine(
              loader: LayaEngine.fileLoader(
                tmp.path,
                variant: variant,
                threads: 1,
              ),
            );
            addTearDown(engine.close);
            final r = await engine.predict(
              'I was charged twice for my March invoice.',
              {
                'dept': {
                  'type': 'choice',
                  'instructions': 'Which department?',
                  'criteria': ['support', 'sales', 'billing'],
                },
              },
              const PredictOptions(),
            );
            expect(
              ((r['answers']! as Map)['dept']! as Map)['choice'],
              'billing',
            );
          }
        },
      );
    },
  );
}

/// An [IOSink] collecting what is written to it.
class _Sink implements IOSink {
  final _buf = StringBuffer();
  String get text => _buf.toString();

  @override
  Encoding encoding = utf8;
  @override
  void write(Object? object) => _buf.write(object);
  @override
  void writeln([Object? object = '']) => _buf.writeln(object);
  @override
  void writeAll(Iterable<Object?> objects, [String separator = '']) =>
      _buf.writeAll(objects, separator);
  @override
  void writeCharCode(int charCode) => _buf.writeCharCode(charCode);
  @override
  void add(List<int> data) => _buf.write(utf8.decode(data));
  @override
  void addError(Object error, [StackTrace? stackTrace]) {}
  @override
  Future<void> addStream(Stream<List<int>> stream) async {}
  @override
  Future<void> flush() async {}
  @override
  Future<void> close() async {}
  @override
  Future<void> get done async {}
}
