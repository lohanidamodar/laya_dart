@Tags(['ort'])
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:laya_server/src/inference/ort/ort_api.dart';
import 'package:laya_server/src/inference/ort/ort_library.dart';
import 'package:laya_server/src/inference/ort/ort_session.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

const _echo = 'test/fixtures/ort/echo.onnx';

Map<String, OrtInput> _echoInputs() => {
  'ids': OrtInt64Input(Int64List.fromList([1, 2, 3, 4, 5, 6]), [2, 3]),
  'mask': OrtBoolInput(Uint8List.fromList([1, 0, 0, 1]), [2, 2]),
};

/// The library the build hook extracted, found under .dart_tool/.
File _bundledLibrary() {
  final name = Platform.isWindows
      ? 'onnxruntime.dll'
      : Platform.isMacOS
      ? 'libonnxruntime.dylib'
      : 'libonnxruntime.so';
  final found = Directory('.dart_tool')
      .listSync(recursive: true, followLinks: false)
      .whereType<File>()
      .where((f) => p.basename(f.path) == name)
      .toList();
  if (found.isEmpty) fail('the build hook left no $name under .dart_tool/');
  return found.first;
}

/// Runs test/support/ort_child.dart with [mode] in its own process.
ProcessResult _child(String mode, Map<String, String> environment) =>
    Process.runSync(Platform.resolvedExecutable, [
      '--packages=${p.absolute('.dart_tool/package_config.json')}',
      'test/support/ort_child.dart',
      mode,
    ], environment: environment);

void main() {
  // First, and before anything calls GetApi: a struct-order mistake then
  // shows up as a version mismatch instead of a crash.
  group('OrtApiBase', () {
    test('GetVersionString before GetApi', () {
      final base = loadOrtApiBase();
      final version = base.ref.GetVersionString
          .asFunction<Pointer<Utf8> Function()>()()
          .toDartString();
      expect(version, '1.30.0');
      final api = base.ref.GetApi.asFunction<Pointer<OrtApi> Function(int)>()(
        ortApiVersion,
      );
      expect(api, isNot(nullptr));
    });
  });

  group('OrtRuntime', () {
    late OrtRuntime ort;
    setUpAll(() => ort = loadOrtRuntime());

    test('version is the pinned release', () => expect(ort.version, '1.30.0'));

    test('runs int64 and bool inputs', () {
      final s = ort.openSession(_echo, intraOpThreads: 1);
      addTearDown(s.close);
      expect(s.inputNames, ['ids', 'mask']);
      expect(s.outputNames, ['ids_f', 'mask_f']);
      final out = s.run(_echoInputs());
      expect(out['ids_f']!.shape, [2, 3]);
      expect(out['ids_f']!.data, [1, 2, 3, 4, 5, 6]);
      expect(out['mask_f']!.shape, [2, 2]);
      expect(out['mask_f']!.data, [1, 0, 0, 1]);
    });

    test('bad path raises OrtException with ORT text', () {
      expect(
        () => ort.openSession('missing.onnx'),
        throwsA(
          isA<OrtException>().having(
            (e) => e.message,
            'message',
            contains('missing.onnx'),
          ),
        ),
      );
    });

    test('explicit wrong library path explains how to fix it', () {
      expect(
        () => loadOrtRuntime(explicitPath: 'nope.dll'),
        throwsA(
          isA<OrtException>().having(
            (e) => e.message,
            'message',
            contains('LAYA_ORT_LIB'),
          ),
        ),
      );
    });

    test('an explicit path to a real library loads it', () {
      final lib = _bundledLibrary();
      final explicit = loadOrtRuntime(explicitPath: lib.path);
      expect(explicit.version, '1.30.0');
      final s = explicit.openSession(_echo, intraOpThreads: 1);
      addTearDown(s.close);
      expect(s.run(_echoInputs())['ids_f']!.data, [1, 2, 3, 4, 5, 6]);
    });

    test('LAYA_ORT_LIB is honoured', () {
      final lib = _bundledLibrary();
      final run = _child('version', {'LAYA_ORT_LIB': lib.path});
      expect(run.exitCode, 0, reason: '${run.stdout}${run.stderr}');
      expect((run.stdout as String).trim(), '1.30.0');

      final bad = _child('version', {
        'LAYA_ORT_LIB': 'no-such-onnxruntime.dll',
      });
      expect(bad.exitCode, isNot(0));
      expect('${bad.stderr}', contains('no-such-onnxruntime.dll'));
      expect('${bad.stderr}', contains('LAYA_ORT_LIB'));
    });

    test('a model path with a space and a non-ASCII character opens', () {
      final dir = Directory.systemTemp.createTempSync('laya_ort');
      addTearDown(() => dir.deleteSync(recursive: true));
      final copy = File(p.join(dir.path, 'tmp dé', 'echo.onnx'))
        ..parent.createSync()
        ..writeAsBytesSync(File(_echo).readAsBytesSync());
      final s = ort.openSession(copy.path, intraOpThreads: 1);
      addTearDown(s.close);
      expect(s.run(_echoInputs())['mask_f']!.data, [1, 0, 0, 1]);
    });

    test('an unknown input name raises OrtException', () {
      final s = ort.openSession(_echo, intraOpThreads: 1);
      addTearDown(s.close);
      expect(
        () => s.run({..._echoInputs(), 'nope': _echoInputs()['ids']!}),
        throwsA(isA<OrtException>()),
      );
    });

    test('shape and data length must agree', () {
      final s = ort.openSession(_echo, intraOpThreads: 1);
      addTearDown(s.close);
      expect(
        () => s.run({
          'ids': OrtInt64Input(Int64List.fromList([1, 2, 3]), [2, 3]),
          'mask': OrtBoolInput(Uint8List.fromList([1, 0, 0, 1]), [2, 2]),
        }),
        throwsArgumentError,
      );
    });

    test('close is idempotent and run after close throws', () {
      final s = ort.openSession(_echo, intraOpThreads: 1);
      s.close();
      s.close();
      expect(() => s.run(_echoInputs()), throwsStateError);
    });

    // Each of these calls lands on a different OrtApi slot; a wrong offset
    // calls the wrong function and fails (or crashes) here.
    test('struct offsets are wired to the right functions', () {
      final f = ort.functions;
      using((arena) {
        final info = arena<Pointer<OrtMemoryInfo>>();
        ort.check(
          f.CreateMemoryInfo(
            'Cpu'.toNativeUtf8(allocator: arena),
            ortArenaAllocator,
            0,
            ortMemTypeDefault,
            info,
          ),
        );
        try {
          final name = arena<Pointer<Utf8>>();
          ort.check(f.MemoryInfoGetName(info.value, name));
          expect(name.value.toDartString(), 'Cpu');
          final id = arena<Int>();
          ort.check(f.MemoryInfoGetId(info.value, id));
          expect(id.value, 0);
          final memType = arena<Int32>();
          ort.check(f.MemoryInfoGetMemType(info.value, memType));
          expect(memType.value, ortMemTypeDefault);

          final data = arena<Int64>(6);
          final shape = arena<Int64>(3)
            ..[0] = 1
            ..[1] = 2
            ..[2] = 3;
          final value = arena<Pointer<OrtValue>>();
          ort.check(
            f.CreateTensorWithDataAsOrtValue(
              info.value,
              data.cast(),
              6 * 8,
              shape,
              3,
              onnxTensorElementDataTypeInt64,
              value,
            ),
          );
          try {
            final isTensor = arena<Int>();
            ort.check(f.IsTensor(value.value, isTensor));
            expect(isTensor.value, 1);
            final ti = arena<Pointer<OrtTensorTypeAndShapeInfo>>();
            ort.check(f.GetTensorTypeAndShape(value.value, ti));
            try {
              final rank = arena<Size>();
              ort.check(f.GetDimensionsCount(ti.value, rank));
              expect(rank.value, 3);
              final dims = arena<Int64>(3);
              ort.check(f.GetDimensions(ti.value, dims, 3));
              expect(dims.asTypedList(3), [1, 2, 3]);
              final type = arena<Int32>();
              ort.check(f.GetTensorElementType(ti.value, type));
              expect(type.value, onnxTensorElementDataTypeInt64);
              final count = arena<Size>();
              ort.check(f.GetTensorShapeElementCount(ti.value, count));
              expect(count.value, 6);
            } finally {
              f.ReleaseTensorTypeAndShapeInfo(ti.value);
            }
          } finally {
            f.ReleaseValue(value.value);
          }
        } finally {
          f.ReleaseMemoryInfo(info.value);
        }
      });
    });

    test('CreateStatus round-trips code and message', () {
      final f = ort.functions;
      using((arena) {
        final status = f.CreateStatus(
          ortInvalidArgument,
          'from dart'.toNativeUtf8(allocator: arena),
        );
        expect(
          () => ort.check(status),
          throwsA(
            isA<OrtException>()
                .having((e) => e.code, 'code', ortInvalidArgument)
                .having((e) => e.message, 'message', 'from dart'),
          ),
        );
      });
    });

    // In a child process: RSS is process-wide, and other suites run alongside.
    test('2,000 runs do not grow RSS by more than 50 MB', () {
      final run = _child('rss', {'LAYA_ORT_LIB': _bundledLibrary().path});
      expect(run.exitCode, 0, reason: '${run.stdout}${run.stderr}');
      final [before, after] = (run.stdout as String)
          .trim()
          .split(' ')
          .map(int.parse)
          .toList();
      final grownMb = (after - before) / (1024 * 1024);
      print(
        'RSS before=${before ~/ 1024} KiB after=${after ~/ 1024} KiB '
        'grown=${grownMb.toStringAsFixed(2)} MB over 2000 runs',
      );
      expect(grownMb, lessThan(50));
    });
  });
}
