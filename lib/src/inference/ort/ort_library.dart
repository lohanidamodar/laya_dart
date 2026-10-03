// Loads the ONNX Runtime shared library and holds the per-isolate runtime.
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'ort_api.dart';
import 'ort_session.dart';

/// Thrown for any failure reported by ONNX Runtime; [message] is ORT's text.
///
/// Failures to find or load the library itself use [code] `-1`.
class OrtException implements Exception {
  /// An exception with ORT's error [code] and [message].
  OrtException(this.code, this.message);

  /// The `OrtErrorCode`, or `-1` when the library could not be loaded.
  final int code;

  /// ORT's error text, or an explanation of why the library did not load.
  final String message;

  @override
  String toString() => 'OrtException($code): $message';
}

const _fixHint =
    'Set LAYA_ORT_LIB to the onnxruntime shared library, version 1.30.x';

/// `OrtGetApiBase` from the library the build hook bundles.
@Native<Pointer<OrtApiBase> Function()>(
  symbol: 'OrtGetApiBase',
  assetId: 'package:laya_server/onnxruntime',
)
external Pointer<OrtApiBase> _bundledOrtGetApiBase();

/// Returns ONNX Runtime's `OrtApiBase`, from [explicitPath], else the
/// `LAYA_ORT_LIB` environment variable, else the bundled native asset.
///
/// Nothing else is called, so a caller can read `GetVersionString` before
/// `GetApi`. Throws [OrtException] when no library can be loaded.
Pointer<OrtApiBase> loadOrtApiBase({String? explicitPath}) {
  final path = _libraryPath(explicitPath);
  if (path == null) {
    try {
      return _bundledOrtGetApiBase();
    } on ArgumentError catch (e) {
      throw OrtException(
        -1,
        'the bundled ONNX Runtime is not available (${e.message}). '
        'This build target may have no prebuilt library. $_fixHint',
      );
    }
  }
  final DynamicLibrary library;
  try {
    library = DynamicLibrary.open(path);
  } on ArgumentError catch (e) {
    throw OrtException(
      -1,
      'could not load ONNX Runtime from $path: ${e.message}. $_fixHint',
    );
  }
  try {
    return library.lookupFunction<
      Pointer<OrtApiBase> Function(),
      Pointer<OrtApiBase> Function()
    >('OrtGetApiBase')();
  } on ArgumentError catch (e) {
    throw OrtException(
      -1,
      '$path is not ONNX Runtime (no OrtGetApiBase: ${e.message}). $_fixHint',
    );
  }
}

String? _libraryPath(String? explicitPath) {
  final path = explicitPath ?? Platform.environment['LAYA_ORT_LIB'];
  return path == null || path.isEmpty ? null : path;
}

final _runtimes = <String, OrtRuntime>{};

/// Loads the ORT shared library: [explicitPath] (`LAYA_ORT_LIB`) or the
/// bundled native asset.
///
/// One runtime is kept per library per isolate, so repeated calls are cheap.
/// Throws [OrtException] with an actionable message when neither is available
/// or the library does not provide the C API version this binding needs.
OrtRuntime loadOrtRuntime({String? explicitPath}) {
  final path = _libraryPath(explicitPath);
  final cached = _runtimes[path ?? ''];
  if (cached != null) return cached;

  final base = loadOrtApiBase(explicitPath: path);
  final version = base.ref.GetVersionString
      .asFunction<Pointer<Utf8> Function()>()()
      .toDartString();
  final api = base.ref.GetApi.asFunction<Pointer<OrtApi> Function(int)>()(
    ortApiVersion,
  );
  if (api == nullptr) {
    throw OrtException(
      -1,
      'ONNX Runtime $version${path == null ? '' : ' at $path'} does not '
      'provide C API version $ortApiVersion. $_fixHint',
    );
  }
  final runtime = OrtRuntime._(OrtFunctions(api), version);
  _runtimes[path ?? ''] = runtime;
  return runtime;
}

/// The loaded runtime; one per isolate.
final class OrtRuntime {
  OrtRuntime._(this.functions, this.version) {
    env = using((arena) {
      final out = arena<Pointer<OrtEnv>>();
      check(
        functions.CreateEnv(
          ortLoggingLevelWarning,
          'laya'.toNativeUtf8(allocator: arena),
          out,
        ),
      );
      return out.value;
    });
  }

  /// The bound C API functions. Low level: callers own every handle they make.
  final OrtFunctions functions;

  /// The library's version string, e.g. "1.30.0".
  final String version;

  /// The `OrtEnv` every session of this runtime shares.
  ///
  /// It is never released: sessions (and their finalizers) may outlive any
  /// point at which it could be.
  late final Pointer<OrtEnv> env;

  /// Releases a session handle when its Dart wrapper is collected unclosed.
  late final NativeFinalizer sessionFinalizer = NativeFinalizer(
    functions.ReleaseSessionPointer.cast(),
  );

  /// Throws [OrtException] with ORT's code and message when [status] is a
  /// failure, releasing it either way. `nullptr` is success.
  void check(Pointer<OrtStatus> status) {
    if (status == nullptr) return;
    try {
      throw OrtException(
        functions.GetErrorCode(status),
        functions.GetErrorMessage(status).toDartString(),
      );
    } finally {
      functions.ReleaseStatus(status);
    }
  }

  /// Opens the ONNX graph at [modelPath], optimized with `ORT_ENABLE_ALL` and,
  /// when given, [intraOpThreads] intra-op threads.
  ///
  /// Throws [OrtException] with ORT's text when the model cannot be loaded.
  OrtSession openSession(String modelPath, {int? intraOpThreads}) =>
      OrtSession.open(this, modelPath, intraOpThreads: intraOpThreads);
}
