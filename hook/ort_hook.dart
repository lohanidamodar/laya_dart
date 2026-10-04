// The logic of hook/build.dart, separate so tests can run it without the
// process exit that `build()` performs on a HookError.
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

import 'ort_fetch.dart';
import 'ort_release.dart';

/// The user-define naming an ONNX Runtime library to bundle instead of
/// downloading one.
const ortLibDefine = 'ort_lib';

/// The user-define that, set to `false`, bundles no ONNX Runtime at all.
const bundleOrtDefine = 'bundle_ort';

/// How to build offline or with one's own ONNX Runtime, for error messages.
///
/// Build hooks run with a filtered environment that drops LAYA_ORT_LIB, so it
/// cannot steer the hook; it is only read at run time.
const ortOptOutHint =
    'To build offline or with your own ONNX Runtime $ortVersion, add to your '
    "app's pubspec.yaml `hooks: user_defines: laya_server: $ortLibDefine: "
    '<path to the onnxruntime shared library>` to bundle that file, or '
    '`$bundleOrtDefine: false` to bundle nothing and set LAYA_ORT_LIB at run '
    'time.';

/// Downloads [url] to [dest] and checks its sha256; [downloadVerified] by
/// default.
typedef OrtDownloader =
    Future<File> Function(Uri url, File dest, String expectedSha256);

/// Adds the `onnxruntime` code asset for the target to [output].
///
/// In order: `bundle_ort: false` adds nothing; `ort_lib: <path>` bundles that
/// file; otherwise the pinned archive is fetched with [download] (only when
/// the shared output directory lacks a verified copy) and its library
/// extracted. Throws [BuildError] or [InfraError] with [ortOptOutHint].
Future<void> buildOrt(
  BuildInput input,
  BuildOutputBuilder output, {
  OrtDownloader download = downloadVerified,
}) async {
  if (!input.config.buildCodeAssets) return;
  final bundle = input.userDefines[bundleOrtDefine];
  if (bundle == false || bundle == 'false') return;

  final code = input.config.code;
  final os = code.targetOS;
  final arch = code.targetArchitecture;

  final supplied = input.userDefines.path(ortLibDefine);
  if (supplied != null) {
    output.dependencies.add(supplied);
    if (!File.fromUri(supplied).existsSync()) {
      throw BuildError(
        message:
            'the $ortLibDefine user-define points at '
            '${supplied.toFilePath()}, which does not exist. $ortOptOutHint',
      );
    }
    _emit(input, output, supplied);
    return;
  }

  final pin = ortRelease[(os.name, arch.name)];
  if (pin == null) {
    final supported = ortRelease.keys.map((k) => '${k.$1}-${k.$2}').join(', ');
    throw BuildError(
      message:
          'laya_server bundles ONNX Runtime $ortVersion only for $supported, '
          'not ${os.name}-${arch.name}. $ortOptOutHint',
    );
  }

  final url = Uri.parse(pin.url);
  final archive = File.fromUri(
    input.outputDirectoryShared.resolve(url.pathSegments.last),
  );
  // A per-digest directory, so a repin never reuses a stale library.
  final lib = File.fromUri(
    input.outputDirectoryShared.resolve(
      'onnxruntime-${pin.sha256.substring(0, 16)}/'
      '${os.dylibFileName('onnxruntime')}',
    ),
  );
  if (!lib.existsSync()) {
    try {
      await download(url, archive, pin.sha256);
    } on Exception catch (e, st) {
      throw InfraError(
        message:
            'could not download ONNX Runtime from $url: $e. $ortOptOutHint',
        wrappedException: e,
        wrappedTrace: st,
      );
    }
    try {
      extractEntry(archive, pin.lib, lib);
    } on Exception catch (e, st) {
      throw BuildError(
        message:
            'could not extract ${pin.lib} from ${archive.path}: $e. '
            '$ortOptOutHint',
        wrappedException: e,
        wrappedTrace: st,
      );
    }
  }
  _emit(input, output, lib.uri);
  output.dependencies.add(input.packageRoot.resolve('hook/ort_release.dart'));
}

void _emit(BuildInput input, BuildOutputBuilder output, Uri file) {
  output.assets.code.add(
    CodeAsset(
      package: input.packageName,
      name: 'onnxruntime',
      linkMode: DynamicLoadingBundled(),
      file: file,
    ),
  );
}
