// Provides the ONNX Runtime shared library for the build target as the code
// asset `package:laya_server/onnxruntime`.
//
// The pinned release archive (hook/ort_release.dart) is downloaded once into
// the hook's shared output directory, verified by sha256, and only the shared
// library is extracted from it. Nothing is written outside that directory.
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

import 'ort_fetch.dart';
import 'ort_release.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    final os = code.targetOS;
    final arch = code.targetArchitecture;
    final pin = ortRelease[(os.name, arch.name)];
    if (pin == null) {
      final supported = ortRelease.keys
          .map((k) => '${k.$1}-${k.$2}')
          .join(', ');
      throw BuildError(
        message:
            'laya_server bundles ONNX Runtime $ortVersion only for '
            '$supported; ${os.name}-${arch.name} is not one of them. Build '
            'onnxruntime $ortVersion for this target and set LAYA_ORT_LIB to '
            'its shared library at run time.',
      );
    }

    final url = Uri.parse(pin.url);
    final archive = File.fromUri(
      input.outputDirectoryShared.resolve(url.pathSegments.last),
    );
    try {
      await downloadVerified(url, archive, pin.sha256);
    } on Exception catch (e, st) {
      throw InfraError(
        message: 'could not download ONNX Runtime from $url: $e',
        wrappedException: e,
        wrappedTrace: st,
      );
    }

    // A per-digest directory, so a repin never reuses a stale library.
    final lib = File.fromUri(
      input.outputDirectoryShared.resolve(
        'onnxruntime-${pin.sha256.substring(0, 16)}/'
        '${os.dylibFileName('onnxruntime')}',
      ),
    );
    if (!lib.existsSync()) extractEntry(archive, pin.lib, lib);

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'onnxruntime',
        linkMode: DynamicLoadingBundled(),
        file: lib.uri,
      ),
    );
    output.dependencies.add(input.packageRoot.resolve('hook/ort_release.dart'));
  });
}
