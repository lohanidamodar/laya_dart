// Provides the ONNX Runtime shared library for the build target as the code
// asset `package:laya_server/onnxruntime`.
//
// By default the pinned release archive (hook/ort_release.dart) is downloaded
// once into the hook's shared output directory, verified by sha256, and only
// the shared library is extracted from it. Nothing is written outside that
// directory. The app's pubspec can instead supply a library or opt out; see
// hook/ort_hook.dart.
import 'package:hooks/hooks.dart';

import 'ort_hook.dart';

void main(List<String> args) async {
  await build(args, buildOrt);
}
