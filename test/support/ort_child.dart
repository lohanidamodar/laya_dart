// A child process for ort_session_test.dart, for checks that need their own
// process: the environment (LAYA_ORT_LIB) is fixed at startup, and RSS is
// process-wide, so other test suites running alongside would skew it.
//
// Usage: dart --packages=<config> test/support/ort_child.dart version|rss
import 'dart:io';
import 'dart:typed_data';

import 'package:laya_server/src/inference/ort/ort_library.dart';
import 'package:laya_server/src/inference/ort/ort_session.dart';

void main(List<String> args) {
  final ort = loadOrtRuntime();
  switch (args.single) {
    case 'version':
      stdout.writeln(ort.version);
    case 'rss':
      final s = ort.openSession(
        'test/fixtures/ort/echo.onnx',
        intraOpThreads: 1,
      );
      final inputs = {
        'ids': OrtInt64Input(Int64List.fromList([1, 2, 3, 4, 5, 6]), [2, 3]),
        'mask': OrtBoolInput(Uint8List.fromList([1, 0, 0, 1]), [2, 2]),
      };
      // Warm up so allocator pools and lazy ORT state are in the baseline.
      for (var i = 0; i < 50; i++) {
        s.run(inputs);
      }
      final before = ProcessInfo.currentRss;
      for (var i = 0; i < 2000; i++) {
        s.run(inputs);
      }
      stdout.writeln('$before ${ProcessInfo.currentRss}');
      s.close();
  }
}
