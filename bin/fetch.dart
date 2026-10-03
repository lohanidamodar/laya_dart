// Downloads and verifies Laya checkpoints.
//
//   dart run laya_server:fetch [--dir models] [--variant fp32|fp16]
//       [--no-verify] <english|multilingual|typed-decisions|all>...
//
// Exit codes: 0 success, 1 failure (one sentence on stderr), 64 usage error.
import 'dart:io';

import 'package:laya_server/src/store/fetch_command.dart';

Future<void> main(List<String> args) async {
  exitCode = await runFetch(args, out: stdout, err: stderr);
}
