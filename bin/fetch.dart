// Downloads and verifies Laya checkpoints.
//
//   dart run laya_server:fetch [--dir models] [--variant fp32|fp16]
//       <english|multilingual|typed-decisions|all>...
//
// Exit codes: 0 success, 1 failure (one sentence on stderr), 64 usage error.
import 'dart:io';

import 'package:args/args.dart';
import 'package:laya_server/laya_server.dart' show CheckpointSpec, ModelStore;

const _usageError = 64;

String _mb(int bytes) => '${(bytes / (1 << 20)).toStringAsFixed(1)} MB';

Future<void> main(List<String> args) async {
  final parser = ArgParser()
    ..addOption('dir', defaultsTo: 'models', help: 'Where checkpoints install.')
    ..addOption(
      'variant',
      defaultsTo: 'fp32',
      allowed: ['fp32', 'fp16'],
      help: 'Which exported weights to fetch.',
    )
    ..addFlag('help', abbr: 'h', negatable: false, help: 'Show this help.');
  final names = CheckpointSpec.all.keys.toList();
  void usage(IOSink sink) {
    sink
      ..writeln(
        'Usage: dart run laya_server:fetch [options] '
        '<${[...names, 'all'].join('|')}>...',
      )
      ..writeln(parser.usage);
  }

  final ArgResults parsed;
  try {
    parsed = parser.parse(args);
  } on FormatException catch (e) {
    stderr.writeln(e.message);
    usage(stderr);
    exitCode = _usageError;
    return;
  }
  if (parsed['help'] as bool) {
    usage(stdout);
    return;
  }
  var wanted = parsed.rest;
  if (wanted.isEmpty) {
    stderr.writeln(
      'Name at least one checkpoint: ${names.join(', ')}, or all.',
    );
    usage(stderr);
    exitCode = _usageError;
    return;
  }
  final unknown = wanted.where((n) => n != 'all' && !names.contains(n));
  if (unknown.isNotEmpty) {
    stderr.writeln(
      'Unknown checkpoint ${unknown.join(', ')}; '
      'valid names: ${names.join(', ')}, all.',
    );
    exitCode = _usageError;
    return;
  }
  if (wanted.contains('all')) wanted = names;
  wanted = wanted.toSet().toList();

  final variant = parsed['variant'] as String;
  final store = ModelStore(parsed['dir'] as String);
  var total = 0;
  for (final n in wanted) {
    for (final f in store.filesFor(n, variant: variant)) {
      total += f.size;
    }
  }
  stdout.writeln(
    'Fetching ${wanted.join(', ')} ($variant) into ${parsed['dir']}: '
    '${_mb(total)} in total.',
  );
  try {
    for (final n in wanted) {
      await store.install(
        n,
        variant: variant,
        onProgress: (path, done, size) {
          final pct = (100 * done / size).toStringAsFixed(0);
          final line = '$n/$path  ${_mb(done)} / ${_mb(size)}  $pct%';
          stdout.write('\r${line.padRight(79)}');
        },
      );
      stdout.writeln('\r${'$n: installed and verified.'.padRight(79)}');
    }
  } on Object catch (e) {
    stdout.writeln();
    stderr.writeln(
      'fetch failed: ${e.toString().replaceAll(RegExp(r'\s+'), ' ')}',
    );
    exitCode = 1;
  }
}
