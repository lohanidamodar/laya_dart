/// The `laya_fetch` command behind `bin/laya_fetch.dart`, separated so tests
/// can run it.
library;

import 'dart:io';

import 'package:args/args.dart';

import '../model/checkpoint.dart';
import 'model_store.dart';

const _usageError = 64;

String _mb(int bytes) => '${(bytes / (1 << 20)).toStringAsFixed(1)} MB';

/// Runs `laya_fetch` with [args] and returns its exit code: 0 on success, 1 on a
/// failure (one sentence on [err]), 64 on a usage error.
///
/// After each checkpoint installs, every file is hashed against its pin
/// (skip with `--no-verify`); a mismatch is listed on [err] and fails the
/// run. [storeFor] builds the store for the `--dir` value (tests inject one).
Future<int> runFetch(
  List<String> args, {
  required IOSink out,
  required IOSink err,
  ModelStore Function(String dir)? storeFor,
}) async {
  final parser = ArgParser()
    ..addOption('dir', defaultsTo: 'models', help: 'Where checkpoints install.')
    ..addOption(
      'variant',
      defaultsTo: 'fp32',
      allowed: ['fp32', 'fp16'],
      help: 'Which exported weights to fetch.',
    )
    ..addFlag(
      'verify',
      defaultsTo: true,
      help: 'Hash every installed file against its pin.',
    )
    ..addFlag('help', abbr: 'h', negatable: false, help: 'Show this help.');
  final names = CheckpointSpec.all.keys.toList();
  void usage(IOSink sink) {
    sink
      ..writeln(
        'Usage: dart run laya_server:laya_fetch [options] '
        '<${[...names, 'all'].join('|')}>...',
      )
      ..writeln(parser.usage);
  }

  final ArgResults parsed;
  try {
    parsed = parser.parse(args);
  } on FormatException catch (e) {
    err.writeln(e.message);
    usage(err);
    return _usageError;
  }
  if (parsed['help'] as bool) {
    usage(out);
    return 0;
  }
  var wanted = parsed.rest;
  if (wanted.isEmpty) {
    err.writeln('Name at least one checkpoint: ${names.join(', ')}, or all.');
    usage(err);
    return _usageError;
  }
  final unknown = wanted.where((n) => n != 'all' && !names.contains(n));
  if (unknown.isNotEmpty) {
    err.writeln(
      'Unknown checkpoint ${unknown.join(', ')}; '
      'valid names: ${names.join(', ')}, all.',
    );
    return _usageError;
  }
  if (wanted.contains('all')) wanted = names;
  wanted = wanted.toSet().toList();

  final variant = parsed['variant'] as String;
  final dir = parsed['dir'] as String;
  final store = storeFor?.call(dir) ?? ModelStore(dir);
  var total = 0;
  for (final n in wanted) {
    for (final f in store.filesFor(n, variant: variant)) {
      total += f.size;
    }
  }
  out.writeln(
    'Fetching ${wanted.join(', ')} ($variant) into $dir: '
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
          out.write('\r${line.padRight(79)}');
        },
      );
      if (parsed['verify'] as bool) {
        out.write('\r${'$n: verifying...'.padRight(79)}');
        final bad = await store.verify(n, variant: variant);
        if (bad.isNotEmpty) {
          out.writeln();
          err.writeln(
            'fetch failed: $n has files that do not match their pinned '
            'sha256: ${bad.join(', ')}; delete them and run laya_fetch again.',
          );
          return 1;
        }
      }
      out.writeln('\r${'$n: installed and verified.'.padRight(79)}');
    }
  } on Object catch (e) {
    out.writeln();
    err.writeln(
      'fetch failed: ${e.toString().replaceAll(RegExp(r'\s+'), ' ')}',
    );
    return 1;
  }
  return 0;
}
