// Runs the three gates, each into its own log file, and reports every exit code.
import 'dart:io';

Future<int> _gate(String name, List<String> args) async {
  final log = File('.tooling/logs/$name.txt')..createSync(recursive: true);
  final result = await Process.run(Platform.resolvedExecutable, args);
  log.writeAsStringSync('${result.stdout}${result.stderr}');
  return result.exitCode;
}

Future<void> main(List<String> args) async {
  final codes = {
    'format': await _gate('format', [
      'format',
      '--output=none',
      '--set-exit-if-changed',
      '.',
    ]),
    'analyze': await _gate('analyze', ['analyze', '--fatal-infos']),
    'test': await _gate('test', ['test', ...args]),
  };
  stdout.writeln(codes.entries.map((e) => '${e.key}=${e.value}').join(' '));
  exitCode = codes.values.any((c) => c != 0) ? 1 : 0;
}
