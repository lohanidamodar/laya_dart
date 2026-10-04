// Locates the real tokenizer.json files the tests run against.
import 'dart:io';

/// The skip reason when a checkpoint's tokenizer.json is on neither path.
const missingTokenizer = 'run dart run tool/gen_fixtures.dart first';

/// The committed copy of [checkpoint]'s tokenizer.json, which works offline and in CI.
///
/// typed-decisions ships a tokenizer.json byte-identical to english's (sha256
/// 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30
/// at the pinned revision), so it shares english's committed copy. multilingual's file is 34 MB
/// and is not committed.
String _committed(String checkpoint) {
  final name = checkpoint == 'typed-decisions' ? 'english' : checkpoint;
  return 'test/fixtures/tokenizers/$name/tokenizer.json';
}

/// The download tool/gen_fixtures.dart leaves under the git-ignored .tooling/.
String _downloaded(String checkpoint) =>
    '.tooling/checkpoints/$checkpoint/tokenizer/tokenizer.json';

String? _find(String checkpoint) => [
  _committed(checkpoint),
  _downloaded(checkpoint),
].where((path) => File(path).existsSync()).firstOrNull;

/// The path of [checkpoint]'s tokenizer.json: the committed copy, else the downloaded one.
///
/// Throws a [StateError] with [missingTokenizer] when neither exists; guard the test or group
/// with `skip: tokenizerSkip(checkpoint)` so a missing file skips instead.
String tokenizerPath(String checkpoint) =>
    _find(checkpoint) ?? (throw StateError(missingTokenizer));

/// The `skip:` value for a test or group that needs [checkpoint]'s tokenizer.json: `false` when
/// the file exists, else the [missingTokenizer] reason.
Object tokenizerSkip(String checkpoint) =>
    _find(checkpoint) == null ? missingTokenizer : false;
