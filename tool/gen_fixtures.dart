// Regenerates test/fixtures/*.json from the Python reference in .tooling/venv.
//
// Usage: dart run tool/gen_fixtures.dart
//
// Only tokenizer and config files are downloaded, at pinned revisions; no
// model weights. The Python steps run with HF_HUB_OFFLINE=1.
import 'dart:convert';
import 'dart:io';

const _rev = '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851';
const _base = 'https://huggingface.co/convaiinnovations/laya/resolve/$_rev';
const _files = {
  'english': [
    'tokenizer/tokenizer.json',
    'tokenizer/tokenizer_config.json',
    'rl_agent_config.json',
  ],
  'multilingual': [
    'multilingual/tokenizer/tokenizer.json',
    'multilingual/tokenizer/tokenizer_config.json',
    'multilingual/rl_agent_config.json',
  ],
  'typed-decisions': [
    'typed-decisions/tokenizer/tokenizer.json',
    'typed-decisions/tokenizer/tokenizer_config.json',
    'typed-decisions/rl_agent_config.json',
  ],
};

// The onnx-community exports' config.json, pinned by revision.
const _onnxConfigs = {
  'english':
      'https://huggingface.co/onnx-community/laya-ONNX/resolve/'
      '42e2a6e3b3708c8ff5f61f4aa75c314b1449199b/config.json',
  'multilingual':
      'https://huggingface.co/onnx-community/laya-multilingual-ONNX/resolve/'
      '46b77bbf5642fec5f14e540570228a8cbe8ab81f/config.json',
  'typed-decisions':
      'https://huggingface.co/onnx-community/laya-typed-decisions-ONNX/resolve/'
      '79c441b22da79eb3727bb01e8452e29a32cfa548/config.json',
};

// Absolute, because Windows does not resolve a relative executable path with slashes.
String get _python => File(
  Platform.isWindows
      ? '.tooling/venv/Scripts/python.exe'
      : '.tooling/venv/bin/python',
).absolute.path;

Future<void> _download(String url, String dest) async {
  final file = File(dest);
  if (file.existsSync()) return;
  file.parent.createSync(recursive: true);
  final client = HttpClient();
  try {
    final response = await (await client.getUrl(Uri.parse(url))).close();
    if (response.statusCode != 200) {
      throw HttpException('GET $url -> ${response.statusCode}');
    }
    final part = File('$dest.part');
    await response.pipe(part.openWrite());
    part.renameSync(dest);
  } finally {
    client.close();
  }
}

Future<void> _run(List<String> args, String out) async {
  final result = await Process.run(
    _python,
    args,
    environment: {
      'PYTHONUTF8': '1',
      'PYTHONDONTWRITEBYTECODE': '1',
      'HF_HUB_OFFLINE': '1',
    },
    stdoutEncoding: utf8,
  );
  if (result.exitCode != 0) {
    stderr.writeln(result.stderr);
    throw ProcessException(
      _python,
      args,
      'exit ${result.exitCode}',
      result.exitCode,
    );
  }
  File(out).writeAsStringSync(result.stdout as String);
  stdout.writeln('wrote $out');
}

Future<void> main() async {
  for (final entry in _files.entries) {
    for (final rel in entry.value) {
      final local = rel.replaceFirst('${entry.key}/', '');
      await _download(
        '$_base/$rel',
        '.tooling/checkpoints/${entry.key}/$local',
      );
    }
  }
  for (final entry in _onnxConfigs.entries) {
    await _download(
      entry.value,
      '.tooling/checkpoints/${entry.key}/onnx_config.json',
    );
  }
  const ref = 'tool/reference';
  await _run([
    '$ref/tokenizer_ids.py',
    '.tooling/checkpoints/english/tokenizer/tokenizer.json',
    'english',
  ], 'test/fixtures/tokenizer_english.json');
  await _run([
    '$ref/tokenizer_ids.py',
    '.tooling/checkpoints/multilingual/tokenizer/tokenizer.json',
    'multilingual',
  ], 'test/fixtures/tokenizer_multilingual.json');
  await _run(['$ref/nfc.py'], 'test/fixtures/nfc.json');
  await _run(['$ref/pyjson.py'], 'test/fixtures/pyjson.json');
  await _run(['$ref/sequence.py'], 'test/fixtures/sequence.json');
  await _run(['$ref/lang.py'], 'test/fixtures/lang.json');
  await _run(['$ref/route.py'], 'test/fixtures/route.json');
  await _run(['$ref/unicode_props.py'], 'test/fixtures/unicode_props.json');
  await _run(['$ref/decode.py'], 'test/fixtures/decode.json');
  await _run(['$ref/numpy_f32.py'], 'test/fixtures/numpy_f32.json');
  // The ONNX Runtime binding's test model; the script writes the binary itself.
  Directory('test/fixtures/ort').createSync(recursive: true);
  await _run([
    '$ref/echo_model.py',
    'test/fixtures/ort/echo.onnx',
  ], '.tooling/logs/echo_model.txt');
}
