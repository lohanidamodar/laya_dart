import 'package:laya_server/laya_server.dart';
import 'package:test/test.dart';

const _aliases =
    "choose one of ['english', 'multilingual', 'typed-decisions'] "
    "(or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', "
    "'laya-typed-decisions', 'ml', 'multi', 'typed', 'typed_decisions'])";

/// The [ArgumentError] message [env] fails with.
String _error(Map<String, String> env) {
  try {
    ServerConfig.fromEnvironment(env);
  } on ArgumentError catch (e) {
    return '${e.message}';
  }
  fail('accepted $env');
}

void main() {
  test('defaults are laya-serve\'s, plus ours', () {
    final c = ServerConfig.fromEnvironment({});
    expect(c.host, '0.0.0.0');
    expect(c.port, 8000);
    expect(c.rootPath, '');
    expect(c.modelDir, './models');
    expect(c.preload, isEmpty); // every checkpoint
    expect(c.preloadEnabled, isTrue);
    expect(c.threads, isNull);
    expect(c.autoTask, isFalse);
    expect(c.defaultModel, 'english');
    expect(c.maxLoaded, 2);
    expect(c.apiKey, isNull);
    expect(c.maxConcurrent, 16);
    expect(c.maxTokenBudget, 8192);
    expect(c.jevStrict, isFalse);
    expect(c.variant, 'fp32');
    expect(c.ortLibraryPath, isNull);
  });

  test('reads every variable', () {
    final c = ServerConfig.fromEnvironment({
      'LAYA_HOST': '127.0.0.1',
      'LAYA_PORT': ' 8_080 ',
      'LAYA_ROOT_PATH': '/laya',
      'LAYA_MODEL_DIR': '/srv/models',
      'LAYA_MODELS': ' english, ml ,, ',
      'LAYA_PRELOAD': 'off',
      'LAYA_THREADS': '4',
      'LAYA_AUTO_TASK': ' Yes ',
      'LAYA_DEFAULT_MODEL': 'ML',
      'LAYA_MAX_LOADED': '3',
      'LAYA_API_KEY': 'k',
      'LAYA_MAX_CONCURRENT': '1',
      'LAYA_MAX_TOKEN_BUDGET': ' 100 ',
      'LAYA_JEV_STRICT': 'on',
      'LAYA_ONNX_VARIANT': 'fp16',
      'LAYA_ORT_LIB': '/opt/ort/libonnxruntime.so',
    });
    expect(c.host, '127.0.0.1');
    expect(c.port, 8080);
    expect(c.rootPath, '/laya');
    expect(c.modelDir, '/srv/models');
    expect(c.preload, ['english', 'ml']);
    expect(c.preloadEnabled, isFalse);
    expect(c.threads, 4);
    expect(c.autoTask, isTrue);
    expect(c.defaultModel, 'multilingual');
    expect(c.maxLoaded, 3);
    expect(c.apiKey, 'k');
    expect(c.maxConcurrent, 1);
    expect(c.maxTokenBudget, 100);
    expect(c.jevStrict, isTrue);
    expect(c.variant, 'fp16');
    expect(c.ortLibraryPath, '/opt/ort/libonnxruntime.so');
  });

  test('booleans are laya _env_bool: only 1/true/yes/on, any case', () {
    bool strict(String v) =>
        ServerConfig.fromEnvironment({'LAYA_JEV_STRICT': v}).jevStrict;
    expect(strict('1'), isTrue);
    expect(strict('TRUE'), isTrue);
    expect(strict('0'), isFalse);
    expect(strict('enabled'), isFalse);
    // Set but empty is false, even where the default is true.
    expect(
      ServerConfig.fromEnvironment({'LAYA_PRELOAD': ''}).preloadEnabled,
      isFalse,
    );
  });

  test('an invalid LAYA_PORT is a startup error', () {
    for (final raw in ['abc', '', '0', '65536', '-1', '80.0']) {
      expect(
        _error({'LAYA_PORT': raw}),
        "invalid LAYA_PORT '$raw': must be an integer 1-65535",
      );
    }
    expect(ServerConfig.fromEnvironment({'LAYA_PORT': '65535'}).port, 65535);
  });

  test('invalid numeric knobs fall back to the defaults', () {
    for (final raw in ['', 'x', '0', '-2', '1.5']) {
      final c = ServerConfig.fromEnvironment({
        'LAYA_MAX_CONCURRENT': raw,
        'LAYA_MAX_LOADED': raw,
        'LAYA_MAX_TOKEN_BUDGET': raw,
        'LAYA_THREADS': raw,
      });
      expect(c.maxConcurrent, 16, reason: raw);
      expect(c.maxLoaded, 2, reason: raw);
      expect(c.maxTokenBudget, 8192, reason: raw);
      expect(c.threads, isNull, reason: raw);
    }
  });

  test('an unknown LAYA_DEFAULT_MODEL is a startup error', () {
    expect(
      _error({'LAYA_DEFAULT_MODEL': ' nope '}),
      "invalid LAYA_DEFAULT_MODEL 'nope': unknown model ' nope '; $_aliases",
    );
    expect(
      ServerConfig.fromEnvironment({'LAYA_DEFAULT_MODEL': '  '}).defaultModel,
      'english',
    );
  });

  test('an unknown LAYA_ONNX_VARIANT is a startup error', () {
    expect(
      _error({'LAYA_ONNX_VARIANT': 'int8'}),
      "invalid LAYA_ONNX_VARIANT 'int8': must be fp32 or fp16",
    );
  });

  test('an empty LAYA_API_KEY means no key', () {
    expect(ServerConfig.fromEnvironment({'LAYA_API_KEY': ''}).apiKey, isNull);
  });
}
