/// Server settings from `LAYA_*` environment variables.
library;

import 'package:logging/logging.dart';

import '../model/render.dart' show pyRepr;
import '../routing/router.dart';

final _log = Logger('laya_server.server');

/// Server settings, read from `LAYA_*` environment variables with
/// laya-serve's names, defaults and parsing (`laya/serve.py`), plus
/// `LAYA_MODEL_DIR`, `LAYA_ONNX_VARIANT` and `LAYA_ORT_LIB`.
///
/// Numeric knobs that do not parse fall back to their defaults, as upstream
/// does; `LAYA_PORT`, `LAYA_DEFAULT_MODEL` and `LAYA_ONNX_VARIANT` have no
/// harmless fallback and are startup errors instead.
class ServerConfig {
  /// Creates settings; prefer [ServerConfig.fromEnvironment].
  const ServerConfig({
    this.host = '0.0.0.0',
    this.port = 8000,
    this.rootPath = '',
    this.modelDir = './models',
    this.preload = const [],
    this.preloadEnabled = true,
    this.threads,
    this.autoTask = false,
    this.defaultModel = 'english',
    this.maxLoaded = 2,
    this.apiKey,
    this.maxConcurrent = 16,
    this.maxTokenBudget = 8192,
    this.jevStrict = false,
    this.variant = 'fp32',
    this.ortLibraryPath,
    this.logLevel = 'info',
  });

  /// Reads [env] (usually `Platform.environment`).
  ///
  /// Throws [ArgumentError] with a one-line message for an invalid
  /// `LAYA_PORT`, an unknown `LAYA_DEFAULT_MODEL` or an unknown
  /// `LAYA_ONNX_VARIANT`.
  factory ServerConfig.fromEnvironment(Map<String, String> env) {
    final variant = env['LAYA_ONNX_VARIANT'] ?? 'fp32';
    if (variant != 'fp32' && variant != 'fp16') {
      throw ArgumentError(
        'invalid LAYA_ONNX_VARIANT ${pyRepr(variant)}: must be fp32 or fp16',
      );
    }
    final models = env['LAYA_MODELS'] ?? '';
    final apiKey = env['LAYA_API_KEY'];
    return ServerConfig(
      host: env['LAYA_HOST'] ?? '0.0.0.0',
      port: _port(env['LAYA_PORT'] ?? '8000'),
      rootPath: env['LAYA_ROOT_PATH'] ?? '',
      modelDir: env['LAYA_MODEL_DIR'] ?? './models',
      preload: [
        for (final m in models.trim().split(','))
          if (m.trim().isNotEmpty) m.trim(),
      ],
      preloadEnabled: _bool(env['LAYA_PRELOAD'], true),
      threads: _positive(env['LAYA_THREADS']),
      autoTask: _bool(env['LAYA_AUTO_TASK'], false),
      defaultModel: _defaultModel(env['LAYA_DEFAULT_MODEL']),
      maxLoaded: _positive(env['LAYA_MAX_LOADED']) ?? 2,
      apiKey: apiKey == null || apiKey.isEmpty ? null : apiKey,
      maxConcurrent: _positive(env['LAYA_MAX_CONCURRENT']) ?? 16,
      maxTokenBudget: _tokenBudget(env['LAYA_MAX_TOKEN_BUDGET']),
      jevStrict: _bool(env['LAYA_JEV_STRICT'], false),
      variant: variant,
      ortLibraryPath: env['LAYA_ORT_LIB'],
      logLevel: env['LAYA_LOG_LEVEL'] ?? 'info',
    );
  }

  /// `LAYA_HOST`: the bind address (default `0.0.0.0`).
  final String host;

  /// `LAYA_PORT`: the bind port, 1-65535 (default 8000).
  final int port;

  /// `LAYA_ROOT_PATH`: the public URL prefix behind a reverse proxy. A
  /// request path under it is served with the prefix removed, as Starlette
  /// does; one without it is served as is.
  final String rootPath;

  /// `LAYA_MODEL_DIR`: where checkpoints are installed (default `./models`).
  final String modelDir;

  /// `LAYA_MODELS`: the checkpoints (names or aliases) to preload; empty
  /// means every checkpoint.
  final List<String> preload;

  /// `LAYA_PRELOAD`: whether to load [preload] at startup (default true).
  final bool preloadEnabled;

  /// `LAYA_THREADS`: ONNX Runtime's intra-op threads; null for its default.
  final int? threads;

  /// `LAYA_AUTO_TASK`: whether a typed-decisions question schema selects
  /// that checkpoint (default false).
  final bool autoTask;

  /// `LAYA_DEFAULT_MODEL`, normalised: the checkpoint for a state whose
  /// language nothing identifies (default `english`).
  final String defaultModel;

  /// `LAYA_MAX_LOADED`: checkpoints kept resident at once (default 2).
  final int maxLoaded;

  /// `LAYA_API_KEY`: when set, `POST /v1/systemone` requires
  /// `Authorization: Bearer <key>` and `/health` reports liveness only to a
  /// caller without it.
  final String? apiKey;

  /// `LAYA_MAX_CONCURRENT`: requests admitted at once; more get 503
  /// (default 16).
  final int maxConcurrent;

  /// `LAYA_MAX_TOKEN_BUDGET`: the cap on a request's `max_len` and
  /// `head_max_len` (default 8192).
  final int maxTokenBudget;

  /// `LAYA_JEV_STRICT`: whether to serve the strict Jev projection.
  final bool jevStrict;

  /// `LAYA_ONNX_VARIANT`: `fp32` (default) or `fp16` weights.
  final String variant;

  /// `LAYA_ORT_LIB`: an explicit ONNX Runtime library path.
  final String? ortLibraryPath;

  /// `LAYA_LOG_LEVEL`: the log level name (default `info`).
  final String logLevel;

  /// laya `_env_bool`: unset is [fallback]; set, only `1`, `true`, `yes`
  /// and `on` (trimmed, any case) are true.
  static bool _bool(String? raw, bool fallback) {
    if (raw == null) return fallback;
    return const ['1', 'true', 'yes', 'on'].contains(raw.trim().toLowerCase());
  }

  /// Python's `int(str)`: surrounding whitespace, a sign and single
  /// underscores between digits are allowed.
  static int? _pyInt(String raw) {
    final s = raw.trim();
    if (!RegExp(r'^[+-]?\d+(_\d+)*$').hasMatch(s)) return null;
    return int.tryParse(s.replaceAll('_', ''));
  }

  /// A positive integer, or null for unset, unparsable or non-positive
  /// (`_resolve_max_concurrent`, `_resolve_max_loaded`, `_apply_thread_limit`).
  static int? _positive(String? raw) {
    if (raw == null || raw.isEmpty) return null;
    final n = _pyInt(raw);
    return n != null && n > 0 ? n : null;
  }

  /// serve.py:409 `_resolve_port`.
  static int _port(String raw) {
    final n = _pyInt(raw);
    if (n == null || n < 1 || n > 65535) {
      throw ArgumentError(
        'invalid LAYA_PORT ${pyRepr(raw)}: must be an integer 1-65535',
      );
    }
    return n;
  }

  /// serve.py:213 `_resolve_max_token_budget`, warnings included.
  static int _tokenBudget(String? raw) {
    const fallback = 8192;
    if (raw == null || raw.isEmpty) return fallback;
    final n = _pyInt(raw);
    if (n == null) {
      _log.warning(
        'invalid LAYA_MAX_TOKEN_BUDGET ${pyRepr(raw)}; '
        'falling back to $fallback',
      );
      return fallback;
    }
    if (n <= 0) {
      _log.warning(
        'LAYA_MAX_TOKEN_BUDGET must be positive (got $n); '
        'falling back to $fallback',
      );
      return fallback;
    }
    return n;
  }

  /// serve.py:380 `_default_model_option`.
  static String _defaultModel(String? raw) {
    if (raw == null || raw.trim().isEmpty) return 'english';
    try {
      return normaliseCheckpointName(raw);
    } on ArgumentError catch (e) {
      throw ArgumentError(
        'invalid LAYA_DEFAULT_MODEL ${pyRepr(raw.trim())}: ${e.message}',
      );
    }
  }
}
