// Serves the Laya decision model over laya-serve's HTTP API.
//
//   dart run laya_server
//
// Configured by LAYA_* environment variables (see ServerConfig). Exit codes:
// 0 after a clean shutdown, 2 when the configuration or the installed
// checkpoints cannot start a server (one sentence on stderr).
import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:laya_server/laya_server.dart';
import 'package:laya_server/src/engine.dart' show isCheckpointInstalled;
import 'package:logging/logging.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;

/// The engine's ceiling on resident checkpoints.
const _maxLoadedCeiling = 1 << 30;

/// Writes [message] to stderr, runs [cleanup], flushes stderr and exits 2.
Future<Never> _fail(String message, {Future<void> Function()? cleanup}) async {
  stderr.writeln(message);
  await cleanup?.call();
  await stderr.flush();
  exit(2);
}

Future<void> main() async {
  // Before the config, so its fallback warnings are printed.
  _setUpLogging(Platform.environment['LAYA_LOG_LEVEL'] ?? 'info');
  final ServerConfig config;
  try {
    config = ServerConfig.fromEnvironment(Platform.environment);
  } on ArgumentError catch (e) {
    await _fail('${e.message}');
  }
  final log = Logger('laya_server.server');

  // laya-serve preloads every checkpoint when LAYA_MODELS is unset; here that
  // means every installed one, so `laya_fetch english` then `laya_server`
  // starts. A checkpoint LAYA_MODELS names must be installed.
  final List<String> preload;
  try {
    preload = config.preload.isEmpty
        ? [
            for (final name in CheckpointSpec.all.keys)
              if (isCheckpointInstalled(
                config.modelDir,
                name,
                variant: config.variant,
              ))
                name,
          ]
        : {
            for (final name in config.preload) normaliseCheckpointName(name),
          }.toList();
  } on ArgumentError catch (e) {
    await _fail('invalid LAYA_MODELS: ${e.message}');
  }
  if (config.preloadEnabled && preload.isEmpty) {
    await _fail(
      'laya_server: cannot start: no checkpoint is installed in '
      '${config.modelDir}; run: dart run laya_server:laya_fetch english, '
      'or set LAYA_MODELS',
    );
  }

  // laya `Router.preload` raises the resident cap to hold every checkpoint it
  // preloads, so a lower LAYA_MAX_LOADED never evicts one of them.
  final engine = LayaEngine(
    loader: LayaEngine.fileLoader(
      config.modelDir,
      variant: config.variant,
      threads: config.threads,
      ortLibraryPath: config.ortLibraryPath,
    ),
    maxLoaded: math.min(
      _maxLoadedCeiling,
      config.preloadEnabled
          ? math.max(config.maxLoaded, preload.length)
          : config.maxLoaded,
    ),
    defaultModel: config.defaultModel,
    autoTaskDetection: config.autoTask,
  );

  if (config.preloadEnabled) {
    try {
      await engine.preload(preload);
    } catch (e) {
      // A missing checkpoint names its fetch command; a missing ONNX Runtime
      // library says where it looked. Either way, one line and no server.
      final reason = switch (e) {
        StateError(:final message) => message,
        ArgumentError(:final message) => '$message',
        _ => '$e',
      };
      await _fail(
        'laya_server: cannot start: '
        '${reason.replaceAll(RegExp(r'\s+'), ' ')}',
        cleanup: engine.close,
      );
    }
  }

  final HttpServer server;
  try {
    server = await shelf_io.serve(
      createHandler(engine, config),
      config.host,
      config.port,
      poweredByHeader: null,
    );
  } on SocketException catch (e) {
    await _fail(
      'laya_server: cannot listen on ${config.host}:${config.port}: '
      '${e.osError?.message ?? e.message}',
      cleanup: engine.close,
    );
  }
  // No dart:io defaults (text/plain, X-Frame-Options, ...): every response
  // sets its own headers, as laya-serve's do. No compression either. Nagle
  // is already off: dart:io sets TCP_NODELAY on every accepted TCP socket
  // (sdk/lib/_http/http_impl.dart, _HttpServer.listen), which is what
  // laya-serve's NoDelayHTTPProtocol does by hand (#620).
  server
    ..defaultResponseHeaders.clear()
    ..autoCompress = false;
  log.info(
    'laya_server listening on http://${config.host}:${server.port} '
    '(loaded: ${engine.loaded.join(', ')})',
  );

  final stop = Completer<void>();
  final signals = [
    ProcessSignal.sigint,
    // Windows cannot watch SIGTERM; a service manager stops the process.
    if (!Platform.isWindows) ProcessSignal.sigterm,
  ];
  final subscriptions = [
    for (final signal in signals)
      signal.watch().listen((_) {
        if (!stop.isCompleted) stop.complete();
      }),
  ];
  await stop.future;
  log.info('laya_server shutting down');
  for (final s in subscriptions) {
    await s.cancel();
  }
  await server.close();
  await engine.close();
}

/// Prints log records to stderr at [level] (laya-serve's `LAYA_LOG_LEVEL`
/// names: critical, error, warning, info, debug, trace).
void _setUpLogging(String level) {
  Logger.root.level = switch (level.trim().toLowerCase()) {
    'critical' => Level.SHOUT,
    'error' => Level.SEVERE,
    'warning' => Level.WARNING,
    'debug' => Level.FINE,
    'trace' => Level.ALL,
    _ => Level.INFO,
  };
  Logger.root.onRecord.listen((r) {
    stderr.writeln('${r.level.name} ${r.loggerName}: ${r.message}');
    if (r.error != null) stderr.writeln('  ${r.error}');
    if (r.stackTrace != null) stderr.writeln('${r.stackTrace}');
  });
}
