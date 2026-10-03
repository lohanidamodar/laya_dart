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
import 'package:laya_server/src/routing/router.dart';
import 'package:logging/logging.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;

Future<void> main() async {
  // Before the config, so its fallback warnings are printed.
  _setUpLogging(Platform.environment['LAYA_LOG_LEVEL'] ?? 'info');
  final ServerConfig config;
  try {
    config = ServerConfig.fromEnvironment(Platform.environment);
  } on ArgumentError catch (e) {
    stderr.writeln(e.message);
    exit(2);
  }
  final log = Logger('laya_server.server');

  // laya `Router.preload` raises the resident cap to hold every checkpoint it
  // preloads, so a lower LAYA_MAX_LOADED never evicts one of them.
  final List<String> preload;
  try {
    preload = {
      for (final name
          in config.preload.isEmpty ? CheckpointSpec.all.keys : config.preload)
        normaliseCheckpointName(name),
    }.toList();
  } on ArgumentError catch (e) {
    stderr.writeln('invalid LAYA_MODELS: ${e.message}');
    exit(2);
  }
  final engine = LayaEngine(
    loader: LayaEngine.fileLoader(
      config.modelDir,
      variant: config.variant,
      threads: config.threads,
      ortLibraryPath: config.ortLibraryPath,
    ),
    maxLoaded: config.preloadEnabled
        ? math.max(config.maxLoaded, preload.length)
        : config.maxLoaded,
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
      stderr.writeln(
        'laya_server: cannot start: '
        '${reason.replaceAll(RegExp(r'\s+'), ' ')}',
      );
      await engine.close();
      exit(2);
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
    stderr.writeln(
      'laya_server: cannot listen on ${config.host}:${config.port}: '
      '${e.osError?.message ?? e.message}',
    );
    await engine.close();
    exit(2);
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
