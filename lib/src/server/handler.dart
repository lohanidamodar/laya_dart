/// The HTTP surface: laya-serve's `GET /health` and `POST /v1/systemone`.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:logging/logging.dart';
import 'package:shelf/shelf.dart';

import '../engine.dart';
import '../json/python_json.dart';
import '../model/checkpoint.dart';
import '../routing/router.dart' show unknownCheckpointMessage;
import 'body.dart';
import 'config.dart';
import 'errors.dart';
import 'limits.dart';
import 'strict.dart';

/// The routes and the methods each serves. FastAPI does not add `HEAD` to a
/// `GET` route, so `HEAD /health` is a 405 there too.
const _routes = {
  '/health': ['GET'],
  '/v1/systemone': ['POST'],
};

/// The shelf handler serving `GET /health` and `POST /v1/systemone` over
/// [engine], with laya-serve's statuses, `{"detail": ...}` texts, check
/// order, limits, auth, admission and timing headers (`laya/serve.py`).
///
/// - `/health` is open; with [ServerConfig.apiKey] set, a caller without
///   the bearer gets `{"status":"ok"}` only.
/// - `/v1/systemone` checks the bearer (401), then admission: past
///   [ServerConfig.maxConcurrent] requests in flight it answers 503 with
///   `Retry-After: 1` before reading the body. It then reads the body (at
///   most 2 MiB, 413), validates it in laya-serve's order and runs one
///   prediction at a time.
/// - Any other path is 404 `Not Found`, another method on a known path 405
///   `Method Not Allowed`, and a known path with its trailing slash toggled
///   a 307 redirect, as FastAPI answers them.
///
/// An inference failure is a 500 `inference failed`; the cause goes to
/// [logger] (default `Logger('laya_server.server')`) at `SEVERE`, with its
/// stack trace. Request bodies and the API key are never logged.
Handler createHandler(
  LayaEngine engine,
  ServerConfig config, {
  Logger? logger,
}) => _Server(engine, config, logger ?? Logger('laya_server.server')).handle;

class _Server {
  _Server(this.engine, this.config, this.log)
    : _expectedAuth = config.apiKey == null
          ? null
          : utf8.encode('Bearer ${config.apiKey}');

  final LayaEngine engine;
  final ServerConfig config;
  final Logger log;

  /// `Bearer <key>` as UTF-8 bytes, compared against the header's bytes.
  final List<int>? _expectedAuth;

  /// Requests past auth whose answer is not yet sent (laya's admission
  /// semaphore). Checked before any body byte is read.
  int _admitted = 0;

  /// One prediction at a time, across checkpoints. laya-serve runs inference
  /// on a single worker thread; the backend isolate already serialises each
  /// checkpoint, and this keeps two checkpoints from oversubscribing the CPU
  /// threads at once. It is joined only once the body is read, so a slow
  /// client holds an admission slot but never this gate.
  final _gate = _Mutex();

  Future<Response> handle(Request request) async {
    try {
      return await _route(request);
    } on ApiException catch (e) {
      return _json(e.status, {'detail': e.detail}, headers: e.headers);
    } catch (error, stackTrace) {
      // Starlette's ServerErrorMiddleware answer for an unhandled error.
      log.severe('unhandled error', error, stackTrace);
      return Response.internalServerError(
        body: 'Internal Server Error',
        headers: {'content-type': 'text/plain; charset=utf-8'},
      );
    }
  }

  Future<Response> _route(Request request) async {
    // Starlette routes on the percent-decoded path, with the root path
    // stripped when the request carries it.
    final path = _unquote(request.requestedUri.path);
    final routePath = _routePath(path);
    final methods = _routes[routePath];
    if (methods != null) {
      if (!methods.contains(request.method)) {
        throw ApiException(
          405,
          'Method Not Allowed',
          headers: {'allow': methods.join(', ')},
        );
      }
      return routePath == '/health' ? _health(request) : _systemOne(request);
    }
    // Starlette's redirect_slashes: the path with its trailing slash toggled.
    if (routePath != '/') {
      final toggled = routePath.endsWith('/')
          ? path.replaceFirst(RegExp(r'/+$'), '')
          : '$path/';
      if (_routes.containsKey(_routePath(toggled))) {
        return Response(
          307,
          headers: {'location': _redirectUrl(request, toggled)},
        );
      }
    }
    throw const ApiException(404, 'Not Found');
  }

  /// Starlette `get_route_path`.
  String _routePath(String path) {
    final root = config.rootPath;
    if (root.isEmpty || !path.startsWith(root)) return path;
    if (path == root) return '';
    return path[root.length] == '/' ? path.substring(root.length) : path;
  }

  bool _authorized(Request request) {
    final expected = _expectedAuth;
    if (expected == null) return true;
    // dart:io decodes header bytes as latin-1, as Starlette does; encoding
    // that text as UTF-8 compares exactly what laya-serve compares.
    final supplied = utf8.encode(request.headers['authorization'] ?? '');
    return _constantTimeEquals(supplied, expected);
  }

  Response _health(Request request) {
    if (!_authorized(request)) return _json(200, {'status': 'ok'});
    final loaded = engine.loaded;
    return _json(200, {
      'status': 'ok',
      'loaded': loaded,
      'revisions': {
        for (final n in loaded) n: CheckpointSpec.all[n]?.onnxRevision,
      },
      // ONNX Runtime runs on the CPU here, so where a checkpoint computes is
      // never a preference that fell back; with nothing loaded the value is
      // the configured preference, as laya-serve reports it.
      'device': 'cpu',
      'device_is_preference': loaded.isEmpty,
      'checkpoint_devices': {for (final n in loaded) n: 'cpu'},
      'cpu_fallbacks': {
        for (final n in loaded) n: {'count': 0, 'last_reason': null},
      },
    });
  }

  Future<Response> _systemOne(Request request) async {
    if (!_authorized(request)) {
      throw const ApiException(401, 'invalid or missing bearer token');
    }
    if (_admitted >= config.maxConcurrent) {
      // Refused, not queued: the bodies buffered at once stay bounded.
      throw const ApiException(
        503,
        'server busy, try again later',
        headers: {'retry-after': '1'},
      );
    }
    _admitted++;
    try {
      return await _systemOneAdmitted(request);
    } finally {
      _admitted--;
    }
  }

  Future<Response> _systemOneAdmitted(Request request) async {
    final decoded = decodeJsonBody(await _readBody(request));
    final body = decoded.json;
    if (body is! Map<String, Object?> || !body.containsKey('questions')) {
      throw const ApiException(
        400,
        "request body must be an object with a 'questions' field",
      );
    }
    final state = body['state'];
    final questions = body['questions'];
    checkRequestLimits(state, questions);
    refuseBodyRefusals(body);
    // After the size checks, which bound what the walk can reach.
    if (decoded.rawSurrogate || hasLoneSurrogate(body)) {
      throw const ApiException(400, loneSurrogateDetail);
    }
    // laya reads `model` through str(); no non-string names a checkpoint.
    final rawModel = body['model'];
    final model = rawModel is String ? rawModel : null;
    final resolved = resolveClientModel(model);
    final cap = config.maxTokenBudget;
    final maxLen = validateBudgetParam(body, 'max_len', cap);
    final headMaxLen = validateBudgetParam(body, 'head_max_len', cap);
    final task = body['task'];
    final lang = validateLanguageParam(body, 'lang');
    final langGuess = validateLanguageParam(body, 'lang_guess');
    final minConfidence = validateMinConfidence(body);
    final options = PredictOptions(
      model: model,
      task: task is String ? task : null,
      lang: lang,
      langGuess: langGuess,
      maxLen: maxLen,
      headMaxLen: headMaxLen,
      minConfidence: minConfidence,
    );

    return _gate.run(() async {
      try {
        // laya routes a non-string task through str() and refuses it, unless
        // an explicit model already chose the checkpoint.
        if (task != null && task is! String && resolved == null) {
          throw ApiException(422, unknownCheckpointMessage(task));
        }
        final watch = Stopwatch()..start();
        var result = await engine.predict(
          state!,
          questions! as Map<String, Object?>,
          options,
        );
        final ms = (watch.elapsedMicroseconds / 1000).toStringAsFixed(2);
        if (config.jevStrict) result = projectJevStrict(result);
        return _json(
          200,
          result,
          headers: {
            'server-timing': 'inference;dur=$ms',
            'x-inference-time-ms': ms,
          },
        );
      } on ApiException {
        rethrow;
      } catch (error, stackTrace) {
        // The client learns nothing about paths, weights or memory; the
        // operator gets the cause. The model name is a checkpoint name or
        // None, and line breaks are escaped so it cannot forge log lines.
        final name = (resolved ?? 'None')
            .replaceAll('\n', r'\n')
            .replaceAll('\r', r'\r');
        log.severe('inference failed for model=$name', error, stackTrace);
        throw const ApiException(500, 'inference failed');
      }
    });
  }

  /// serve.py `_read_json_body` / `_read_body_capped`: refuses a declared
  /// length over the cap, then streams, abandoning the body once past it.
  Future<Uint8List> _readBody(Request request) async {
    const tooLarge = ApiException(413, 'request body too large');
    final declared = request.headers['content-length'];
    if (declared != null && declared.isNotEmpty) {
      final n = int.tryParse(declared.trim());
      if (n != null && n > maxBodyBytes) throw tooLarge;
    }
    final out = BytesBuilder(copy: false);
    final body = Completer<Uint8List>();
    late final StreamSubscription<List<int>> subscription;
    subscription = request.read().listen(
      (chunk) {
        if (out.length + chunk.length > maxBodyBytes) {
          // Stop reading, but pause rather than cancel: dart:io drops the
          // connection at once when a request body is cancelled, before the
          // 413 is written. Paused, the 413 goes out and dart:io then closes
          // the connection, since the body was not read to its end.
          subscription.pause();
          body.completeError(tooLarge);
          return;
        }
        out.add(chunk);
      },
      onError: (Object error, StackTrace stackTrace) {
        if (!body.isCompleted) body.completeError(error, stackTrace);
      },
      onDone: () {
        if (!body.isCompleted) body.complete(out.takeBytes());
      },
      cancelOnError: true,
    );
    return body.future;
  }
}

/// A JSON response, serialised as Starlette's `JSONResponse` does
/// (`json.dumps(..., ensure_ascii=False, separators=(",", ":"))`).
///
/// A NaN or infinity throws [ArgumentError], which the caller reports as an
/// inference failure.
Response _json(
  int status,
  Object? body, {
  Map<String, String> headers = const {},
}) {
  _checkFinite(body);
  return Response(
    status,
    body: utf8.encode(
      pythonJsonDumps(body, itemSeparator: ',', keySeparator: ':'),
    ),
    headers: {'content-type': 'application/json', ...headers},
  );
}

/// Starlette's `JSONResponse` sets `allow_nan=False`: a non-finite number
/// is not JSON.
void _checkFinite(Object? value) {
  final stack = <Object?>[value];
  while (stack.isNotEmpty) {
    final item = stack.removeLast();
    if (item is double && !item.isFinite) {
      throw ArgumentError.value(item, 'body', 'is not JSON compliant');
    } else if (item is Map) {
      stack.addAll(item.values);
    } else if (item is List) {
      stack.addAll(item);
    }
  }
}

/// `hmac.compare_digest`: equal lengths are compared in time independent of
/// where the bytes differ.
bool _constantTimeEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  var diff = 0;
  for (var i = 0; i < a.length; i++) {
    diff |= a[i] ^ b[i];
  }
  return diff == 0;
}

/// Python's `urllib.parse.unquote`, as the ASGI server decodes the path:
/// `%XX` escapes are decoded as UTF-8, invalid sequences replaced with
/// U+FFFD, and a malformed escape kept literally.
String _unquote(String s) {
  if (!s.contains('%')) return s;
  final bytes = <int>[];
  for (var i = 0; i < s.length; i++) {
    final c = s.codeUnitAt(i);
    if (c == 0x25 && i + 2 < s.length) {
      final v = int.tryParse(s.substring(i + 1, i + 3), radix: 16);
      if (v != null && !s.substring(i + 1, i + 3).contains(RegExp('[+-]'))) {
        bytes.add(v);
        i += 2;
        continue;
      }
    }
    bytes.addAll(utf8.encode(s[i]));
  }
  return utf8.decode(bytes, allowMalformed: true);
}

/// Starlette's redirect URL for [path]: scheme, the `Host` header, the path
/// and the raw query, through Python's `quote` with Starlette's safe set.
String _redirectUrl(Request request, String path) {
  final uri = request.requestedUri;
  final host = request.headers['host'] ?? uri.authority;
  final query = uri.query.isEmpty ? '' : '?${uri.query}';
  return _quote('${uri.scheme}://$host$path$query');
}

/// Python's `urllib.parse.quote(s, safe=":/%#?=@[]!$&'()*+,;")`.
String _quote(String s) {
  const safe = r":/%#?=@[]!$&'()*+,;_.-~";
  final out = StringBuffer();
  for (final b in utf8.encode(s)) {
    final ok =
        (b >= 0x30 && b <= 0x39) ||
        (b >= 0x41 && b <= 0x5a) ||
        (b >= 0x61 && b <= 0x7a) ||
        (b < 0x80 && safe.contains(String.fromCharCode(b)));
    out.write(
      ok
          ? String.fromCharCode(b)
          : '%${b.toRadixString(16).toUpperCase().padLeft(2, '0')}',
    );
  }
  return out.toString();
}

/// A first-come, first-served async lock.
class _Mutex {
  Future<void> _tail = Future.value();

  /// Runs [body] once every earlier [run] has finished.
  Future<T> run<T>(Future<T> Function() body) {
    final previous = _tail;
    final done = Completer<void>();
    _tail = done.future;
    return previous.then((_) => body()).whenComplete(done.complete);
  }
}
