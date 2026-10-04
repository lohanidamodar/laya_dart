import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/laya_server.dart';
import 'package:laya_server/src/engine.dart' show resolveClientModel;
import 'package:laya_server/src/routing/router.dart';
import 'package:logging/logging.dart';
import 'package:shelf/shelf.dart';
import 'package:test/test.dart';

import '../support/checkpoints.dart';

/// A response, read.
class _Reply {
  _Reply(this.status, this.headers, this.text);
  final int status;
  final Map<String, String> headers;
  final String text;
  Object? get json => text.isEmpty ? null : jsonDecode(text);
}

Future<_Reply> _send(
  Handler handler,
  String method,
  String path, {
  Map<String, String> headers = const {},
  Object? body,
}) async {
  final r = await handler(
    Request(
      method,
      Uri.parse('http://testserver$path'),
      headers: {'host': 'testserver', ...headers},
      body: body,
    ),
  );
  return _Reply(r.statusCode, r.headers, await r.readAsString());
}

Future<_Reply> _post(
  Handler handler,
  Object? json, {
  Map<String, String> headers = const {},
}) => _send(
  handler,
  'POST',
  '/v1/systemone',
  headers: headers,
  body: utf8.encode(jsonEncode(json)),
);

/// The engine `tool/reference/serve.py` gives laya-serve: real routing, then
/// a fixed answer per question in place of a forward pass, then laya's
/// confidence gate (`Router.predict`).
class _StubEngine implements LayaEngine {
  final calls = <PredictOptions>[];
  final _loaded = <String>[];

  @override
  List<String> get loaded => List.of(_loaded);

  @override
  Future<Map<String, Object?>> predict(
    Object state,
    Map<String, Object?> questions,
    PredictOptions options,
  ) async {
    calls.add(options);
    final RouteDecision decision;
    try {
      decision = CheckpointRouter().route(
        state,
        questions,
        model: resolveClientModel(options.model),
        task: options.task,
        lang: options.lang,
        langGuess: options.langGuess,
      );
    } on ArgumentError catch (e) {
      throw ApiException(422, '${e.message}');
    }
    if (!_loaded.contains(decision.model)) _loaded.add(decision.model);
    final answers = <String, Object?>{
      for (final id in questions.keys)
        id: <String, Object?>{
          'type': 'noul',
          'noul': 0.9,
          'confidence': 0.9,
          'answer_confidence': 0.9,
          'action': {'act_probability': 1.0},
        },
    };
    for (final a in answers.values) {
      options.minConfidence?.gate(a! as Map<String, Object?>);
    }
    return {
      'model': 'laya-rl-agent',
      'answers': answers,
      'usage': {'input_tokens': 7, 'output_tokens': 0},
      'routing': decision.toJson(),
    };
  }

  @override
  Future<void> preload(Iterable<String> names) async {}

  @override
  Future<void> close() async {}
}

/// The arguments laya's `Router.predict` would have received.
Map<String, Object?> _asLayaCall(PredictOptions o) => {
  'model': resolveClientModel(o.model),
  'kwargs': {
    if (o.maxLen != null) 'max_len': o.maxLen,
    if (o.headMaxLen != null) 'head_max_len': o.headMaxLen,
    if (o.task != null) 'task': o.task,
    if (o.lang != null) 'lang': o.lang,
    if (o.langGuess != null) 'lang_guess': o.langGuess,
    if (o.minConfidence != null) 'min_confidence': o.minConfidence!.toJson(),
  },
};

List<int> _bodyBytes(Map<String, Object?> spec) {
  if (spec['text'] case final String text) return utf8.encode(text);
  if (spec['b64'] case final String b64) return base64.decode(b64);
  return [
    for (final part in spec['parts']! as List)
      for (var i = 0; i < ((part as List)[1] as int); i++)
        ...utf8.encode(part[0] as String),
  ];
}

/// Cases where laya_server deliberately answers differently (see the report).
const _differences = <String, Map<String, Object?>>{
  // v1 scope: the batch route is not served.
  'batch': {
    'status': 404,
    'json': {'detail': 'Not Found'},
    'text': '{"detail":"Not Found"}',
  },
  // Dart's JSON parser refuses the non-standard NaN / Infinity literals.
  'NaN literal': {
    'status': 400,
    'json': {'detail': 'request body must be valid JSON'},
    'text': '{"detail":"request body must be valid JSON"}',
  },
  'Infinity in state': {
    'status': 400,
    'json': {'detail': 'request body must be valid JSON'},
    'text': '{"detail":"request body must be valid JSON"}',
  },
};

const _questions = {
  'queue': {
    'type': 'choice',
    'instructions': 'Which team should handle this?',
    'criteria': {
      'billing': 'Payments and invoices',
      'tech': 'Bugs and outages',
      'other': 'Anything else',
    },
  },
  'urgency': {
    'type': 'score',
    'instructions': 'How upset is the customer?',
    'criteria': ['calm', 'firm', 'angry', 'furious'],
  },
};
const _state =
    'I was charged twice for my March invoice and nobody answers my emails.';

class _Throwing implements InferenceBackend {
  @override
  Future<InferenceOutput> run(InferenceBatch batch) async =>
      throw StateError('CUDA out of memory at /secret/path');

  @override
  Future<void> close() async {}
}

/// Counts how many runs overlap, across every backend it wraps.
class _Overlap {
  int running = 0;
  int peak = 0;
}

class _Counted implements InferenceBackend {
  _Counted(this.inner, this.overlap);
  final InferenceBackend inner;
  final _Overlap overlap;

  @override
  Future<InferenceOutput> run(InferenceBatch batch) async {
    overlap.running++;
    if (overlap.running > overlap.peak) overlap.peak = overlap.running;
    try {
      return await inner.run(batch);
    } finally {
      overlap.running--;
    }
  }

  @override
  Future<void> close() => inner.close();
}

void main() {
  group('replays laya-serve (test/fixtures/serve.json)', () {
    final fixture =
        jsonDecode(File('test/fixtures/serve.json').readAsStringSync()) as Map;
    final cases = (fixture['cases'] as List).cast<Map<String, Object?>>();

    test('the fixture covers the cases this suite expects', () {
      expect(cases, hasLength(142));
    });

    for (final c in cases) {
      final name = c['name']! as String;
      test(name, () async {
        final env = (c['env']! as Map).cast<String, String>();
        final engine = _StubEngine();
        final handler = createHandler(
          engine,
          ServerConfig.fromEnvironment(env),
        );
        final spec = c['body'] as Map<String, Object?>?;
        final bytes = spec == null ? null : _bodyBytes(spec);
        final chunked = c['chunked'] == true;
        final reply = await _send(
          handler,
          c['method']! as String,
          c['path']! as String,
          headers: (c['headers']! as Map).cast<String, String>(),
          body: bytes == null
              ? null
              : chunked
              ? Stream.fromIterable([
                  for (var i = 0; i < bytes.length; i += 65536)
                    bytes.sublist(i, (i + 65536).clamp(0, bytes.length)),
                ])
              : bytes,
        );

        final want = _differences[name] ?? c;
        expect(reply.status, want['status'], reason: reply.text);
        if (c['method'] != 'HEAD') {
          // Byte for byte: key order and int/float spelling are what a client
          // sees, and map equality (where 0 == 0.0) would not catch them.
          expect(reply.text, want['text']);
          expect(reply.json, want['json']);
        }
        if (_differences.containsKey(name)) return;

        final got = <String, Object?>{
          for (final h in [
            'content-type',
            'allow',
            'retry-after',
            'www-authenticate',
            'location',
          ])
            if (reply.headers[h] != null) h: reply.headers[h],
          'server-timing?': reply.headers.containsKey('server-timing'),
          'x-inference-time-ms?': reply.headers.containsKey(
            'x-inference-time-ms',
          ),
        };
        expect(got, c['response_headers']);
        if (reply.status == 200 && c['path'] != '/health') {
          expect([
            for (final call in engine.calls) _asLayaCall(call),
          ], c['predict']);
        }
      });
    }

    final depth = fixture['depth']! as Map;
    final max = depth['max_state_depth']! as int;
    Future<_Reply> nested(int d) => _send(
      createHandler(_StubEngine(), ServerConfig.fromEnvironment({})),
      'POST',
      '/v1/systemone',
      body: utf8.encode('{"state":${'[' * d}${']' * d},"questions":{}}'),
    );

    test('accepts a state nested $max deep, as laya-serve does', () async {
      final r = await nested(max);
      expect(r.status, (depth['at_max']! as Map)['status']);
      expect(r.text, (depth['at_max']! as Map)['text']);
      expect(r.json, (depth['at_max']! as Map)['json']);
    });

    test(
      'refuses a state nested ${max + 1} deep, as laya-serve does',
      () async {
        final r = await nested(max + 1);
        expect(r.status, (depth['over_max']! as Map)['status']);
        expect(r.text, (depth['over_max']! as Map)['text']);
        expect(r.json, (depth['over_max']! as Map)['json']);
      },
    );
  });

  test('resolveClientModel strips and lowers as laya-serve does', () {
    // Checked against laya 0.3.24 `serve._resolve_model`: U+001C-U+001F are
    // whitespace to Python's strip(), U+FEFF is not.
    expect(
      resolveClientModel('\x1cconvaiinnovations/laya-multilingual'),
      'multilingual',
    );
    expect(resolveClientModel('\x1cEnglish\x1f'), 'english');
    expect(resolveClientModel('\u{FEFF}english'), isNull);
    expect(
      resolveClientModel('\u{FEFF}convaiinnovations/laya-multilingual'),
      isNull,
    );
  });

  group('mounted under a shelf prefix', () {
    Future<_Reply> send(
      Handler inner,
      String method,
      String path, {
      Object? body,
    }) {
      // What a router or Cascade mounting the handler at /laya does.
      Response notFound(Request _) => Response.notFound('outside the mount');
      FutureOr<Response> mounted(Request r) =>
          r.url.pathSegments.firstOrNull == 'laya'
          ? inner(r.change(path: 'laya'))
          : notFound(r);
      return _send(mounted, method, path, body: body);
    }

    final request = utf8.encode(
      jsonEncode({'state': _state, 'questions': _questions}),
    );

    test('serves /laya/health and /laya/v1/systemone', () async {
      final handler = createHandler(_StubEngine(), const ServerConfig());
      final health = await send(handler, 'GET', '/laya/health');
      expect(health.status, 200, reason: health.text);
      expect((health.json! as Map)['status'], 'ok');

      final answer = await send(
        handler,
        'POST',
        '/laya/v1/systemone',
        body: request,
      );
      expect(answer.status, 200, reason: answer.text);
      expect((answer.json! as Map)['answers'], contains('queue'));
    });

    test('404s, 405s and redirects below the mount point', () async {
      final handler = createHandler(_StubEngine(), const ServerConfig());
      final missing = await send(handler, 'GET', '/laya/nope');
      expect(missing.status, 404);
      expect(missing.text, '{"detail":"Not Found"}');
      final wrongMethod = await send(handler, 'GET', '/laya/v1/systemone');
      expect(wrongMethod.status, 405);
      final slash = await send(handler, 'GET', '/laya/health/');
      expect(slash.status, 307);
      expect(slash.headers['location'], 'http://testserver/laya/health');
    });

    test('LAYA_ROOT_PATH is still stripped below the mount point', () async {
      final handler = createHandler(
        _StubEngine(),
        const ServerConfig(rootPath: '/api'),
      );
      expect((await send(handler, 'GET', '/laya/api/health')).status, 200);
      expect((await send(handler, 'GET', '/laya/health')).status, 200);
    });
  });

  group('over a real engine', skip: tokenizerSkip('english'), () {
    late LayaTokenizer tok;
    setUpAll(
      () async => tok = await LayaTokenizer.load(tokenizerPath('english')),
    );

    late Map<String, InferenceBackend Function()> backends;
    late List<LogRecord> logs;
    late StreamSubscription<LogRecord> logSub;
    setUp(() {
      backends = {};
      logs = [];
      Logger.root.level = Level.ALL;
      logSub = Logger.root.onRecord.listen(logs.add);
    });
    tearDown(() => logSub.cancel());

    InferenceBackend ascending({Duration delay = Duration.zero}) => FakeBackend(
      (r, k) => [for (var i = 0; i < k; i++) i.toDouble()],
      delay: delay,
    );

    LayaEngine engine() {
      final e = LayaEngine(
        loader: (name) async {
          final spec = CheckpointSpec.all[name]!;
          return LayaAgent(
            spec: spec,
            tokenizer: tok,
            config: CheckpointConfig(
              maxLen: spec.maxLen,
              headMaxLen: spec.headMaxLen,
              temperature: [1.0, 1.0, 1.0],
              temperatureByOptions: const {},
            ),
            backend: (backends[name] ?? ascending)(),
          );
        },
      );
      addTearDown(e.close);
      return e;
    }

    Handler handler([Map<String, String> env = const {}, LayaEngine? e]) =>
        createHandler(e ?? engine(), ServerConfig.fromEnvironment(env));

    test('answers the queue/urgency example with the full payload', () async {
      final r = await _post(handler(), {
        'state': _state,
        'questions': _questions,
      });
      expect(r.status, 200, reason: r.text);
      expect(r.headers['content-type'], 'application/json');
      final timing = RegExp(
        r'^inference;dur=(\d+\.\d{2})$',
      ).firstMatch(r.headers['server-timing']!);
      expect(timing, isNotNull, reason: r.headers['server-timing']);
      expect(r.headers['x-inference-time-ms'], timing!.group(1));
      final body = r.json! as Map;
      expect(body.keys, ['model', 'answers', 'usage', 'routing']);
      expect(body['model'], 'laya-rl-agent');
      final answers = body['answers']! as Map;
      expect(answers.keys, ['queue', 'urgency']);
      expect((answers['queue']! as Map).keys, [
        'type',
        'choice',
        'probabilities',
        'confidence',
        'answer_confidence',
        'action',
      ]);
      expect((answers['urgency']! as Map).keys, [
        'type',
        'score',
        'legend',
        'probabilities',
        'confidence',
        'answer_confidence',
        'action',
      ]);
      // FakeBackend's ascending logits favour the last option.
      expect((answers['queue']! as Map)['choice'], 'other');
      expect((body['usage']! as Map).keys, [
        'input_tokens',
        'output_tokens',
        'state_tokens',
        'state_tokens_dropped',
        'truncated',
        'truncated_questions',
      ]);
      expect((body['routing']! as Map)['model'], 'english');
    });

    test('LAYA_JEV_STRICT=1 serves the strict projection', () async {
      final r = await _post(handler({'LAYA_JEV_STRICT': '1'}), {
        'state': _state,
        'questions': {
          ..._questions,
          'refund': {'type': 'noul', 'instructions': 'Wants a refund?'},
        },
      });
      expect(r.status, 200);
      final body = r.json! as Map;
      expect(body.keys, ['model', 'answers', 'usage']);
      final answers = body['answers']! as Map;
      expect((answers['queue']! as Map).keys, [
        'type',
        'choice',
        'confidence',
        'probabilities',
      ]);
      expect((answers['urgency']! as Map).keys, [
        'type',
        'score',
        'confidence',
        'probabilities',
        'legend',
      ]);
      expect((answers['refund']! as Map).keys, ['type', 'noul']);
      expect((body['usage']! as Map).keys, ['input_tokens', 'output_tokens']);
      expect(r.headers['server-timing'], startsWith('inference;dur='));
    });

    test('a min_confidence map gates each answer at its bucket', () async {
      final r = await _post(handler(), {
        'state': _state,
        'questions': _questions,
        'min_confidence': {'choice:3-5': 1.0, 'default': 0.0},
      });
      expect(r.status, 200, reason: r.text);
      final answers = (r.json! as Map)['answers']! as Map;
      final queue = answers['queue']! as Map;
      expect(queue['abstention'], 'abstained');
      expect(queue['abstention_threshold'], 1.0);
      final urgency = answers['urgency']! as Map;
      expect(urgency['abstention'], 'passed');
      expect(urgency['abstention_threshold'], 0.0);
    });

    test('an unknown task is laya\'s 422', () async {
      final r = await _post(handler(), {
        'state': _state,
        'questions': _questions,
        'task': 'nope',
      });
      expect(r.status, 422);
      expect(
        (r.json! as Map)['detail'],
        startsWith("unknown model 'nope'; choose one of"),
      );
    });

    test('/health reports the loaded checkpoints in full', () async {
      final e = engine();
      final h = handler(const {}, e);
      var health = await _send(h, 'GET', '/health');
      expect(health.json, {
        'status': 'ok',
        'loaded': <Object?>[],
        'revisions': <String, Object?>{},
        'device': 'cpu',
        'device_is_preference': true,
        'checkpoint_devices': <String, Object?>{},
        'cpu_fallbacks': <String, Object?>{},
      });
      expect(
        (await _post(h, {'state': _state, 'questions': _questions})).status,
        200,
      );
      health = await _send(h, 'GET', '/health');
      expect(health.status, 200);
      expect(health.headers['content-type'], 'application/json');
      expect(health.text, contains('"device_is_preference":false'));
      expect(health.json, {
        'status': 'ok',
        'loaded': ['english'],
        'revisions': {'english': CheckpointSpec.all['english']!.onnxRevision},
        'device': 'cpu',
        'device_is_preference': false,
        'checkpoint_devices': {'english': 'cpu'},
        'cpu_fallbacks': {
          'english': {'count': 0, 'last_reason': null},
        },
      });
    });

    test(
      'a key-locked /health answers liveness only without the key',
      () async {
        final h = handler({'LAYA_API_KEY': 'k'});
        expect((await _send(h, 'GET', '/health')).text, '{"status":"ok"}');
        final full = await _send(
          h,
          'GET',
          '/health',
          headers: {'authorization': 'Bearer k'},
        );
        expect((full.json! as Map).keys, [
          'status',
          'loaded',
          'revisions',
          'device',
          'device_is_preference',
          'checkpoint_devices',
          'cpu_fallbacks',
        ]);
        // A non-ASCII byte (latin-1 decoded, as dart:io gives it) is a 401.
        final odd = await _post(
          h,
          {'state': _state, 'questions': _questions},
          headers: {'authorization': 'Bearer sécret'},
        );
        expect(odd.status, 401);
        expect(odd.json, {'detail': 'invalid or missing bearer token'});
      },
    );

    test(
      'an inference failure is a 500 with the cause only in the log',
      () async {
        backends['english'] = _Throwing.new;
        final r = await _post(
          handler({'LAYA_API_KEY': 'sekrit'}),
          {'state': _state, 'questions': _questions, 'model': 'english'},
          headers: {'authorization': 'Bearer sekrit'},
        );
        expect(r.status, 500);
        expect(r.json, {'detail': 'inference failed'});
        expect(r.text, isNot(contains('secret')));
        final severe = logs.where((l) => l.level == Level.SEVERE).toList();
        expect(severe, hasLength(1));
        expect(severe.single.loggerName, 'laya_server.server');
        expect(severe.single.message, 'inference failed for model=english');
        expect(severe.single.error, isA<StateError>());
        expect(severe.single.stackTrace, isNotNull);
        // Neither the body nor the key is logged.
        for (final l in logs) {
          expect('${l.message} ${l.error}', isNot(contains('sekrit')));
          expect('${l.message} ${l.error}', isNot(contains('March invoice')));
        }
      },
    );

    test('an auto-routed failure logs model=None, as laya-serve', () async {
      backends['english'] = _Throwing.new;
      final r = await _post(handler(), {
        'state': _state,
        'questions': _questions,
        'model': 'jev-1',
      });
      expect(r.status, 500);
      expect(
        logs.singleWhere((l) => l.level == Level.SEVERE).message,
        'inference failed for model=None',
      );
    });

    test('a checkpoint that is not installed is a 500, logged', () async {
      final e = LayaEngine(
        loader: (name) async => throw StateError(
          'checkpoint $name is not installed in /m; '
          'run: dart run laya_server:laya_fetch $name',
        ),
      );
      addTearDown(e.close);
      final r = await _post(handler(const {}, e), {
        'state': _state,
        'questions': _questions,
      });
      expect(r.status, 500);
      expect(r.json, {'detail': 'inference failed'});
      expect(
        logs.singleWhere((l) => l.level == Level.SEVERE).error.toString(),
        contains('laya_server:laya_fetch english'),
      );
    });

    test('a NaN in the result is a 500, logged', () async {
      backends['english'] = () =>
          FakeBackend((r, k) => [for (var i = 0; i < k; i++) double.nan]);
      final r = await _post(handler(), {
        'state': _state,
        'questions': _questions,
      });
      expect(r.status, 500);
      expect(r.json, {'detail': 'inference failed'});
      expect(logs.where((l) => l.level == Level.SEVERE), hasLength(1));
    });

    test(
      'a Content-Length over 2 MiB is refused before the body is read',
      () async {
        var listened = false;
        final body = Stream<List<int>>.fromIterable([utf8.encode('{}')])
            .transform<List<int>>(
              StreamTransformer.fromHandlers(
                handleData: (d, sink) {
                  listened = true;
                  sink.add(d);
                },
              ),
            );
        final r = await _send(
          handler(),
          'POST',
          '/v1/systemone',
          headers: {'content-length': '${2 * 1024 * 1024 + 1}'},
          body: body,
        );
        expect(r.status, 413);
        expect(r.json, {'detail': 'request body too large'});
        expect(listened, isFalse);
      },
    );

    test(
      'request 2 past LAYA_MAX_CONCURRENT=1 gets 503, /health stays fast',
      () async {
        backends['english'] = () =>
            ascending(delay: const Duration(milliseconds: 300));
        final h = handler({'LAYA_MAX_CONCURRENT': '1'});
        final first = _post(h, {'state': _state, 'questions': _questions});
        await Future<void>.delayed(const Duration(milliseconds: 50));
        final busy = await _post(h, {'state': _state, 'questions': _questions});
        expect(busy.status, 503);
        expect(busy.json, {'detail': 'server busy, try again later'});
        expect(busy.headers['retry-after'], '1');
        final watch = Stopwatch()..start();
        final health = await _send(h, 'GET', '/health');
        watch.stop();
        expect(health.status, 200);
        expect(watch.elapsedMilliseconds, lessThan(50));
        expect((await first).status, 200);
        // The slot is released once the first request is answered.
        expect(
          (await _post(h, {'state': _state, 'questions': _questions})).status,
          200,
        );
      },
    );

    test(
      'the 17th concurrent request is refused at the default of 16',
      () async {
        final gate = Completer<void>();
        final h = handler();
        final slow = [
          for (var i = 0; i < 16; i++)
            _send(
              h,
              'POST',
              '/v1/systemone',
              body: (() async* {
                await gate.future;
                yield utf8.encode(
                  jsonEncode({'state': 'x', 'questions': <String, Object?>{}}),
                );
              })(),
            ),
        ];
        await Future<void>.delayed(const Duration(milliseconds: 20));
        final r = await _post(h, {
          'state': 'x',
          'questions': <String, Object?>{},
        });
        expect(r.status, 503);
        expect(r.headers['retry-after'], '1');
        gate.complete();
        for (final s in await Future.wait(slow)) {
          expect(s.status, 200);
        }
      },
    );

    test('a slow-streaming body holds admission but not inference', () async {
      final h = handler({'LAYA_MAX_CONCURRENT': '2'});
      final body = StreamController<List<int>>();
      final slow = _send(h, 'POST', '/v1/systemone', body: body.stream);
      await Future<void>.delayed(const Duration(milliseconds: 20));
      // The second slot is free, and inference is not blocked by the slow body.
      final watch = Stopwatch()..start();
      final fast = await _post(h, {'state': _state, 'questions': _questions});
      expect(fast.status, 200);
      expect(watch.elapsedMilliseconds, lessThan(2000));
      // ...but the slow body still holds the first slot.
      final h1 = handler({'LAYA_MAX_CONCURRENT': '1'});
      final body1 = StreamController<List<int>>();
      final slow1 = _send(h1, 'POST', '/v1/systemone', body: body1.stream);
      await Future<void>.delayed(const Duration(milliseconds: 20));
      expect(
        (await _post(h1, {'state': _state, 'questions': _questions})).status,
        503,
      );
      for (final c in [body, body1]) {
        c.add(
          utf8.encode(
            jsonEncode({'state': 'x', 'questions': <String, Object?>{}}),
          ),
        );
        await c.close();
      }
      expect((await slow).status, 200);
      expect((await slow1).status, 200);
    });

    test('inference is serialised across checkpoints', () async {
      final overlap = _Overlap();
      for (final name in ['english', 'multilingual']) {
        backends[name] = () => _Counted(
          ascending(delay: const Duration(milliseconds: 100)),
          overlap,
        );
      }
      final h = handler();
      final replies = await Future.wait([
        for (final model in ['english', 'multilingual', 'english'])
          _post(h, {'state': _state, 'questions': _questions, 'model': model}),
      ]);
      expect([for (final r in replies) r.status], [200, 200, 200]);
      expect(overlap.peak, 1);
    });
  });
}
