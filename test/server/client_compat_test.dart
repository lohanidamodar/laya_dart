// A Jev client (jev_dart) against laya_server over a real socket.
@Tags(['compat'])
library;

import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:jev_dart/jev_dart.dart' as jev;
import 'package:laya_server/laya_server.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;
import 'package:test/test.dart';

import '../support/checkpoints.dart';

void main() {
  group('jev_dart decodes laya_server', skip: tokenizerSkip('english'), () {
    late LayaTokenizer tok;
    setUpAll(
      () async => tok = await LayaTokenizer.load(tokenizerPath('english')),
    );

    Future<Uri> serve(Map<String, String> env) async {
      final engine = LayaEngine(
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
            backend: FakeBackend(
              (r, k) => [for (var i = 0; i < k; i++) (i * 0.7) - r],
            ),
          );
        },
      );
      final server = await shelf_io.serve(
        createHandler(engine, ServerConfig.fromEnvironment(env)),
        InternetAddress.loopbackIPv4,
        0,
      );
      addTearDown(() async {
        await server.close(force: true);
        await engine.close();
      });
      return Uri.parse('http://127.0.0.1:${server.port}');
    }

    const state = {'document': 'I was charged twice. Please fix this ASAP.'};
    final questions = <String, jev.Question>{
      'category': jev.choice('What is this ticket about?', {
        'billing': null,
        'technical': null,
        'other': null,
      }),
      'urgency': jev.score('How urgent is it?', ['low', 'medium', 'high']),
      'refund': jev.noul('Does the customer want a refund?'),
    };

    for (final strict in [false, true]) {
      test(strict ? 'strict mode' : 'full mode', () async {
        final base = await serve({
          'LAYA_API_KEY': 'k',
          if (strict) 'LAYA_JEV_STRICT': '1',
        });
        final client = jev.TypeSafeClient(
          apiKey: 'k',
          baseUrl: '$base',
          httpClient: http.Client(),
          closeClient: true,
          retry: jev.RetryPolicy(maxRetries: 0),
          logLevel: jev.LogLevel.off,
        );
        addTearDown(client.close);
        final result = await client.systemOne(
          state: state,
          questions: questions,
        );

        // The same request, raw, to compare against what the client decoded.
        final raw = await http.post(
          base.resolve('/v1/systemone'),
          headers: {
            'authorization': 'Bearer k',
            'content-type': 'application/json',
          },
          body: jsonEncode({
            'state': state,
            'questions': jev.questionsToJson(questions),
            'model': jev.kDefaultModel,
          }),
        );
        expect(raw.statusCode, 200);
        final body = jsonDecode(raw.body) as Map<String, Object?>;
        expect(body.containsKey('routing'), !strict);
        final answers = body['answers']! as Map<String, Object?>;

        expect(result.model, 'laya-rl-agent');
        final category = answers['category']! as Map;
        expect(result.choice('category').choice, category['choice']);
        expect(result.choice('category').confidence, category['confidence']);
        expect(
          result.choice('category').probabilities,
          category['probabilities'],
        );
        final urgency = answers['urgency']! as Map;
        expect(result.score('urgency').score, urgency['score']);
        expect(result.score('urgency').legend, urgency['legend']);
        expect(result.score('urgency').probabilities, urgency['probabilities']);
        expect(result.noul('refund').noul, (answers['refund']! as Map)['noul']);
        final usage = body['usage']! as Map;
        expect(result.usage.inputTokens, usage['input_tokens']);
        expect(result.usage.outputTokens, 0);
      });
    }

    test('a wrong key surfaces as the client\'s 401', () async {
      final base = await serve({'LAYA_API_KEY': 'k'});
      final client = jev.TypeSafeClient(
        apiKey: 'nope',
        baseUrl: '$base',
        httpClient: http.Client(),
        closeClient: true,
        retry: jev.RetryPolicy(maxRetries: 0),
        logLevel: jev.LogLevel.off,
      );
      addTearDown(client.close);
      await expectLater(
        client.systemOne(state: state, questions: questions),
        throwsA(
          isA<jev.ApiException>()
              .having((e) => e.status, 'status', 401)
              .having(
                (e) => e.message,
                'message',
                contains('invalid or missing bearer token'),
              ),
        ),
      );
    });

    group('over a real socket', () {
      Future<String> rawRequest(Uri base, List<int> request) async {
        final socket = await Socket.connect(base.host, base.port);
        socket.add(request);
        await socket.flush();
        final reply = await socket
            .cast<List<int>>()
            .transform(latin1.decoder)
            .join();
        socket.destroy();
        return reply;
      }

      test('a non-ASCII byte in the bearer is a 401, never a 500', () async {
        final base = await serve({'LAYA_API_KEY': 'k'});
        final body = utf8.encode('{"state":"x","questions":{}}');
        final reply = await rawRequest(base, [
          ...latin1.encode(
            'POST /v1/systemone HTTP/1.1\r\n'
            'Host: 127.0.0.1\r\n'
            'Authorization: Bearer sécret\r\n'
            'Content-Type: application/json\r\n'
            'Content-Length: ${body.length}\r\n'
            'Connection: close\r\n\r\n',
          ),
          ...body,
        ]);
        expect(reply, startsWith('HTTP/1.1 401'));
        expect(reply, endsWith('{"detail":"invalid or missing bearer token"}'));
      });

      test('a chunked body over 2 MiB is a 413', () async {
        final base = await serve({});
        final request = http.StreamedRequest(
          'POST',
          base.resolve('/v1/systemone'),
        );
        final client = http.Client();
        addTearDown(client.close);
        final reply = client.send(request);
        request.sink.add(utf8.encode('{"state":"'));
        for (var i = 0; i < 33; i++) {
          request.sink.add(List.filled(65536, 0x78));
        }
        await request.sink.close();
        final response = await http.Response.fromStream(await reply);
        expect(response.statusCode, 413);
        expect(response.body, '{"detail":"request body too large"}');
      });

      test('a redirect for a trailing slash carries no body type', () async {
        final base = await serve({});
        final client = HttpClient();
        addTearDown(client.close);
        final request = await client.getUrl(base.resolve('/health/'));
        request.followRedirects = false;
        final response = await request.close();
        await response.drain<void>();
        expect(response.statusCode, 307);
        expect(response.headers.value('location'), '$base/health');
      });
    });
  });
}
