// laya_server on the onnx-community graphs against laya-serve (torch) on the upstream weights.
//
// The fixtures are tool/parity.dart's recording of laya-serve; this replays each recorded body
// through the real handler over LayaEngine.fileLoader('models') and compares the responses.
//
// Run: dart test --run-skipped -t model test/parity
@Tags(['model'])
library;

import 'dart:convert';
import 'dart:io';

import 'package:laya_server/laya_server.dart';
import 'package:shelf/shelf.dart';
import 'package:test/test.dart';

/// ONNX Runtime against torch at fp32, after laya's 4-decimal rounding.
const _tolerance = 2e-4;

/// The checkpoints whose weights this run did not have, and why.
const _untested = {
  'typed-decisions':
      'typed-decisions was not downloaded, so it is not tested on real weights',
};

bool _installed(String name) => [
  'models/$name/tokenizer.json',
  'models/$name/config.json',
  'models/$name/onnx/model.onnx',
  'models/$name/onnx/model.onnx_data',
].every((p) => File(p).existsSync());

/// What a comparison found, beyond pass or fail.
class _Stats {
  double maxDelta = 0;
  String maxAt = '';
  final nearTies = <String>[];
  int compared = 0;

  /// Cases whose response was compared in full.
  int cases = 0;
}

/// Asserts [got] matches the reference [ref] at [path]: equal key sets and
/// strings, numbers within [_tolerance], `usage` and `routing` exact, and a
/// `choice` equal unless the reference's top two probabilities are a
/// near-tie. The top-level `model` may differ.
void _compare(Object? ref, Object? got, String path, _Stats stats) {
  if (ref is Map) {
    expect(got, isA<Map<Object?, Object?>>(), reason: path);
    got as Map<Object?, Object?>;
    expect(got.keys.toSet(), ref.keys.toSet(), reason: '$path keys');
    for (final key in ref.keys) {
      final at = '$path.$key';
      if (at == r'$.model') continue;
      if (at == r'$.usage' || at == r'$.routing') {
        expect(got[key], ref[key], reason: at);
      } else if (key == 'choice' && ref['probabilities'] is Map) {
        if (got[key] != ref[key]) {
          final p = (ref['probabilities'] as Map).values.cast<num>().toList()
            ..sort((a, b) => b.compareTo(a));
          final tie = p.length > 1 && (p[0] - p[1]).abs() <= _tolerance;
          expect(tie, isTrue, reason: '$at: ${got[key]} != ${ref[key]}');
          stats.nearTies.add(at);
        }
      } else {
        _compare(ref[key], got[key], at, stats);
      }
    }
  } else if (ref is List) {
    expect(got, isA<List<Object?>>(), reason: path);
    got as List<Object?>;
    expect(got.length, ref.length, reason: '$path length');
    for (var i = 0; i < ref.length; i++) {
      _compare(ref[i], got[i], '$path[$i]', stats);
    }
  } else if (ref is num) {
    expect(got, isA<num>(), reason: path);
    final delta = (ref - (got as num)).abs().toDouble();
    stats.compared++;
    if (delta > stats.maxDelta) {
      stats.maxDelta = delta;
      stats.maxAt = path;
    }
    expect(delta, lessThanOrEqualTo(_tolerance), reason: '$path: $got vs $ref');
  } else {
    expect(got, ref, reason: path);
  }
}

void main() {
  for (final name in ['english', 'multilingual', 'typed-decisions']) {
    final fixture = File('test/fixtures/parity_$name.json');
    final skip =
        _untested[name] ??
        (!fixture.existsSync()
            ? 'no parity fixture; run dart run tool/parity.dart'
            : !_installed(name)
            ? 'models/$name is not installed; run dart run laya_server:laya_fetch $name'
            : null);
    if (skip != null) {
      // Not `skip:`, which the --run-skipped this suite needs would override.
      test('$name parity', () => markTestSkipped(skip));
      continue;
    }
    group(name, () {
      late LayaEngine engine;
      late Handler handler;
      final stats = _Stats();
      final cases =
          ((jsonDecode(fixture.readAsStringSync())
                      as Map<String, Object?>)['cases']
                  as List)
              .cast<Map<String, Object?>>();

      setUpAll(() {
        engine = LayaEngine(loader: LayaEngine.fileLoader('models'));
        handler = createHandler(engine, const ServerConfig());
      });
      tearDownAll(() async {
        await engine.close();
        // Read by the person recording the results in doc/api.md.
        print(
          'PARITY $name: ${cases.length} cases, ${stats.compared} numbers, '
          'max |delta| ${stats.maxDelta} at ${stats.maxAt}, '
          'near-ties ${stats.nearTies}',
        );
      });

      for (final c in cases) {
        test('request ${c['request']} ${c['variant']}', () async {
          final resp = await handler(
            Request(
              'POST',
              Uri.parse('http://localhost/v1/systemone'),
              body: c['body'] as String,
              headers: {'content-type': 'application/json'},
            ),
          );
          final text = await resp.readAsString();
          expect(resp.statusCode, c['status'], reason: text);
          _compare(c['json'], jsonDecode(text), r'$', stats);
          stats.cases++;
        });
      }

      // Runs last: an empty fixture or a run that compared nothing fails
      // instead of passing vacuously.
      test('compared at least one case', () {
        expect(cases, isNotEmpty);
        expect(stats.cases, greaterThan(0));
        expect(stats.compared, greaterThan(0));
      });
    });
  }
}
