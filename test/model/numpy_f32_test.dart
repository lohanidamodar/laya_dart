import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:laya_server/src/model/numpy_f32.dart';
import 'package:test/test.dart';

void main() {
  final fixture =
      jsonDecode(File('test/fixtures/numpy_f32.json').readAsStringSync())
          as Map;
  final box = Float32List(1);
  final bits = Uint32List.view(box.buffer);
  double fromBits(int b) {
    bits[0] = b;
    return box[0];
  }

  int toBits(double x) {
    box[0] = x;
    return bits[0];
  }

  for (final c in (fixture['cases'] as List).cast<Map<String, Object?>>()) {
    final fn = c['fn']! as String;
    test('np.$fn float32 is reproduced bit for bit', () {
      final ins = (c['in']! as List).cast<int>();
      final outs = (c['out']! as List).cast<int>();
      final bad = <String>[];
      for (var i = 0; i < ins.length; i++) {
        final x = fromBits(ins[i]);
        final got = fn == 'exp' ? npExpF32(x) : npLogF32(x);
        if (toBits(got) != outs[i]) {
          bad.add('$fn($x) = $got, numpy ${fromBits(outs[i])}');
        }
      }
      expect(bad, isEmpty, reason: bad.take(5).join('\n'));
    });
  }

  test('fmaF32 rounds the exact a * b + c once', () {
    // (1 + 2^-12)^2 - 2^-11 = 1 + 2^-24 exactly: a float32 tie, to even.
    const a = 1.0 + 1.0 / (1 << 12);
    expect(fmaF32(a, a, -1.0 / (1 << 11)), 1.0);
    // With c = 0 it is the float32 rounding of the exact product.
    expect(fmaF32(a, a, 0.0), f32(a * a));
    expect(fmaF32(a, a, 0.0), isNot(a * a));
    expect(fmaF32(0.5, 2.0, 1.0), 2.0);
  });

  test('npPairwiseSum follows numpy pairwise_sum', () {
    expect(npPairwiseSum([1, 2, 3]), 6);
    expect(
      npPairwiseSum(List<double>.filled(9, 0.1), inDouble: true),
      ((0.1 + 0.1) + (0.1 + 0.1)) + ((0.1 + 0.1) + (0.1 + 0.1)) + 0.1,
    );
  });
}
