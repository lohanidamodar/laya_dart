/// Bit-exact float32 arithmetic as numpy does it, for `decodeAnswer`.
///
/// laya decodes the head's float32 output with numpy, and numpy's float32
/// `exp` and `log` are not correctly rounded: numpy 2.x on x86-64 with AVX2
/// and FMA3 (and on every Windows build, where the AVX512F kernels are
/// compiled out) runs its own SIMD kernels, `simd_exp_FLOAT` and
/// `simd_log_FLOAT` in `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`
/// (numpy v2.5.3), which are up to 2.52 and 3.83 ulp off. Those differences
/// reach the fourth decimal often enough to break parity, so this ports the
/// kernels operation for operation: every `_mm256_*_ps` rounds to float32,
/// and every `_mm256_fmadd_ps` rounds once.
///
/// The constants are the C `float` literals of `npy_simd_data.h` and
/// `npy_math.h`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

final _f32Box = Float32List(1);
final _f32Bits = Uint32List.view(_f32Box.buffer);
final _f64Box = Float64List(1);
final _f64Bits = Int64List.view(_f64Box.buffer);

/// Rounds [x] to the nearest float32 (ties to even).
double f32(double x) {
  _f32Box[0] = x;
  return _f32Box[0];
}

/// `fmaf(a, b, c)` on float32 operands: `a * b + c` with one rounding.
///
/// The float32 product is exact in a double; the sum is then rounded to odd
/// in double (53 bits), which makes the final rounding to float32 (24 bits)
/// a correct single rounding.
double fmaF32(double a, double b, double c) {
  final p = a * b;
  final s = p + c;
  if (!s.isFinite) return f32(s);
  // TwoSum: the exact error of `p + c`.
  final bb = s - p;
  final err = (p - (s - bb)) + (c - bb);
  if (err == 0) return f32(s);
  _f64Box[0] = s;
  final bits = _f64Bits[0];
  if (bits.isEven) {
    // Round to odd: step one ulp towards the exact value.
    _f64Bits[0] = (err > 0) == (s > 0) ? bits + 1 : bits - 1;
  }
  return f32(_f64Box[0]);
}

// npy_simd_data.h and npy_math.h, as C rounds the float literals.
final _xmax = f32(88.72283935546875);
final _xmin = f32(-103.97208404541015625);
final _cvtMagic = f32(12582912.0); // 0x1.8p+23
final _log2e = f32(1.442695040888963407359924681001892137);
final _codyW1 = f32(-6.93145752e-1);
final _codyW2 = f32(-1.42860677e-6);
final _expP = [
  9.999999999980870924916e-01,
  7.257664613233124478488e-01,
  2.473615434895520810817e-01,
  5.114512081637298353406e-02,
  6.757896990527504603057e-03,
  5.082762527590693718096e-04,
].map(f32).toList();
final _expQ = [
  1.000000000000000000000e+00,
  -2.742335390411667452936e-01,
  2.159509375685829852307e-02,
].map(f32).toList();
final _logP = [
  0.000000000000000000000e+00,
  9.999999999999998702752e-01,
  2.112677543073053063722e+00,
  1.480000633576506585156e+00,
  3.808837741388407920751e-01,
  2.589979117907922693523e-02,
].map(f32).toList();
final _logQ = [
  1.000000000000000000000e+00,
  2.612677543073109236779e+00,
  2.453006071784736363091e+00,
  9.864942958519418960339e-01,
  1.546476374983906719538e-01,
  5.875095403124574342950e-03,
].map(f32).toList();
final _loge2 = f32(0.693147180559945309417232121458176568);
final _sqrt1_2 = f32(0.707106781186547524400844362104849039);

/// numpy's float32 `np.exp` (`simd_exp_FLOAT`, AVX2/FMA3 path).
double npExpF32(double input) {
  final x = f32(input);
  if (x.isNaN) return double.nan;
  if (x >= _xmax) return double.infinity;
  if (x <= _xmin) return 0.0;
  // Round x / ln 2 to an integer with the 1.5 * 2^23 trick. The source has
  // `_mm256_mul_ps` then `_mm256_add_ps`, but the compiler contracts them
  // into one FMA: numpy's exp(-72.43388f) only matches with the fused form,
  // where -104.500003 rounds to -105 instead of first rounding to -104.5.
  final q = f32(fmaF32(x, _log2e, _cvtMagic) - _cvtMagic);
  // Cody-Waite range reduction; the third step's c3 is 0.
  var r = fmaF32(q, _codyW1, x);
  r = fmaF32(q, _codyW2, r);
  r = fmaF32(q, 0.0, r);
  var num = fmaF32(_expP[5], r, _expP[4]);
  num = fmaF32(num, r, _expP[3]);
  num = fmaF32(num, r, _expP[2]);
  num = fmaF32(num, r, _expP[1]);
  num = fmaF32(num, r, _expP[0]);
  var den = fmaF32(_expQ[2], r, _expQ[1]);
  den = fmaF32(den, r, _expQ[0]);
  final poly = f32(num / den);
  // fma_scalef_ps: an exact exponent add, or for q <= -125 one rounding
  // into the subnormals; both equal the float32 rounding of poly * 2^q.
  return f32(poly * math.pow(2.0, q));
}

/// numpy's float32 `np.log` (`simd_log_FLOAT`, AVX2/FMA3 path).
double npLogF32(double input) {
  final x = f32(input);
  if (x.isNaN || x < 0) return double.nan;
  if (x == 0) return double.negativeInfinity;
  if (x == double.infinity) return double.infinity;
  // fma_get_exponent / fma_get_mantissa: x = m * 2^e with m in [0.5, 1);
  // a subnormal is scaled by 2^100 first.
  var scaled = x;
  var bias = 0.0;
  if (x < 1.1754943508222875e-38) {
    scaled = f32(x * 1.2676506002282294e30); // 2^100
    bias = 100.0;
  }
  _f32Box[0] = scaled;
  final bits = _f32Bits[0];
  var e = ((bits >> 23) - 126) - bias;
  _f32Bits[0] = (bits & 0x7fffff) | (126 << 23);
  var m = _f32Box[0];
  if (m <= _sqrt1_2) {
    m = f32(m + m);
    e = e - 1;
  }
  m = f32(m - 1.0);
  var num = fmaF32(_logP[5], m, _logP[4]);
  num = fmaF32(num, m, _logP[3]);
  num = fmaF32(num, m, _logP[2]);
  num = fmaF32(num, m, _logP[1]);
  num = fmaF32(num, m, _logP[0]);
  var den = fmaF32(_logQ[5], m, _logQ[4]);
  den = fmaF32(den, m, _logQ[3]);
  den = fmaF32(den, m, _logQ[2]);
  den = fmaF32(den, m, _logQ[1]);
  den = fmaF32(den, m, _logQ[0]);
  final poly = f32(num / den);
  return fmaF32(e, _loge2, poly);
}

/// numpy's `add.reduce` over a contiguous row: `pairwise_sum` in
/// `loops_utils.h.src`, accumulating in float32, or in double when
/// [inDouble] is set (a float64 row).
double npPairwiseSum(List<double> a, {bool inDouble = false}) {
  double add(double x, double y) => inDouble ? x + y : f32(x + y);

  double sum(int start, int n) {
    if (n < 8) {
      var res = 0.0;
      for (var i = 0; i < n; i++) {
        res = add(res, a[start + i]);
      }
      return res;
    }
    if (n <= 128) {
      final r = List<double>.generate(8, (j) => a[start + j]);
      var i = 8;
      for (; i < n - (n % 8); i += 8) {
        for (var j = 0; j < 8; j++) {
          r[j] = add(r[j], a[start + i + j]);
        }
      }
      var res = add(
        add(add(r[0], r[1]), add(r[2], r[3])),
        add(add(r[4], r[5]), add(r[6], r[7])),
      );
      for (; i < n; i++) {
        res = add(res, a[start + i]);
      }
      return res;
    }
    var n2 = n ~/ 2;
    n2 -= n2 % 8;
    return add(sum(start, n2), sum(start + n2, n - n2));
  }

  // The reduction starts from the identity 0, which changes nothing but -0.
  return add(0.0, sum(0, a.length));
}
