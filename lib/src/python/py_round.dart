/// Python's float rounding: `round(x)` and `round(x, 4)`, ties to even.
library;

import 'dart:typed_data';

/// Python's `round(x)` for a finite float, and `'%.0f' % x`: the nearest
/// integer, an exact tie going to the even one.
int pyRoundHalfEven(double x) {
  final floor = x.floorToDouble();
  final diff = x - floor; // exact for a finite double
  final f = floor.toInt();
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f.isEven ? f : f + 1;
}

/// Python's `round(x, 4)`: the decimal nearest the exact binary value of [x],
/// with an exact tie rounded half to even.
double round4(double x) {
  if (!x.isFinite) return x;
  if (_isTie(x)) {
    // x * 10000 is exactly n + 0.5 here, and representable, so floor is exact.
    final n = (x.abs() * 10000).floor();
    final even = n.isEven ? n : n + 1;
    return x.isNegative ? -even / 10000 : even / 10000;
  }
  // Dart formats the exact binary value, so only exact ties (handled above)
  // can round differently from Python, whose ties go to even.
  return double.parse(x.toStringAsFixed(4));
}

/// Whether [x] is exactly halfway between two 4-decimal values: `x * 20000`
/// is an odd integer.
///
/// With `x = m * 2^e` (m odd), `x * 20000 = m * 625 * 2^(e + 5)`, and 625 is
/// odd, so that is an odd integer exactly when `e == -5`.
bool _isTie(double x) {
  if (x == 0) return false;
  final bits = (ByteData(8)..setFloat64(0, x)).getUint64(0);
  final biased = (bits >> 52) & 0x7ff;
  var m = bits & 0xfffffffffffff;
  var e = biased == 0 ? -1074 : biased - 1075;
  if (biased != 0) m |= 1 << 52;
  while (m.isEven) {
    m >>= 1;
    e++;
  }
  return e == -5;
}
