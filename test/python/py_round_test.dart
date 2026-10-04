import 'package:laya_server/src/python/py_round.dart';
import 'package:test/test.dart';

void main() {
  group('round4 is Python round(x, 4)', () {
    test('brief vectors', () {
      expect(round4(0.12345), 0.1235); // stored just above .12345
      expect(round4(2.675e-1), 0.2675);
      // The brief expected 1.0 here, but 1.00005 is stored just above the
      // midpoint and Python round(1.00005, 4) is 1.0001.
      expect(round4(1.00005), 1.0001);
      expect(round4(2.00005), 2.0); // stored below the midpoint
      expect(round4(0.50005), 0.5); // stored below the midpoint
      expect(round4(0.00015), 0.0001); // stored below the midpoint
    });

    test('exact binary ties round half to even', () {
      expect(round4(0.03125), 0.0312); // 1/32
      expect(round4(0.09375), 0.0938); // 3/32
      expect(round4(0.15625), 0.1562); // 5/32
      expect(round4(1.03125), 1.0312);
      expect(round4(2.65625), 2.6562);
    });

    test('plain values', () {
      expect(round4(0.0), 0.0);
      expect(round4(1.0), 1.0);
      expect(round4(0.25), 0.25);
      expect(round4(0.99996), 1.0);
      expect(round4(4.9e-5), 0.0);
      expect(round4(7.123456), 7.1235);
    });
  });

  group('pyRoundHalfEven is Python round(x)', () {
    test('exact .5 ties go to the even integer', () {
      expect(pyRoundHalfEven(0.5), 0);
      expect(pyRoundHalfEven(1.5), 2);
      expect(pyRoundHalfEven(2.5), 2);
      expect(pyRoundHalfEven(3.5), 4);
      expect(pyRoundHalfEven(12.5), 12);
      expect(pyRoundHalfEven(-0.5), 0);
      expect(pyRoundHalfEven(-1.5), -2);
      expect(pyRoundHalfEven(-2.5), -2);
    });

    test('off-tie values round to the nearest', () {
      expect(pyRoundHalfEven(2.4999999999999996), 2);
      expect(pyRoundHalfEven(2.5000000000000004), 3);
      expect(pyRoundHalfEven(0.49999999999999994), 0);
      expect(pyRoundHalfEven(99.6), 100);
      expect(pyRoundHalfEven(-3.7), -4);
      expect(pyRoundHalfEven(7.0), 7);
    });
  });
}
