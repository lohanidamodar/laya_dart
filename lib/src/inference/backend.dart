/// The inference backend interface, its output, and a fake for tests.
library;

import 'dart:typed_data';

import 'batch.dart';

/// Per-row outputs of the decision head for one [InferenceBatch].
///
/// Storage stays float32, as ONNX Runtime returns it: decoding rounds through
/// float32 to match laya's numpy arithmetic.
class InferenceOutput {
  /// Wraps the head's `logits` (`[rows, markers]`) and `act_logits`
  /// (`[rows, 2]`), both row-major.
  InferenceOutput(this.logits, this.actLogits, this.markers);

  /// `logits`, row-major `[rows, markers]`.
  final Float32List logits;

  /// `act_logits`, row-major `[rows, 2]`.
  final Float32List actLogits;

  /// The batch's marker width, K: the row stride of [logits].
  final int markers;

  /// The first [k] logits of [row]: its question's option logits in slot
  /// order. Throws [RangeError] for a row out of range or `k > markers`.
  List<double> logitsFor(int row, int k) {
    RangeError.checkValueInInterval(k, 0, markers, 'k');
    RangeError.checkValidIndex(row, null, 'row', actLogits.length ~/ 2);
    final start = row * markers;
    return logits.sublist(start, start + k);
  }

  /// The act logits of [row]. Throws [RangeError] for a row out of range.
  List<double> actFor(int row) {
    RangeError.checkValidIndex(row, null, 'row', actLogits.length ~/ 2);
    return actLogits.sublist(row * 2, row * 2 + 2);
  }
}

/// Runs a checkpoint's graph. Implementations must not block the caller's
/// isolate.
abstract interface class InferenceBackend {
  /// Runs [batch] and returns the head's outputs for each of its rows.
  ///
  /// Throws [StateError] with message `backend closed` after [close].
  Future<InferenceOutput> run(InferenceBatch batch);

  /// Releases the backend. Idempotent.
  Future<void> close();
}

/// Deterministic stand-in for tests and examples: logits from a
/// caller-supplied function.
class FakeBackend implements InferenceBackend {
  /// A backend whose row `r` logits are `logits(r, batch.markers)`, zero
  /// padded to the batch's marker width, after waiting [delay].
  ///
  /// Act logits come from [act] when given, and are `[0, 0]` otherwise.
  FakeBackend(this.logits, {this.delay = Duration.zero, this.act});

  /// Produces the logits of a row, given the row index and the batch's marker
  /// width; it may return fewer entries than that, but not more.
  final List<double> Function(int row, int markers) logits;

  /// Produces the two act logits of a row.
  final List<double> Function(int row)? act;

  /// How long each [run] waits before answering.
  final Duration delay;

  bool _closed = false;

  @override
  Future<InferenceOutput> run(InferenceBatch batch) async {
    if (_closed) throw StateError('backend closed');
    await Future<void>.delayed(delay);
    final k = batch.markers;
    final out = Float32List(batch.rows * k);
    final actOut = Float32List(batch.rows * 2);
    for (var r = 0; r < batch.rows; r++) {
      final row = logits(r, k);
      if (row.length > k) {
        throw ArgumentError.value(
          row.length,
          'logits',
          'row $r has more entries than the batch has markers ($k)',
        );
      }
      out.setAll(r * k, row);
      final a = act?.call(r) ?? const [0.0, 0.0];
      if (a.length != 2) {
        throw ArgumentError.value(a.length, 'act', 'row $r must have 2');
      }
      actOut.setAll(r * 2, a);
    }
    return InferenceOutput(out, actOut, k);
  }

  @override
  Future<void> close() async => _closed = true;
}
