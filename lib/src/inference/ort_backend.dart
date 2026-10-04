/// The ONNX Runtime inference backend: one session on a dedicated isolate.
///
/// The worker isolate loads the runtime, opens the session and then runs one
/// request at a time, in arrival order, mirroring `laya-serve`'s single
/// inference worker. Tensors cross the isolate boundary as
/// [TransferableTypedData]; errors come back as `(code, message)` and are
/// rethrown as [OrtException].
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:meta/meta.dart';

import 'backend.dart';
import 'batch.dart';
import 'ort/ort_api.dart';
import 'ort/ort_library.dart';
import 'ort/ort_session.dart';

/// The decision graph's inputs, fixed by the onnx-community export.
const _layaInputs = [
  'input_ids',
  'attention_mask',
  'marker_pos',
  'marker_mask',
  'qtype',
];

/// The decision graph's outputs.
const _layaOutputs = ['logits', 'act_logits'];

/// ORT-backed implementation owning one session on a dedicated isolate.
///
/// Calls to [run] never block the caller's isolate; concurrent calls are
/// queued and run one at a time. [close] must be called: until then the
/// worker isolate keeps the process alive.
class OrtBackend implements InferenceBackend {
  OrtBackend._(this._worker);

  final _Worker _worker;

  /// Opens the Laya decision graph at [modelPath] on a new worker isolate.
  ///
  /// [intraOpThreads] sets ORT's intra-op thread count (its default when
  /// null). [ortLibraryPath] overrides `LAYA_ORT_LIB` and the bundled
  /// library. Throws [OrtException] when the library cannot be loaded, the
  /// model cannot be opened, or the graph's inputs are not exactly
  /// `input_ids`, `attention_mask`, `marker_pos`, `marker_mask` and `qtype`
  /// (or it lacks the `logits` and `act_logits` outputs); the message then
  /// names the expected and actual names.
  static Future<OrtBackend> open(
    String modelPath, {
    int? intraOpThreads,
    String? ortLibraryPath,
  }) async => OrtBackend._(
    await _Worker.spawn(
      modelPath,
      intraOpThreads: intraOpThreads,
      ortLibraryPath: ortLibraryPath,
      requireInputs: _layaInputs,
      requireOutputs: _layaOutputs,
    ),
  );

  /// Opens any graph at [modelPath] on a worker isolate, with no signature
  /// check, returning its raw outputs. A test hook for the isolate machinery.
  @visibleForTesting
  static Future<OrtRawBackend> openRaw(
    String modelPath, {
    int? intraOpThreads,
    String? ortLibraryPath,
  }) async => OrtRawBackend._(
    await _Worker.spawn(
      modelPath,
      intraOpThreads: intraOpThreads,
      ortLibraryPath: ortLibraryPath,
    ),
  );

  /// Runs [batch] through the graph.
  ///
  /// Throws [OrtException] for anything ORT rejects or for outputs of an
  /// unexpected shape, and [StateError] (`backend closed`) after [close].
  @override
  Future<InferenceOutput> run(InferenceBatch batch) async {
    final n = batch.rows;
    final k = batch.markers;
    final out = await _worker.run({
      'input_ids': OrtInt64Input(batch.inputIds, [n, batch.seqLen]),
      'attention_mask': OrtInt64Input(batch.attentionMask, [n, batch.seqLen]),
      'marker_pos': OrtInt64Input(batch.markerPos, [n, k]),
      'marker_mask': OrtBoolInput(batch.markerMask, [n, k]),
      'qtype': OrtInt64Input(batch.qtype, [n]),
    });
    final logits = _output(out, 'logits', [n, k]);
    final act = _output(out, 'act_logits', [n, 2]);
    return InferenceOutput(logits, act, k);
  }

  static Float32List _output(
    Map<String, OrtTensorF32> out,
    String name,
    List<int> shape,
  ) {
    final t = out[name];
    if (t == null) {
      throw OrtException(ortFail, 'the graph returned no $name output');
    }
    if (t.shape.join(',') != shape.join(',')) {
      throw OrtException(
        ortFail,
        'output $name has shape ${t.shape}, expected $shape',
      );
    }
    return t.data;
  }

  /// Releases the session and shuts the worker isolate down. Idempotent.
  ///
  /// Runs already submitted complete first (the worker handles requests in
  /// order, and the shutdown request queues behind them); the returned future
  /// completes once the isolate has exited. [run] throws afterwards.
  @override
  Future<void> close() => _worker.close();
}

/// A worker-isolate session that returns raw outputs by name; see
/// [OrtBackend.openRaw].
@visibleForTesting
class OrtRawBackend {
  OrtRawBackend._(this._worker);

  final _Worker _worker;

  /// The graph's input names, in graph order.
  List<String> get inputNames => _worker.inputNames;

  /// The graph's output names, in graph order.
  List<String> get outputNames => _worker.outputNames;

  /// Runs the graph with named [inputs] on the worker isolate.
  ///
  /// Throws [OrtException] for anything the session rejects and
  /// [StateError] (`backend closed`) after [close].
  Future<Map<String, OrtTensorF32>> run(Map<String, OrtInput> inputs) =>
      _worker.run(inputs);

  /// Shuts the worker down, as [OrtBackend.close] does.
  Future<void> close() => _worker.close();

  /// Completes when the worker isolate has exited.
  Future<void> get workerExited => _worker.exited;

  /// Kills the worker isolate without asking it to close, as a crash would.
  @visibleForTesting
  void killWorker() => _worker.kill();
}

// ---------------------------------------------------------------------------
// The isolate machinery, shared by OrtBackend and OrtRawBackend.

/// The main-isolate side of a worker.
final class _Worker {
  _Worker._();

  final _port = ReceivePort('laya-ort');
  final _ready = Completer<void>();
  final _exited = Completer<void>();
  final _pending = <int, Completer<Map<String, OrtTensorF32>>>{};
  late final SendPort _commands;
  late final List<String> inputNames;
  late final List<String> outputNames;
  var _nextId = 0;
  Future<void>? _closing;
  late final Isolate _isolate;

  /// Why the worker died, when it exited without being asked to.
  OrtException? _death;

  /// An uncaught error the worker reported before exiting.
  String? _crash;

  Future<void> get exited => _exited.future;

  static Future<_Worker> spawn(
    String modelPath, {
    int? intraOpThreads,
    String? ortLibraryPath,
    List<String>? requireInputs,
    List<String>? requireOutputs,
  }) async {
    final w = _Worker._();
    w._port.listen(w._onMessage);
    try {
      w._isolate = await Isolate.spawn(
        _workerMain,
        _Start(
          w._port.sendPort,
          modelPath,
          intraOpThreads,
          ortLibraryPath,
          requireInputs,
          requireOutputs,
        ),
        onExit: w._port.sendPort,
        onError: w._port.sendPort,
        debugName: 'laya-ort',
      );
    } catch (e) {
      w._port.close();
      throw OrtException(ortFail, 'could not start the inference worker: $e');
    }
    await w._ready.future;
    return w;
  }

  void _onMessage(Object? message) {
    switch (message) {
      case _Ready():
        _commands = message.commands;
        inputNames = message.inputNames;
        outputNames = message.outputNames;
        _ready.complete();
      case _Failed(:final code, :final message):
        // The worker returns after this, so its exit message (null) follows.
        _ready.completeError(OrtException(code, message));
      case _Result(:final id, :final outputs):
        _pending.remove(id)?.complete({
          for (final MapEntry(:key, :value) in outputs.entries)
            key: OrtTensorF32(
              value.data.materialize().asFloat32List(),
              value.shape,
            ),
        });
      case _RunFailed(:final id, :final code, :final message):
        _pending.remove(id)?.completeError(OrtException(code, message));
      case [final Object? error, _]:
        // onError: [error, stack trace], as strings. The isolate exits next.
        _crash = '$error';
      case null:
        // onExit: the worker isolate has terminated.
        final asked = _closing != null;
        final death = OrtException(
          ortFail,
          'the inference worker exited unexpectedly'
          '${_crash == null ? '' : ': $_crash'}',
        );
        if (!asked) _death = death;
        if (!_ready.isCompleted) _ready.completeError(death);
        for (final c in _pending.values) {
          c.completeError(asked ? StateError('backend closed') : death);
        }
        _pending.clear();
        _port.close();
        _exited.complete();
    }
  }

  Future<Map<String, OrtTensorF32>> run(Map<String, OrtInput> inputs) async {
    if (_closing != null) throw StateError('backend closed');
    final death = _death;
    if (death != null) throw death;
    final id = _nextId++;
    final result = Completer<Map<String, OrtTensorF32>>();
    _commands.send(
      _Run(id, {
        for (final MapEntry(:key, :value) in inputs.entries)
          key: switch (value) {
            OrtInt64Input(:final data, :final shape) => _InTensor(
              false,
              TransferableTypedData.fromList([data]),
              shape,
            ),
            OrtBoolInput(:final data, :final shape) => _InTensor(
              true,
              TransferableTypedData.fromList([data]),
              shape,
            ),
          },
      }),
    );
    // Only once the send succeeded: a run whose message could not be sent
    // throws here and leaves no completer for the exit handler to fail.
    _pending[id] = result;
    return result.future;
  }

  /// Terminates the isolate at once; its exit fails every pending run.
  void kill() => _isolate.kill(priority: Isolate.immediate);

  Future<void> close() => _closing ??= () async {
    if (!_exited.isCompleted) _commands.send(const _Close());
    await _exited.future;
  }();
}

/// The worker isolate's entry point.
void _workerMain(_Start start) {
  final OrtSession session;
  try {
    session = loadOrtRuntime(
      explicitPath: start.ortLibraryPath,
    ).openSession(start.modelPath, intraOpThreads: start.intraOpThreads);
  } on OrtException catch (e) {
    start.reply.send(_Failed(e.code, e.message));
    return;
  } catch (e) {
    start.reply.send(_Failed(ortFail, '$e'));
    return;
  }

  final problem = _signatureProblem(
    start.modelPath,
    session,
    start.requireInputs,
    start.requireOutputs,
  );
  if (problem != null) {
    session.close();
    start.reply.send(_Failed(ortInvalidArgument, problem));
    return;
  }

  final commands = ReceivePort('laya-ort-worker');
  start.reply.send(
    _Ready(commands.sendPort, session.inputNames, session.outputNames),
  );
  commands.listen((message) {
    switch (message) {
      case _Run(:final id, :final inputs):
        start.reply.send(_runOne(session, id, inputs));
      case _Close():
        session.close();
        commands.close();
    }
  });
}

/// Runs one request; returns the reply to send.
Object _runOne(OrtSession session, int id, Map<String, _InTensor> inputs) {
  try {
    final out = session.run({
      for (final MapEntry(:key, :value) in inputs.entries)
        key: value.isBool
            ? OrtBoolInput(value.data.materialize().asUint8List(), value.shape)
            : OrtInt64Input(
                value.data.materialize().asInt64List(),
                value.shape,
              ),
    });
    return _Result(id, {
      for (final MapEntry(:key, :value) in out.entries)
        key: _OutTensor(
          TransferableTypedData.fromList([value.data]),
          value.shape,
        ),
    });
  } on OrtException catch (e) {
    return _RunFailed(id, e.code, e.message);
  } on ArgumentError catch (e) {
    return _RunFailed(id, ortInvalidArgument, '$e');
  } catch (e) {
    return _RunFailed(id, ortFail, '$e');
  }
}

/// Why [session] is not the graph the caller asked for, or null when it is
/// (or no names were required).
String? _signatureProblem(
  String modelPath,
  OrtSession session,
  List<String>? inputs,
  List<String>? outputs,
) {
  final problems = <String>[];
  if (inputs != null &&
      (session.inputNames.length != inputs.length ||
          !session.inputNames.toSet().containsAll(inputs))) {
    problems.add(
      'its inputs are ${session.inputNames}, expected exactly $inputs',
    );
  }
  if (outputs != null && !session.outputNames.toSet().containsAll(outputs)) {
    problems.add(
      'its outputs are ${session.outputNames}, expected to include $outputs',
    );
  }
  if (problems.isEmpty) return null;
  return '$modelPath is not the Laya decision graph: ${problems.join('; ')}. '
      'Point at the single-file decision-model ONNX export, not a split '
      'encoder.onnx or another model.';
}

// Messages. Both isolates are in one isolate group, so plain objects cross.

final class _Start {
  const _Start(
    this.reply,
    this.modelPath,
    this.intraOpThreads,
    this.ortLibraryPath,
    this.requireInputs,
    this.requireOutputs,
  );
  final SendPort reply;
  final String modelPath;
  final int? intraOpThreads;
  final String? ortLibraryPath;
  final List<String>? requireInputs;
  final List<String>? requireOutputs;
}

final class _Ready {
  const _Ready(this.commands, this.inputNames, this.outputNames);
  final SendPort commands;
  final List<String> inputNames;
  final List<String> outputNames;
}

final class _Failed {
  const _Failed(this.code, this.message);
  final int code;
  final String message;
}

final class _InTensor {
  const _InTensor(this.isBool, this.data, this.shape);
  final bool isBool;
  final TransferableTypedData data;
  final List<int> shape;
}

final class _OutTensor {
  const _OutTensor(this.data, this.shape);
  final TransferableTypedData data;
  final List<int> shape;
}

final class _Run {
  const _Run(this.id, this.inputs);
  final int id;
  final Map<String, _InTensor> inputs;
}

final class _Result {
  const _Result(this.id, this.outputs);
  final int id;
  final Map<String, _OutTensor> outputs;
}

final class _RunFailed {
  const _RunFailed(this.id, this.code, this.message);
  final int id;
  final int code;
  final String message;
}

final class _Close {
  const _Close();
}
