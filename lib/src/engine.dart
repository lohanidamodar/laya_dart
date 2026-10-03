/// Routing, lazy loading and LRU residency over the Laya checkpoints: laya's
/// `Router.predict` as `laya-serve` drives it.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:logging/logging.dart';
import 'package:path/path.dart' as p;

import 'inference/ort_backend.dart';
import 'model/agent.dart';
import 'model/checkpoint.dart';
import 'model/confidence.dart';
import 'model/question.dart';
import 'routing/router.dart';
import 'server/errors.dart';
import 'tokenizer/tokenizer.dart';

/// Options a request may set: the controls laya's `Router.predict` takes from
/// a `laya-serve` request body.
class PredictOptions {
  /// Creates options; every field is optional.
  const PredictOptions({
    this.model,
    this.task,
    this.lang,
    this.langGuess,
    this.maxLen,
    this.headMaxLen,
    this.minConfidence,
  });

  /// The client's `model` field. A checkpoint name, alias or public Hugging
  /// Face id picks that checkpoint; anything else (`jev-latest`, the bundle
  /// id `convaiinnovations/laya`, ...) means auto-route.
  final String? model;

  /// A checkpoint to use, by name or alias. An unknown one is a 422.
  final String? task;

  /// The state's language code, when the caller knows it.
  final String? lang;

  /// The caller's guess at the state's language code.
  final String? langGuess;

  /// Overrides the checkpoint's sequence length.
  final int? maxLen;

  /// Overrides the checkpoint's question-head token budget.
  final int? headMaxLen;

  /// The abstention threshold (laya `min_confidence`): one for every answer,
  /// or one per option-count bucket. See [ConfidenceThreshold.check].
  final ConfidenceThreshold? minConfidence;
}

final _log = Logger('laya_server.engine');

/// The public Hugging Face ids a client may send as `model`, from laya's
/// `STANDALONE_MODELS` without the bundle repo (`laya/serve.py`
/// `_published_model_ids`): `convaiinnovations/laya` means auto-route.
const _publishedModelIds = {
  'convaiinnovations/laya-multilingual': 'multilingual',
  'convaiinnovations/laya-typed-decisions': 'typed-decisions',
};

/// laya-serve `_resolve_model`: the checkpoint a client's `model` names, or
/// null to auto-route. It never throws.
String? resolveClientModel(String? model) {
  if (model == null || model.isEmpty) return null;
  // Python `str.strip().lower()`; the ids it is compared with are ASCII.
  final published = _publishedModelIds[model.trim().toLowerCase()];
  if (published != null) return published;
  try {
    return normaliseCheckpointName(model);
  } on ArgumentError {
    return null;
  }
}

/// A checkpoint's place in the residency list.
class _Slot {
  _Slot(this.agent);

  /// The load, shared by every request for this checkpoint.
  final Future<LayaAgent> agent;

  /// Requests holding this slot, including ones still waiting for the load.
  int inUse = 0;

  /// Whether the LRU dropped it; its backend closes once [inUse] is 0.
  bool evicted = false;

  /// Whether [agent] completed with a value.
  bool ready = false;

  /// Completes once an evicted slot's backend is closed.
  final closed = Completer<void>();
}

/// Routes requests to checkpoints, loading them on demand and keeping at most
/// `maxLoaded` resident (least recently used out first).
///
/// Loads are memoised per checkpoint, so concurrent requests share one load.
/// An evicted checkpoint's backend is closed once the requests already using
/// it finish; none is closed under a running request.
class LayaEngine {
  /// Creates an engine that loads checkpoints with [loader] (see
  /// [fileLoader]).
  ///
  /// [maxLoaded] (at least 1) bounds the resident checkpoints; the least
  /// recently used is evicted only once another load succeeds. Loads run
  /// concurrently, so peak memory can briefly exceed [maxLoaded] checkpoints.
  /// [defaultModel] answers a state whose language nothing identifies, and
  /// [autoTaskDetection] lets a typed-decisions question schema select that
  /// checkpoint. Throws [ArgumentError] for a [maxLoaded] below 1 or an
  /// unknown [defaultModel].
  LayaEngine({
    required Future<LayaAgent> Function(String checkpoint) loader,
    int maxLoaded = 2,
    String defaultModel = 'english',
    bool autoTaskDetection = false,
  }) : _loader = loader,
       _maxLoaded = maxLoaded,
       _router = CheckpointRouter(
         defaultModel: defaultModel,
         autoTaskDetection: autoTaskDetection,
       ) {
    RangeError.checkValueInInterval(maxLoaded, 1, 1 << 30, 'maxLoaded');
  }

  final Future<LayaAgent> Function(String) _loader;
  final int _maxLoaded;
  final CheckpointRouter _router;

  /// Resident and loading checkpoints, least recently used first.
  // A map literal is a LinkedHashMap: insertion order is the LRU order.
  final _slots = <String, _Slot>{};

  /// Evicted checkpoints whose backend is not closed yet, awaited by [close].
  final _draining = <_Slot>{};

  bool _closed = false;

  /// A loader for checkpoints installed under [modelDir], laid out as
  /// `<modelDir>/<checkpoint>/` with `tokenizer.json`, `config.json` (the
  /// onnx-community config) and `onnx/model.onnx` plus its
  /// `onnx/model.onnx_data` sidecar (`model_fp16.onnx` and its sidecar for
  /// [variant] `fp16`).
  ///
  /// [threads] sets ONNX Runtime's intra-op threads and [ortLibraryPath]
  /// overrides the runtime library (see [OrtBackend.open]). The returned
  /// loader throws [StateError] naming the fetch command when a file is
  /// missing, and [ArgumentError] for an unknown checkpoint. Throws
  /// [ArgumentError] for a [variant] other than `fp32` or `fp16`.
  static Future<LayaAgent> Function(String) fileLoader(
    String modelDir, {
    String variant = 'fp32',
    int? threads,
    String? ortLibraryPath,
  }) {
    final model = switch (variant) {
      'fp32' => 'model.onnx',
      'fp16' => 'model_fp16.onnx',
      _ => throw ArgumentError.value(
        variant,
        'variant',
        'must be fp32 or fp16',
      ),
    };
    return (name) async {
      final spec =
          CheckpointSpec.all[name] ??
          (throw ArgumentError.value(
            name,
            'checkpoint',
            'is not a checkpoint',
          ));
      final dir = p.join(modelDir, name);
      final tokenizer = p.join(dir, 'tokenizer.json');
      final config = p.join(dir, 'config.json');
      final graph = p.join(dir, 'onnx', model);
      for (final path in [tokenizer, config, graph, '${graph}_data']) {
        if (!File(path).existsSync()) {
          throw StateError(
            'checkpoint $name is not installed in $modelDir; '
            'run: dart run laya_server:fetch $name',
          );
        }
      }
      final configJson = jsonDecode(await File(config).readAsString());
      if (configJson is! Map<String, Object?>) {
        throw FormatException('$config: expected a JSON object');
      }
      final cfg = CheckpointConfig.fromOnnxConfig(configJson);
      final tok = await LayaTokenizer.load(tokenizer);
      final backend = await OrtBackend.open(
        graph,
        intraOpThreads: threads,
        ortLibraryPath: ortLibraryPath,
      );
      return LayaAgent(
        spec: spec,
        tokenizer: tok,
        config: cfg,
        backend: backend,
      );
    };
  }

  /// Loads [names] (checkpoint names or aliases) now, in order. Throws
  /// [ArgumentError] for an unknown name.
  Future<void> preload(Iterable<String> names) async {
    for (final name in names) {
      if (_closed) throw StateError('engine closed');
      final slot = _acquire(normaliseCheckpointName(name));
      try {
        await slot.agent;
      } finally {
        _release(slot);
      }
    }
  }

  /// The loaded checkpoints, least recently used first (laya
  /// `Router.loaded`). A checkpoint still loading is not listed.
  List<String> get loaded => [
    for (final e in _slots.entries)
      if (e.value.ready) e.key,
  ];

  /// Answers [questions] about [state] as laya-serve's `Router.predict` does,
  /// returning `{model, answers, usage, routing}`.
  ///
  /// [state] is decoded JSON: a string, map or list. [questions] maps each
  /// question id to its raw definition. Throws [ApiException] 422 for an
  /// invalid question, an unknown [PredictOptions.task] or options that do
  /// not fit the token budget, and [StateError] after [close]. A loader
  /// failure propagates as it was thrown.
  Future<Map<String, Object?>> predict(
    Object state,
    Map<String, Object?> questions,
    PredictOptions options,
  ) async {
    if (_closed) throw StateError('engine closed');
    final RouteDecision decision;
    try {
      decision = _router.route(
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
    // laya validates after loading the checkpoint; validating first means a
    // bad request never costs a load. Only the error precedence differs, and
    // only when a load would also have failed.
    final parsed = {
      for (final e in questions.entries) e.key: Question.parse(e.key, e.value),
    };
    final slot = _acquire(decision.model);
    try {
      final agent = await slot.agent;
      final result = await agent.predict(
        state,
        parsed,
        maxLen: options.maxLen,
        headMaxLen: options.headMaxLen,
        minConfidence: options.minConfidence,
      );
      result['routing'] = decision.toJson();
      return result;
    } finally {
      _release(slot);
    }
  }

  /// Closes every loaded backend, waiting for running requests first. Later
  /// calls to [predict] throw [StateError]. Idempotent.
  Future<void> close() async {
    _closed = true;
    for (final slot in _slots.values.toList()) {
      _evict(slot);
    }
    _slots.clear();
    await Future.wait([for (final s in _draining.toList()) s.closed.future]);
  }

  /// Takes a hold on [name]'s slot, loading it if absent, and marks it most
  /// recently used. Every call must be paired with [_release].
  _Slot _acquire(String name) {
    var slot = _slots.remove(name);
    if (slot == null) {
      final created = slot = _Slot(_loader(name));
      unawaited(
        created.agent.then<void>(
          (_) {
            created.ready = true;
            // As laya (router.py:640-649): evict only once a load succeeds,
            // so a failing load never costs a healthy checkpoint.
            if (identical(_slots[name], created)) _trim(created);
          },
          onError: (Object _) {
            // A failed load is forgotten so the next request retries it.
            if (identical(_slots[name], created)) _slots.remove(name);
          },
        ),
      );
    }
    _slots[name] = slot;
    slot.inUse++;
    return slot;
  }

  /// Evicts the least recently used loaded checkpoints, never [loaded]
  /// itself, until at most `maxLoaded` are loaded. Checkpoints still loading
  /// do not count.
  void _trim(_Slot loaded) {
    var ready = _slots.values.where((s) => s.ready).length;
    for (final e in _slots.entries.toList()) {
      if (ready <= _maxLoaded) break;
      if (!e.value.ready || identical(e.value, loaded)) continue;
      _evict(_slots.remove(e.key)!);
      ready--;
    }
  }

  void _release(_Slot slot) {
    slot.inUse--;
    if (slot.evicted && slot.inUse == 0) _closeBackend(slot);
  }

  void _evict(_Slot slot) {
    if (slot.evicted) return;
    slot.evicted = true;
    _draining.add(slot);
    if (slot.inUse == 0) _closeBackend(slot);
  }

  void _closeBackend(_Slot slot) => unawaited(() async {
    try {
      final LayaAgent agent;
      try {
        agent = await slot.agent;
      } catch (_) {
        return; // A failed load has nothing to close; its request saw why.
      }
      try {
        await agent.backend.close();
      } catch (error, stackTrace) {
        // No request is left to tell, but a failed close can leak a session.
        _log.warning(
          'closing evicted checkpoint ${agent.spec.name} failed',
          error,
          stackTrace,
        );
      }
    } finally {
      _draining.remove(slot);
      slot.closed.complete();
    }
  }());
}
