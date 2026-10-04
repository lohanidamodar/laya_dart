/// Static facts about the three shipped Laya checkpoints.
library;

import 'dart:math' as math;

import 'question.dart';

/// Static facts about one checkpoint.
class CheckpointSpec {
  /// Creates a spec.
  const CheckpointSpec({
    required this.name,
    required this.repo,
    required this.onnxRepo,
    required this.onnxRevision,
    required this.maxLen,
    required this.headMaxLen,
  });

  /// `english`, `multilingual` or `typed-decisions`.
  final String name;

  /// The id laya's router reports in `routing.repo`: the bundle repo, with the
  /// checkpoint's subfolder appended when it has one.
  final String repo;

  /// The `onnx-community/...` repository holding the exported graph.
  final String onnxRepo;

  /// The pinned revision of [onnxRepo].
  final String onnxRevision;

  /// The default sequence length.
  final int maxLen;

  /// The default token budget for the question half of a sequence.
  final int headMaxLen;

  /// The three shipped checkpoints, keyed by name.
  static const Map<String, CheckpointSpec> all = {
    'english': CheckpointSpec(
      name: 'english',
      repo: 'convaiinnovations/laya',
      onnxRepo: 'onnx-community/laya-ONNX',
      onnxRevision: '42e2a6e3b3708c8ff5f61f4aa75c314b1449199b',
      maxLen: 512,
      headMaxLen: 192,
    ),
    'multilingual': CheckpointSpec(
      name: 'multilingual',
      repo: 'convaiinnovations/laya/multilingual',
      onnxRepo: 'onnx-community/laya-multilingual-ONNX',
      onnxRevision: '46b77bbf5642fec5f14e540570228a8cbe8ab81f',
      maxLen: 1024,
      headMaxLen: 256,
    ),
    'typed-decisions': CheckpointSpec(
      name: 'typed-decisions',
      repo: 'convaiinnovations/laya/typed-decisions',
      onnxRepo: 'onnx-community/laya-typed-decisions-ONNX',
      onnxRevision: '79c441b22da79eb3727bb01e8452e29a32cfa548',
      maxLen: 1024,
      headMaxLen: 256,
    ),
  };
}

/// Calibration and windows read from a checkpoint's config: the `laya` section
/// of the onnx-community `config.json`.
///
/// For all three shipped checkpoints that section equals upstream
/// `rl_agent_config.json`, which is what laya's `ONNXAgent` reads
/// (`test/model/decode_test.dart` pins this).
class CheckpointConfig {
  /// Creates a config; every temperature is clamped to [temperatureClamp]
  /// (laya `clamp_temperature`), and a NaN or infinite one becomes 1.0.
  CheckpointConfig({
    required this.maxLen,
    required this.headMaxLen,
    required List<double> temperature,
    required Map<String, double> temperatureByOptions,
    (double, double) temperatureClamp = defaultTemperatureClamp,
  }) : temperature = List.unmodifiable([
         for (final t in temperature) _clamp(t, temperatureClamp),
       ]),
       temperatureByOptions = Map.unmodifiable({
         for (final e in temperatureByOptions.entries)
           e.key: _clamp(e.value, temperatureClamp),
       });

  /// Reads the `laya` section of an onnx-community `config.json`: `max_len`,
  /// `head_max_len`, `temperature`, `temperature_by_options` and
  /// `temperature_clamp` (default [defaultTemperatureClamp]).
  ///
  /// As laya's `clamp_temperature` does, a temperature that is not a number
  /// (a bool included) becomes 1.0. Throws [FormatException] when the section
  /// or a window is missing.
  factory CheckpointConfig.fromOnnxConfig(Map<String, Object?> configJson) {
    final laya = configJson['laya'];
    if (laya is! Map) {
      throw const FormatException('config.json has no "laya" section');
    }
    int window(String key) => switch (laya[key]) {
      final int v => v,
      _ => throw FormatException('config.json laya.$key is not an integer'),
    };
    final clamp = laya['temperature_clamp'];
    final temperature = laya['temperature'] ?? const [1.0, 1.0, 1.0];
    final byOptions =
        laya['temperature_by_options'] ?? const <String, Object?>{};
    if (temperature is! List || byOptions is! Map) {
      throw const FormatException(
        'config.json laya.temperature must be a list and '
        'laya.temperature_by_options a map',
      );
    }
    return CheckpointConfig(
      maxLen: window('max_len'),
      headMaxLen: window('head_max_len'),
      temperature: [for (final t in temperature) _number(t)],
      temperatureByOptions: {
        for (final e in byOptions.entries) '${e.key}': _number(e.value),
      },
      temperatureClamp:
          clamp is List && clamp.length == 2 && clamp.every((v) => v is num)
          ? ((clamp[0] as num).toDouble(), (clamp[1] as num).toDouble())
          : defaultTemperatureClamp,
    );
  }

  /// laya's `TEMP_MIN` and `TEMP_MAX` (`common.py:706-707`).
  static const defaultTemperatureClamp = (0.5, 5.0);

  /// The default sequence length.
  final int maxLen;

  /// The default token budget for the question half of a sequence.
  final int headMaxLen;

  /// Per-type temperatures, indexed by [QuestionType.id]; clamped.
  final List<double> temperature;

  /// Temperatures by option-count bucket (`choice:3-5` and so on); clamped.
  final Map<String, double> temperatureByOptions;

  /// Temperature for a question of [type] with [k] options: the
  /// [temperatureBucket], then the type's [temperature], then 1.0
  /// (laya `onnx_agent.py:651`).
  double temperatureFor(QuestionType type, int k) =>
      temperatureByOptions[temperatureBucket(type, k)] ??
      (type.id < temperature.length ? temperature[type.id] : 1.0);

  // A non-number becomes NaN here, which _clamp turns into 1.0.
  static double _number(Object? v) => v is num ? v.toDouble() : double.nan;

  static double _clamp(double t, (double, double) range) =>
      t.isFinite ? math.min(range.$2, math.max(range.$1, t)) : 1.0;
}

/// laya `temp_bucket` (`common.py:697`): `<type>:<2|3-5|6-10|11+>`.
String temperatureBucket(QuestionType type, int k) {
  final size = k <= 2
      ? '2'
      : k <= 5
      ? '3-5'
      : k <= 10
      ? '6-10'
      : '11+';
  return '${type.name}:$size';
}
