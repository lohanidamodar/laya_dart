/// Static facts about the three shipped Laya checkpoints.
library;

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
