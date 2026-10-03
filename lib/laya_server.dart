/// Run the Laya decision model locally and serve it over a Jev-compatible API.
///
/// [LayaEngine] routes each request to a checkpoint, loads it on demand and
/// answers with laya's `{model, answers, usage, routing}` result.
/// [LayaEngine.fileLoader] loads installed checkpoints through ONNX Runtime
/// ([OrtBackend]); [FakeBackend] stands in for tests. [createHandler] serves
/// an engine over laya-serve's HTTP API, configured by [ServerConfig].
library;

export 'src/engine.dart' show LayaEngine, PredictOptions;
export 'src/inference/backend.dart'
    show FakeBackend, InferenceBackend, InferenceOutput;
export 'src/inference/batch.dart' show InferenceBatch;
export 'src/inference/ort_backend.dart' show OrtBackend;
export 'src/model/agent.dart' show LayaAgent;
export 'src/model/checkpoint.dart' show CheckpointConfig, CheckpointSpec;
export 'src/model/confidence.dart'
    show BucketThresholds, ConfidenceThreshold, ScalarThreshold;
export 'src/model/question.dart'
    show ChoiceQuestion, NoulQuestion, Question, QuestionType, ScoreQuestion;
export 'src/server/config.dart' show ServerConfig;
export 'src/server/errors.dart' show ApiException;
export 'src/server/handler.dart' show createHandler;
export 'src/store/model_store.dart' show ModelStore, PinnedFile;
export 'src/tokenizer/tokenizer.dart' show LayaTokenizer;
