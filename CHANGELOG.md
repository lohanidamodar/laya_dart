## 0.1.0-dev.1

- Initial development version.
- `POST /v1/systemone` and `GET /health`, with laya-serve's limits, statuses,
  error texts, bearer auth, admission control (503) and timing headers.
- The `english`, `multilingual` and `typed-decisions` checkpoints, with
  language and script routing, `task` selection and optional auto-task
  detection.
- Pure-Dart tokenizers (both BPE variants), sequence building, answer decoding,
  `min_confidence` abstention (single value or per-bucket) and the strict Jev
  response (`LAYA_JEV_STRICT`).
- ONNX Runtime 1.30.0 through `dart:ffi`, bundled by a build hook for Windows,
  Linux and Apple-silicon macOS. User-defines `bundle_ort` and `ort_lib`, and
  `LAYA_ORT_LIB` at run time, supply another library.
- `fetch` downloads pinned, sha256-verified checkpoints (fp32 or fp16).
- Library API: `LayaEngine`, `createHandler`, `ServerConfig`, `ModelStore`.
- Not in this version: `POST /v1/systemone/batch` and `GET /v1/models`.
