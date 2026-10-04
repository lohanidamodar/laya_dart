# laya_server

Run the open [Laya](https://huggingface.co/convaiinnovations/laya) decision
model locally and serve it over an HTTP API compatible with `laya-serve` and the
TypeSafe System One ("Jev") contract. Pure Dart: no Python, no Docker. The only
native piece is ONNX Runtime, which a build hook fetches for you.

Laya reads a piece of text (a "state") and answers several questions about it at
once (pick a label, rate on a scale, answer yes or no), with calibrated
probabilities. Three checkpoints exist: `english`, `multilingual` and
`typed-decisions`. Requests are routed to the right one by script and language.

> Independent port, not affiliated with Convai Innovations or TypeSafe.

## Quick start

```sh
dart pub global activate laya_server

laya_fetch english     # downloads the pinned checkpoint, about 1.6 GiB
laya_server            # serves on http://0.0.0.0:8000
```

Without a global install, use `dart run laya_server:laya_fetch english` and
`dart run laya_server` inside a project that depends on the package.

`laya_fetch` verifies every file against a pinned sha256. It takes `--dir` (default
`models`), `--variant fp32|fp16` and `--no-verify`, and the names `english`,
`multilingual`, `typed-decisions` or `all`. It exits 0 on success, 1 on a
failure and 64 on a usage error. Set `HF_TOKEN` if you need an authenticated
download.

Ask it a question:

```sh
curl -s http://localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "I was charged twice this month and I am furious.",
    "questions": {
      "queue": {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "payments", "tech": "bugs", "other": "else"}},
      "urgency": {"type": "score", "instructions": "How upset is the customer?",
                  "criteria": ["calm", "firm", "angry", "furious"]}
    }
  }'
```

## Using it as a library

```dart
final engine = LayaEngine(loader: LayaEngine.fileLoader('models'));
final result = await engine.predict(
  'I was charged twice this month and I am furious.',
  {
    'queue': {
      'type': 'choice',
      'instructions': 'Which team?',
      'criteria': {'billing': 'payments', 'tech': 'bugs', 'other': 'else'},
    },
  },
  const PredictOptions(),
);
await engine.close();
```

See `example/main.dart`. `createHandler(engine, config)` returns a `shelf`
handler to mount in your own server, at the root or under a path prefix.

## Configuration

`laya_server` reads the same `LAYA_*` variables as `laya-serve`, plus three of
its own.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LAYA_HOST` | `0.0.0.0` | Bind address |
| `LAYA_PORT` | `8000` | Bind port, 1-65535 (invalid is a startup error) |
| `LAYA_ROOT_PATH` | empty | Public URL prefix behind a reverse proxy |
| `LAYA_MODEL_DIR` | `./models` | Where checkpoints are installed (own) |
| `LAYA_ONNX_VARIANT` | `fp32` | `fp32` or `fp16` weights (own) |
| `LAYA_ORT_LIB` | unset | Path to an ONNX Runtime library to load instead of the bundled one (own) |
| `LAYA_MODELS` | every installed one | Comma-separated checkpoints to preload |
| `LAYA_PRELOAD` | `1` | Load `LAYA_MODELS` at startup; `0` loads on first use |
| `LAYA_THREADS` | ONNX Runtime default | Intra-op threads |
| `LAYA_AUTO_TASK` | `0` | Let a typed-decisions question schema select that checkpoint |
| `LAYA_DEFAULT_MODEL` | `english` | Checkpoint for text whose language nothing identifies |
| `LAYA_MAX_LOADED` | `2` | Checkpoints resident at once (least recently used out first) |
| `LAYA_API_KEY` | unset | Require `Authorization: Bearer <key>` on `POST /v1/systemone` |
| `LAYA_MAX_CONCURRENT` | `16` | Requests admitted at once; more get 503 |
| `LAYA_MAX_TOKEN_BUDGET` | `8192` | Cap on a request's `max_len` and `head_max_len` |
| `LAYA_JEV_STRICT` | `0` | Serve the strict Jev response shape |
| `LAYA_LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug` or `trace` |

Booleans are true for `1`, `true`, `yes` and `on` (any case). Numeric values that
do not parse fall back to their default, except `LAYA_PORT`,
`LAYA_DEFAULT_MODEL` and `LAYA_ONNX_VARIANT`, which stop the server with one
line on stderr and exit code 2. Startup also exits 2 when a checkpoint named in
`LAYA_MODELS` is not installed, or, with `LAYA_MODELS` unset, when none is.

Startup preloads every checkpoint in `LAYA_MODELS` (with it unset, every
checkpoint installed under `LAYA_MODEL_DIR`) and raises the resident cap to
hold them, so preloading never evicts. Set `LAYA_MODELS=english` to keep memory
down.

## API

`GET /health` and `POST /v1/systemone`. The full wire protocol (request fields,
limits, response shapes, error table, strict mode) is in [doc/api.md](doc/api.md).

## Memory and disk

Sizes are the pinned files of each checkpoint (tokenizer, config and ONNX graph
with its weights). Expect about the fp32 figure in resident memory per loaded
checkpoint, plus working memory for inference.

| Checkpoint | fp32 | fp16 |
| --- | --- | --- |
| `english` | 1.69 GB (1615 MiB) | 0.85 GB (811 MiB) |
| `multilingual` | 1.33 GB (1264 MiB) | 0.68 GB (650 MiB) |
| `typed-decisions` | 1.69 GB (1615 MiB) | 0.85 GB (811 MiB) |

All three at fp32 need about 4.7 GB of disk. fp16 halves that. It is opt-in
(`laya_fetch --variant fp16` and `LAYA_ONNX_VARIANT=fp16`), as fp32 is the
reference precision.

## ONNX Runtime

A build hook downloads ONNX Runtime 1.30.0 once, checks its sha256 and bundles
the shared library as a code asset. Prebuilt archives exist for Windows (x64,
arm64), Linux (x64, arm64) and macOS (arm64). **There is none for Intel macOS**:
there, supply your own library with `ort_lib` or `LAYA_ORT_LIB`.

Configure the hook in the pubspec of the application that depends on
`laya_server`:

```yaml
hooks:
  user_defines:
    laya_server:
      bundle_ort: false          # bundle nothing; set LAYA_ORT_LIB at run time
      # ort_lib: /path/to/libonnxruntime.so   # or bundle this file instead
```

`LAYA_ORT_LIB` is read at run time only. Build hooks run with a filtered
environment, so it cannot steer the build.

## Differences from laya-serve

Same routes, limits, statuses, `{"detail": ...}` texts and byte-for-byte
response bodies for everything it serves, except as listed below. The answer
decoding is bit-exact against numpy's AVX2/FMA float32 kernels (x86-64), which
is what laya-serve runs on such machines; elsewhere numpy's kernels differ and
a rounded value can differ in its last digit.

- **No batch and no `/v1/models` in v1.** `POST /v1/systemone/batch` and
  `GET /v1/models` are 404.
- **`device` is always `cpu`.** There is no `LAYA_DEVICE`; ONNX Runtime runs on
  the CPU.
- **fp16 is opt-in** through `LAYA_ONNX_VARIANT`; fp32 is the default.
- **`LAYA_MODELS` unset preloads the installed checkpoints only.** laya-serve
  preloads all three; here a checkpoint that is not installed is skipped, so
  `laya_fetch english` then `laya_server` starts. A checkpoint `LAYA_MODELS`
  names must be installed.
- **Integers beyond int64 decode as doubles.** A `max_len` such as `1e20` written
  as an integer literal is reported as "must be an integer", and such integers
  inside a state serialize differently.
- **The `NaN`, `Infinity` and `-Infinity` literals are a 400** (invalid JSON).
  Python accepts them: laya-serve answers `min_confidence: NaN` with 422 and
  takes `Infinity` inside a state.
- **A NaN or infinity in the result is a 500** `inference failed`, logged.
  laya-serve builds the response inside its error handler and answers 422
  `Out of range float values are not JSON compliant: nan`.
- **`LAYA_MAX_TOKEN_BUDGET` and `LAYA_JEV_STRICT` are read once at startup.**
  laya-serve re-reads them on every request.
- **No body-read timeout**, the same as laya-serve. A slow client holds an
  admission slot; use a reverse proxy with timeouts in front of untrusted
  traffic.
- **JSON nesting is limited to 2993 levels**; deeper bodies are a 400. That is
  the depth CPython 3.12 manages. Under uvicorn the exact ceiling depends on the
  C stack and may differ by a few levels.
- **`LAYA_ROOT_PATH` strips the prefix.** A request carrying the prefix is served
  with it removed, and a request without it is served too, so both
  `/laya/v1/systemone` and `/v1/systemone` work. Redirects keep the prefix.
- **Signals.** Windows cannot watch SIGTERM, so only SIGINT stops the server
  there.

## Client compatibility

The [`jev_dart`](https://pub.dev/packages/jev_dart) 0.1.1 client decodes both the
full payload and the `LAYA_JEV_STRICT` payload. Later versions (0.1.2 to 0.1.5)
could not be tested: their analyzer dependency conflicts with `test`. `jev_dart`
reads `choice` as a `String`.

## Development

```sh
dart run tool/check.dart        # format, analyze --fatal-infos, test
```

The tests run without model weights. Reference fixtures under `test/fixtures/`
come from the Python `laya` package and are regenerated with

```sh
dart run tool/gen_fixtures.dart
```

which needs a Python environment at `.tooling/venv` (laya 0.3.24, tokenizers,
transformers, onnx, onnxruntime, torch). It downloads only tokenizer and config
files, never weights. `dart run tool/pin_models.dart` and
`dart run tool/pin_ort.dart` refresh the pinned checkpoint and ONNX Runtime
tables.

## Licence and credits

Apache-2.0; see `LICENSE`. The sequence format, answer decoding, router and
language analysis are Dart ports of [laya](https://github.com/NandhaKishorM/laya)
by Convai Innovations (Apache-2.0), modified for Dart; see `NOTICE`. The model
weights are published by Convai Innovations under Apache-2.0, and the ONNX
conversions this package downloads by onnx-community under Apache-2.0. No
weights ship in this package.
