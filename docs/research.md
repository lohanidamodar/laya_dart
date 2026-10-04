# laya_server — Phase 1 research

Status: research complete, awaiting a decision on the inference route before
implementation. Written 2026-10-03.

Sources were read directly, not summarised from blog posts, unless marked
otherwise:

- Upstream `laya` repository, `NandhaKishorM/laya` at commit
  `fa9a2a7070b1789912a49ae24603bbfb1a78b001` (2026-10-02), PyPI `laya` 0.3.24.
  `laya-serve` is `laya/serve.py` in that repository; its page is
  `docs/http-api.md`.
- Hugging Face `convaiinnovations/laya` at revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (2026-09-24): file tree,
  `rl_agent_config.json`, `encoder/config.json`, both `tokenizer.json` files.
- `nomideusz/laya-railway`, a second Jev-compatible deployment that adds
  `GET /v1/models`.
- TypeSafe's Jev docs (`docs.typesafe.ai/api`, `/models`, the SDK exception
  pages) and DigitalOcean's System One page. These were read through a fetch
  summariser rather than verbatim, so treat their exact wording as approximate.

Paths like `serve.py:870` below refer to the upstream `laya` repository.

---

## 1. The wire protocol

### 1.1 Jev (TypeSafe), the contract clients are written against

| | |
|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` |
| Auth | `Authorization: Bearer <key>` |
| Model list | `GET /v1/models` → models with `name`, `description`, `release_date` |
| Models | `jev-1.13.0`; aliases `jev-latest`, `jev-preview` |
| Streaming | none: one request, one JSON body |
| Batching | not documented for Jev |
| Limits | 64k tokens per request; 32k for state + longest question |
| Errors | 401 bad key, 422 validation, 429 rate limit, 529 overloaded. The SDKs also map 400/403/404/5xx. **The error body's JSON shape is not documented** |

Request:

```json
{
  "model": "jev-latest",
  "state": "string | object | array",
  "questions": {
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "tech": null}},
    "<id>": {"type": "score",  "instructions": "...", "criteria": ["low", "medium", "high"]},
    "<id>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}}
  }
}
```

- `instructions` and criterion values may be a string, object or array.
- `choice` takes up to 255 options.
- `score` takes 2–10 levels.
- `noul` criteria are optional.

Response (the strict contract):

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "q1": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.95, "tech": 0.05}, "confidence": 0.8},
    "q2": {"type": "score", "score": 1.7, "legend": {"0": "low", "1": "medium", "2": "high"},
           "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.3},
    "q3": {"type": "noul", "noul": 0.91}
  },
  "usage": {"input_tokens": 83, "output_tokens": 0}
}
```

Jev defines confidence as `(n·p_max − 1)/(n − 1)`, which differs from Laya's
(see 2.5).

### 1.2 `laya-serve`, the reference we imitate (`laya/serve.py`, FastAPI)

**Endpoints**

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | open | Returns `{"status":"ok"}` only, when a key is set and the caller lacks it. Otherwise the full payload: `loaded`, `revisions`, `device`, `device_is_preference`, `checkpoint_devices`, `cpu_fallbacks` |
| `POST /v1/systemone` | bearer, if `LAYA_API_KEY` is set | One state, many questions |
| `POST /v1/systemone/batch` | same | `{"states": [...≤64], "questions": {...}}` → `{"results": [<single response>...], "total_usage": {input_tokens, output_tokens}}` |

Upstream has **no `GET /v1/models`**. `laya-railway` adds one, shaped as
`{"models": [{"name", "description", "release_date"}]}`. It lists `laya-auto`
and `laya-<checkpoint>`, and treats any `jev*` model name as `laya-auto`.

**Auth**

- When `LAYA_API_KEY` is set, the full header `Bearer <key>` is compared in
  constant time, as bytes.
- A failure returns 401 with `invalid or missing bearer token`.

**Request controls beyond `state`, `questions` and `model`**

`task`, `lang`, `lang_guess`, `max_len`, `head_max_len` and `min_confidence`.
Each is forwarded only when present. The hook arguments are refused with a 422:
`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`.

The batch endpoint also takes:
- `batch_size`: an integer ≥ 1.
- `sort_by_length`: a boolean.

**`model` resolution**

- These values pick a checkpoint:
  - the public HF ids `convaiinnovations/laya-multilingual` and
    `…/laya-typed-decisions`;
  - the names `english`, `multilingual`, `typed-decisions`;
  - aliases such as `en`, `ml`, `multi`, `typed`.
- Anything else means auto-route. That includes `jev-1`, `jev-latest` and the
  bundle id `convaiinnovations/laya`.

**Full response** (default, verbatim sample from `docs/http-api.md`)

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "queue": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.9519, "tech": 0.0327, "other": 0.0154},
              "confidence": 0.797, "answer_confidence": 0.9519,
              "action": {"act_probability": 1.0}},
    "urgency": {"type": "score", "score": 1.6994,
                "legend": {"0": "calm", "1": "firm", "2": "angry", "3": "furious"},
                "probabilities": {"0": 0.0249, "1": 0.4136, "2": 0.3985, "3": 0.1629},
                "confidence": 0.1925, "answer_confidence": 0.4136,
                "action": {"act_probability": 1.0}}
  },
  "usage": {"input_tokens": 83, "output_tokens": 0, "state_tokens": 12,
            "state_tokens_dropped": 0, "truncated": false, "truncated_questions": []},
  "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
              "detection": {"script": "latin", "script_profile": {"latin": 1.0}, "language": "en",
                            "is_english": true, "language_undecided": false, "diacritic_rate": 0.0,
                            "non_latin_fraction": 0.0, "mixed_segment": null},
              "workflow": null}
}
```

**Further response details**

- `noul` answers carry `noul`, `confidence`, `answer_confidence` and `action`.
- `usage.options` appears only when options collapsed under the head budget.
- `abstention`, `abstention_threshold` and `low_confidence` appear on every
  answer, but only when `min_confidence` was sent.
- `LAYA_JEV_STRICT=1` projects the response onto the strict contract in 1.1:
  - no `routing`;
  - choice keeps `choice`, `probabilities`, `confidence`;
  - score keeps `score`, `probabilities`, `confidence`, `legend`;
  - noul keeps `noul` only;
  - usage keeps the two counts only.
- Headers: `Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms`.

**Errors**

The body is always FastAPI's `{"detail": "<message>"}`.

| Status | When |
|---|---|
| 400 | not JSON or not an object; no `questions`; `state` missing or null; `questions` not an object; an unpaired `\udXXX` surrogate anywhere in the body; batch `states` empty or not a list |
| 401 | the bearer token is wrong or missing |
| 413 | body over 2 MiB (enforced while streaming, not only via Content-Length); state over 50,000 characters of serialized text; over 64 questions; over 100 choice options; over 32 score levels; over 512 options in total; over 64 batch states |
| 422 | unknown type; options exceed the head token budget; null score level; unknown `task`; bad `lang`, `min_confidence`, `max_len` or `head_max_len` (the last two are capped by `LAYA_MAX_TOKEN_BUDGET`, default 8192); a hook argument was sent |
| 500 | anything else, always with `inference failed` and nothing leaked |
| 503 | `LAYA_MAX_CONCURRENT` (16) requests are already admitted. Comes with `Retry-After: 1`. Load is refused, not queued |

**Concurrency**

- An admission semaphore runs before the body is read.
- After that, a single inference worker runs one forward pass at a time.

**Environment**

`LAYA_HOST`, `LAYA_PORT` (8000), `LAYA_ROOT_PATH`, `LAYA_DEVICE`, `LAYA_PRELOAD`,
`LAYA_MODELS`, `LAYA_THREADS`, `LAYA_AUTO_TASK`, `LAYA_DEFAULT_MODEL`,
`LAYA_MAX_LOADED`, `LAYA_API_KEY`, `LAYA_MAX_CONCURRENT`,
`LAYA_MAX_TOKEN_BUDGET`, `LAYA_JEV_STRICT`.

### 1.3 Proposed compatibility target for laya_server

- Match `laya-serve` byte-for-byte in shape:
  - the same routes, limits, status codes and `{"detail"}` bodies;
  - the same env var names, so the two are interchangeable;
  - `LAYA_JEV_STRICT`.
- Also add `laya-railway`'s `GET /v1/models`, so the TypeSafe SDKs'
  `models.list()` works.

## 2. The model

### 2.1 Licence and checkpoints

- The weights and the upstream code are **Apache-2.0**. Porting upstream logic
  to Dart means keeping its copyright notice and stating the changes; see 4.4.
- Weights are **safetensors only**. The HF repo has no `.onnx` file.

| Checkpoint | Location in `convaiinnovations/laya` | Encoder | Size | `max_len` / `head_max_len` |
|---|---|---|---|---|
| english | root (`model.safetensors`, 843 MB) | ModernBERT-large: 28 layers, hidden 1024, 16 heads, vocab 50368 | 421M | 512 / 192 |
| multilingual | `multilingual/` (644 MB) | mmBERT-base: 22 layers, hidden 768, 12 heads, vocab 256000 | 322M | 1024 / 256 |
| typed-decisions | `typed-decisions/` (843 MB) | ModernBERT-large, fine-tuned | 421M | 1024 / 256 |

Each checkpoint folder contains:
- `rl_agent_config.json`;
- `encoder/config.json`, the architecture only;
- `tokenizer/tokenizer.json` and `tokenizer_config.json`;
- `model.safetensors`, holding encoder and head together.

**Encoder attention**

- Both encoders alternate global and sliding-window attention: a global layer
  every 3rd layer, and a local window of 128.
- RoPE theta: English uses 160000 for global layers and 10000 for local ones.
  mmBERT uses 160000 for both.

### 2.2 Architecture (`laya/common.py:472`, `DecisionModel`)

```
h      = encoder(input_ids, attention_mask).last_hidden_state   # [n, L, d]
h      = h + type_emb(qtype)[:, None, :]                        # qtype: choice=0 score=1 noul=2
h      = 2 × TransformerEncoderLayer(d, d/64 heads, 4d FFN, norm_first) with padding mask
m      = gather(h, marker_pos)                                  # the [MASK] of each option
logits = scorer(m)  →  LayerNorm, Linear(d,d), GELU, Linear(d,1)  → [n, K], masked to -1e4
feats  = [top1, top1-top2, normalized entropy, k/255]
act    = act_head(cat(h[:,0], feats)) → Linear(d+4,256), GELU, Linear(256, 2)   # [act, escalate]
```

### 2.3 Input format (`build_head` / `build_sequence`, `common.py:136-247`)

One row per (state, question):

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] " opt0" [MASK] " opt1" … [SEP] <state> [SEP]
```

**Text normalisation**

- **Instructions** are `str(ins)` with the tokenizer's mask token replaced by a
  space.
- **Options** are each encoded separately as `" " + text`, without special
  tokens, and cut to 48 tokens.
  - choice: `"key: <criterion>"`, or just `key` when the criterion is null or
    empty.
  - score: `"level i: <criterion>"`.
  - noul: `"false: <crit or 'no, the statement does not hold'>"` and
    `"true: <crit or 'yes, the statement holds'>"`, always in false, true
    order.
  - Structured criteria are rendered as compact JSON with separators `", "` and
    `": "`, keeping non-ASCII characters (Python's
    `json.dumps(ensure_ascii=False, separators=(", ", ": "))`).
- **State** is the string itself, or `json.dumps(state, ensure_ascii=False)`
  with default separators `", "` and `": "`, with mask tokens replaced by a
  space.

**Head budget**

- If the options total more than `head_max_len - 16` tokens, each option is cut
  to `max(4, (head_max_len-16)//n)`.
- The instruction is cut to `max(8, remaining budget)`.
- If two options end up with identical token spans, that question is reported
  in `usage.options`.

**State room and truncation**

- The state gets `max_len - len(head) - 1` tokens.
- A plain state is truncated from the right, so its start is kept.
- A conversation (list) state is truncated from the left, so its end is kept.

**Python formatting the port must reproduce exactly**

- Key order and `ensure_ascii=False`.
- Python float repr, for example `1e-05` and `0.1`.
- Python's `json.dumps` default separators.

`laya-ts` reproduces these, and its `parity-sequence` tests cover them.

### 2.4 Tokenisers (from the two `tokenizer.json` files)

| | english / typed-decisions (ModernBERT) | multilingual (mmBERT, Gemma-style) |
|---|---|---|
| Model | BPE: 50,280 vocab, 50,009 merges, no byte fallback | BPE: 256,000 vocab, 580,604 merges, `byte_fallback`, `fuse_unk`, unk `<unk>` |
| Normalizer | NFC | Replace `" "` → `"▁"` |
| Pre-tokenizer | ByteLevel (GPT-2 regex, `add_prefix_space=false`) | Metaspace (`▁`, `prepend_scheme=always`, `split=true`) |
| Specials | `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]` 50284 (`lstrip`) | `<bos>` 2 (CLS), `<eos>` 1 (SEP), `<pad>` 0, `<mask>` 4 (`lstrip`), `<unk>` 3 |
| Added tokens | 116, including runs of spaces (`normalized=true`) | 249 |
| File size | 3.5 MB | 34 MB |

Both are tractable to implement in pure Dart. Dart `RegExp` supports `\p{L}`
and `\p{N}` with `unicode: true`. Parity has to be proven, not assumed:

- `laya-ts/src/tokenizer.ts` is a working port, but it has divergences from HF
  `tokenizers`:
  - it ignores Metaspace `split` and `prepend_scheme`;
  - it ignores `single_word`;
  - it forces NFC on ByteLevel.
- Its only real-checkpoint reference ids are two mmBERT strings, for example
  `"choice question: Which department should handle this request?"` →
  `[6241, 2872, 235292, 12236, 9888, 1412, 6589, 736, 3853, 235336]`.
- **Reference ids for the English tokenizer must be generated with HF
  `tokenizers`.** That needs Python with `tokenizers`, which is not installed on
  this machine (`python` is not on PATH).

### 2.5 Post-processing (`onnx_agent.py:642-702`, `laya-ts/src/agent.ts:715-769`)

**Temperature**

- Bucket key: `<type>:<2|3-5|6-10|11+>`.
- Lookup order: `temperature_by_options[bucket]`, then `temperature[qtype]`,
  then 1.0.
- Clamped to [0.5, 5.0]. This neutralises the English checkpoint's
  `choice:11+ = 0.1006`.

**Probabilities and answers**

- `p = softmax(logits[:k] / T)`, then un-permute `option_order`.
- choice: `choice` is the argmax key.
- score: `score = Σ i·pᵢ`, and `legend` is `{"i": rendered criterion}`.
- noul: `noul = p[1]`.
- `confidence`:
  - choice and score: `1 − H(p)/ln k`, clipped;
  - noul: `max(p, 1−p)`.
- `answer_confidence` is `max(p)`.
- `action.act_probability` is `softmax(act_logits)[0]`.
- Every value is rounded to 4 decimals.

**Usage**

- `input_tokens` is the non-pad tokens summed over that state's question rows.
- `output_tokens` is always 0.
- The truncation fields are described in 1.2.

**Batching**

- Rows are padded to the batch max length with the pad id.
- Markers are padded with position 0 and mask false.

### 2.6 Routing (`laya/router.py:869-960`, `laya/lang.py`)

The router checks these in order and stops at the first that applies:

1. explicit `model`;
2. explicit `task`;
3. question ids match one of 4 typed-decisions workflows (only when
   `auto_task_detection` is on);
4. `lang`;
5. `lang_guess`;
6. `lang.analyse(state)`, which:
   - profiles the script;
   - measures diacritic rate and non-Latin fraction (thresholds 0.02, 0.06,
     0.2, 0.1, 10 letters);
   - identifies the language from stopword lists.

Each step writes a fixed `reason` sentence. `lang.py` is 701 lines of
deterministic, pure logic, so it ports to Dart directly.

## 3. Inference route for Dart

### 3.1 ONNX export feasibility

Upstream already exports to ONNX (`laya-ts/scripts/export_onnx.py`):

- It uses `torch.onnx.export` with dynamo and explicit dynamic dimensions: batch
  1–64, sequence 1–8192, markers 1–256. The opset is 18.
- It produces a split pair:
  - `encoder.onnx`: `input_ids` and `attention_mask`, both int64 `[n,L]` →
    `last_hidden_state`;
  - `head.onnx`: `hidden_states` f32 `[n,L,d]`, `marker_pos` int64 `[n,K]`,
    `marker_mask` **bool** `[n,K]`, `qtype` int64 `[n,1]`, `attention_mask`
    int64 `[n,L]` → `logits`, `act_logits`.
- Before writing anything, it checks that torch and ORT agree within 1e-4 at
  sequence length 16 and at 512 (past the 128 sliding window), and at batch 1
  and 2.

Upstream made the head exportable on purpose: `_DynamicMultiheadAttention`
exists because the stock module baked the traced length into the graph. The
Python `ONNXAgent` uses a single combined `laya.onnx` instead.

So **ModernBERT + head is designed to export cleanly, and upstream ships the
tooling and a numeric check for it.**

What I could not verify:

- **I have not run the export.** It needs Python with torch, transformers,
  onnx, onnxruntime and onnxscript, plus a 0.6–0.8 GB weight download per
  checkpoint. None of that is installed, and the brief says to ask before
  downloading weights.
- Whether the export holds for mmBERT and typed-decisions at our pinned
  versions, and whether a long-sequence run (1024+ for multilingual) stays
  within 1e-4.
- `laya-ts` examples load `encoder.onnx` from
  `huggingface.co/convaiinnovations/laya/resolve/...`, but the HF tree at the
  pinned revision has no ONNX files. Those examples either assume a
  self-exported mirror or are stale. **We must host our own exports**, or
  document how users export them.
- int8 quantisation (about 4× smaller) is untested, and so is its effect on
  calibration.
- CPU latency on the target hardware. `laya-railway` reports 0.2–0.9 s per call
  on 8 vCPU with torch.

### 3.2 Candidates

| Route | For | Against |
|---|---|---|
| **A. ONNX Runtime C API via our own `dart:ffi` bindings** (generated with `ffigen` from `onnxruntime_c_api.h`) | Native ORT speed, multithreaded, `ORT_ENABLE_ALL` fusion. Single process. The exact graph upstream verifies. The C API is stable and versioned (latest release v1.30.0). Bool and int64 tensors are supported | We must ship or locate `libonnxruntime` per platform. Small FFI surface to own: about 15 C API calls |
| A′. Same, using an existing pub package | Less code | Every mature package (`onnxruntime` 1.4.1, `flutter_onnxruntime`, `onnxruntime_plus`) **depends on Flutter**, so it is unusable on a Dart server. `dart_onnx` 0.1.1 is pure Dart, but it is 6 months old with 1 like and its dtype coverage is undocumented. Not a dependency to publish on |
| B. Pure-Dart interpreter (`onnx_runtime_dart` 0.10.7, MIT) | No native code at all | It reports 4.2× slower than ORT on a 22M-param MiniLM, single-threaded. On a 421M ModernBERT-large that projects to several seconds per call. ModernBERT-large's exported op set is unverified on it |
| C. libtorch / candle bridge | Loads safetensors directly, no export | Needs a C or Rust shim we build and ship per platform, and the native artifacts are large. Candle would need a reimplementation of ModernBERT and the head in Rust. The most native code, for no gain over A |
| D. Dart front end + Python sidecar (`laya-serve` itself) | Exact parity for free | Two runtimes and two processes to deploy. Hardly a "Dart server" and not useful on pub.dev. Dart would only be a proxy |

### 3.3 Recommendation: route A, split ONNX, with Dart owning everything else

**What Dart owns**

- HTTP (`shelf`), validation and limits.
- Both tokenisers, sequence building, routing and language analysis.
- Batching, post-processing and the response shape.

ORT runs only the two graphs, `encoder.onnx` and `head.onnx`, through a thin
`dart:ffi` layer we generate and own.

**Why**

- It is the only route that is both fast enough for a 421M encoder on CPU and
  publishable as a pure-Dart (non-Flutter) package.
- The split graph is exactly what upstream verifies and what `laya-ts` runs, so
  `laya-ts` serves as a second reference implementation for every
  post-processing detail.

**Loading the native library**

1. An explicit path wins: `LAYA_ORT_LIB` or a constructor argument.
2. Otherwise, a Dart **build hook** (`hook/build.dart`, native assets, stable
   in Dart 3.13) fetches the pinned official ORT release for the host OS and
   checks its sha256.
3. Otherwise, a clear error saying how to install ORT.

No binaries are committed and no system-wide install is needed.

**Weights**

- Weights are never bundled.
- A Dart CLI (`dart run laya_server:fetch`) downloads `tokenizer.json` and
  `rl_agent_config.json` from HF at a pinned revision, plus our exported ONNX
  from wherever we host it.
- It also documents running upstream's `export_onnx.py` yourself.

**Fallback**

The engine sits behind an `InferenceBackend` interface. That keeps a pure-Dart
backend (route B) or a sidecar backend (route D) possible later, and lets the
API tests run with a fake backend and no model.

## 4. Implications for publishing on pub.dev and as open source

1. **Package shape.** The proposed layout is one repository with:
   - `lib/`, the library: tokenizer, sequence, router, decode, ORT backend,
     shelf handler;
   - `bin/laya_server.dart`, the server executable;
   - `bin/fetch.dart`;
   - `example/`.

   A later split into `laya` (core) and `laya_server` (HTTP) is possible.
   Starting with one package keeps the pub score simple.
2. **Pub score.** The package needs:
   - `dart analyze` clean under `package:lints/recommended` (stricter if we
     choose);
   - dartdoc on every public API;
   - an `example/`, a `CHANGELOG.md` and a `LICENSE`;
   - `repository`/`issue_tracker` in the pubspec, and no Flutter dependency.

   An FFI package scores only on the platforms it supports: Windows, Linux and
   macOS. There is no web support, which is acceptable for a server.
3. **Tests run without the model.** CI must not download 2 GB. The tests split
   three ways:
   - tokeniser tests against committed reference-id fixtures, generated once
     with HF `tokenizers` by a Dart-driven script;
   - API tests against a fake backend, using the request/response examples
     above;
   - an opt-in integration test (`--tags model`) that runs real inference when
     the ONNX files are present.
4. **Licence.** Upstream is Apache-2.0, and we are porting its logic. These are:
   - the sequence format, post-processing, router and `lang.py` thresholds;
   - the TS tokenizer behaviour.

   The simplest compliant choice is **Apache-2.0 for laya_server too**, with a
   `NOTICE` crediting `laya` (© its authors) and the files derived from it.
   This decision is yours.
5. **Naming.** `laya_server` is free to check on pub.dev; I have not checked it
   yet. The README should state plainly that this is an independent port, not
   affiliated with ConvAI Innovations or TypeSafe.
6. **Publishing** goes through you. Nothing gets published or pushed without an
   explicit go-ahead.

## 5. Existing ports (surveyed 2026-10-03)

### 5.1 ONNX exports on Hugging Face

About 50 community exports exist. The two serious ones both cover all three
checkpoints.

**`onnx-community/laya-ONNX`**, plus `-multilingual-ONNX` and
`-typed-decisions-ONNX` (Apache-2.0; `laya-ONNX` was at revision
`42e2a6e3b3708c8ff5f61f4aa75c314b1449199b` when surveyed).

- **Graph:** one graph holding encoder and head together. It is the same I/O
  as upstream's Python `ONNXAgent`:

  | | Name | Type | Shape |
  |---|---|---|---|
  | in | `input_ids` | int64 | `[b,s]` |
  | in | `attention_mask` | int64 | `[b,s]` |
  | in | `marker_pos` | int64 | `[b,k]` |
  | in | `marker_mask` | bool | `[b,k]` |
  | in | `qtype` | int64 | `[b]` |
  | out | `logits` | | `[b,k]` |
  | out | `act_logits` | | `[b,2]` |

- **Files:** `onnx/model.onnx` plus `model.onnx_data` (fp32, 1.69 GB), and
  `model_fp16` (0.85 GB). Also `tokenizer.json`, and `config.json`, whose
  `laya` section carries `max_len`, `head_max_len` and the temperatures. The
  conversion scripts are in `conversion/`.
- **Export:** dynamo, opset 18, with dynamic batch, sequence and marker axes.
- **Stated parity (we have not verified it):**
  - ORT CPU against PyTorch, up to 2,048 tokens: max |Δlogit| 2.4e-5.
  - Against laya's Python `Agent.predict` on 15 cases (several languages,
    edge cases): token ids identical. fp32 max |Δp| < 1e-4; fp16 max |Δp|
    0.007, with no top answer changed.

**`ti3x-m/laya-onnx`**, plus `-multilingual-` and `-typed-decisions-`, mirrored
as `Suarify/…` (Apache-2.0).

- It ships a sha256 `manifest.json` and PyTorch reference fixtures (34 cases),
  plus a weight-only int8 CPU variant of about 0.8 GB.
- The batch size is fixed at 1, so each question is a separate ORT call.

### 5.2 Dart packages on pub.dev

- **`laya_dart` 0.0.4 and `laya_flutter` 0.1.3:** on-device inference, both
  depending on Flutter (`onnxruntime_v2` and `flutter_onnxruntime`). Not usable
  on a Dart server. Neither is a server.
- **`jev_dart`, `system_one`, `typesafe_ai_dart`:** pure-Dart Jev clients.
  Useful as real clients in compatibility tests.
- **`hf_tokenizers` 1.2.2:** pure Dart over FFI to HF's Rust `tokenizers`.
  Its native binary is a prebuilt download or built with `cargo`. Not chosen;
  see decision 6 below.

## 6. Decisions (2026-10-03)

1. **Route A**: ONNX Runtime through our own `dart:ffi` bindings. Revised by
   decision 3: we use the single combined graph rather than the split
   encoder/head pair.
2. **Local Python environment approved.** It is a uv venv at `.tooling/venv`
   (gitignored), not a system-wide install. With the export no longer needed
   (decision 3), it is used for:
   - English tokenizer reference ids, generated with HF `tokenizers`;
   - parity references from laya's Python `Agent` and `laya-serve`.
3. **ONNX source**: `onnx-community/laya-ONNX`,
   `onnx-community/laya-multilingual-ONNX` and
   `onnx-community/laya-typed-decisions-ONNX`, pinned by revision and
   downloaded by our fetch CLI. We do not export or host our own. This replaces
   the earlier "own HF repo" choice.
6. **Tokeniser**: our own pure-Dart implementation of both BPE variants, read
   from `tokenizer.json` and tested against HF-generated reference ids. ORT
   remains the only native dependency.
4. **Licence**: Apache-2.0 plus a `NOTICE` crediting `laya`.
5. **v1 scope**:
   - In: the english, multilingual and typed-decisions checkpoints, with
     `task`/auto-task routing.
   - Out: `POST /v1/systemone/batch` and `GET /v1/models`, deferred until after
     v1.
7. **Measured parity (2026-10-04, Task 17)**: laya_server matches laya-serve
   on real weights.
   - Cases: 65 english and 40 multilingual, all matched.
   - Max |Δp| after rounding: 0 on both checkpoints. Before rounding, the max
     logit drift is 1.2e-5 and 4.0e-5.
   - Median latency, laya-serve vs laya_server: english 672 ms vs 1261 ms,
     multilingual 277 ms vs 725 ms (24 threads). The gap is ONNX Runtime's own.
   - typed-decisions was not tested on real weights.

   See `doc/api.md`, "Measured parity".

The options as they were put:

1. Inference route: **A (recommended)**, B, C or D.
2. Permission to set up the export toolchain. That means a Python environment
   with torch and friends, ORT, and the weights (about 0.8 GB for English,
   about 0.6 GB for multilingual). It is needed to:
   - produce the ONNX files;
   - generate English tokenizer reference ids;
   - run parity checks against `laya-serve`.

   Alternatively, you provide exported files.
3. Where exported ONNX files will be hosted for users: our own HF repo, GitHub
   release assets, or "export it yourself".
4. Licence: Apache-2.0 + NOTICE (recommended).
5. Scope of v1:
   - all three checkpoints, or English and multilingual first;
   - whether the batch endpoint and `/v1/models` are in v1;
   - whether typed-decisions auto-task routing is in v1.

1. Inference route: **A (recommended)**, B, C or D.
2. Permission to set up the export toolchain. That means a Python environment
   with torch and friends, ORT, and the weights (about 0.8 GB for English,
   about 0.6 GB for multilingual). It is needed to:
   - produce the ONNX files;
   - generate English tokenizer reference ids;
   - run parity checks against `laya-serve`.

   Alternatively, you provide exported files.
3. Where exported ONNX files will be hosted for users: our own HF repo, GitHub
   release assets, or "export it yourself".
4. Licence: Apache-2.0 + NOTICE (recommended).
5. Scope of v1:
   - all three checkpoints, or English and multilingual first;
   - whether the batch endpoint and `/v1/models` are in v1;
   - whether typed-decisions auto-task routing is in v1.
