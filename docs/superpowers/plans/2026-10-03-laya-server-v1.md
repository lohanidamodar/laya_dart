# laya_server v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Dart, pub.dev-publishable package and server, `laya_server`, that runs the Laya decision model
locally through ONNX Runtime and serves `laya-serve`'s Jev-compatible HTTP API.

**Architecture:**
- Dart owns:
  - the HTTP surface (`shelf`);
  - request validation and limits;
  - both tokenisers;
  - the Laya sequence format;
  - language analysis and checkpoint routing;
  - batching;
  - turning logits into Jev answers.
- ONNX Runtime runs only the single combined graph published by `onnx-community`. It is reached
  through our own minimal `dart:ffi` bindings and runs on a worker isolate.
- The ORT native library is provided by a Dart build hook (native assets) or by an explicit path.
- Model files are fetched by a CLI at pinned revisions. They are never bundled.

**Tech Stack:**
- Dart ≥ 3.10 (developed on 3.13.4).
- Packages: `shelf`, `ffi`, `hooks`/`code_assets` (native assets), `archive`, `crypto`, `http`,
  `args`, `logging`, `meta`, `path`.
- Test tooling: `test`, `lints`.
- ONNX Runtime 1.30.0 C API.
- Reference tooling: the Python venv at `.tooling/venv`, with laya 0.3.24, tokenizers 0.23.2,
  onnxruntime 1.30.0 and torch 2.14.1+cpu.

**Spec:** `docs/research.md` (the wire protocol in §1, the model in §2, the decisions in §6). Every
behaviour this plan ports is defined by the upstream source it cites. Read the cited lines; they are the
spec, and the golden fixtures enforce them.

Upstream source checkout: `NandhaKishorM/laya` at `fa9a2a7070b1789912a49ae24603bbfb1a78b001`. The
installed copy is at `.tooling/venv/Lib/site-packages/laya/` (laya 0.3.24 from PyPI, the same version).
Cite paths below as `laya/<file>.py:<line>`.

## Global Constraints

- **Package name** `laya_server`. **Licence** Apache-2.0, with a `NOTICE` crediting `laya` (Convai
  Innovations, Apache-2.0) for the ported sequence format, decoding, router and language logic.
- **No Flutter dependency** anywhere. `dart analyze` must be clean, with `package:lints/recommended.yaml`
  plus the stricter rules in Task 1.
- **Every public API member** has a dartdoc comment.
- **Tooling and scripts** are Dart (`tool/*.dart`). The one exception: `tool/reference/*.py` exists only
  to call the Python reference implementation, and is always driven by a Dart script.
- **Gates** are read from a file with the exit code captured, for example:
  `dart test > .tooling/logs/test.txt 2>&1; echo "exit=$?"`. Never pipe a gate into `tail` or `head`.
- **Pinned artifacts:**

  | Artifact | Pin |
  |---|---|
  | `onnx-community/laya-ONNX` | `42e2a6e3b3708c8ff5f61f4aa75c314b1449199b` |
  | `onnx-community/laya-multilingual-ONNX` | `46b77bbf5642fec5f14e540570228a8cbe8ab81f` |
  | `onnx-community/laya-typed-decisions-ONNX` | `79c441b22da79eb3727bb01e8452e29a32cfa548` |
  | `convaiinnovations/laya` (tokenizers, configs) | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |
  | ONNX Runtime | `v1.30.0` |

- **Checkpoint names** `english`, `multilingual`, `typed-decisions`. Aliases come from
  `laya/router.py:77-82`.
- **Response `model` field:** `"laya-rl-agent"`. This is the value `laya-serve` (torch) reports, per
  `docs/http-api.md:140`.
- **Answer numbers** are rounded to 4 decimals, Python `round()` semantics (half-to-even on the binary
  value).
- **Error bodies** are always `{"detail": "<message>"}`, with `laya-serve`'s exact texts and statuses
  (research.md §1.2).
- **Env var names** are identical to `laya-serve`'s:
  - `LAYA_HOST`, `LAYA_PORT`, `LAYA_ROOT_PATH`, `LAYA_PRELOAD`, `LAYA_MODELS`, `LAYA_THREADS`;
  - `LAYA_AUTO_TASK`, `LAYA_DEFAULT_MODEL`, `LAYA_MAX_LOADED`, `LAYA_API_KEY`, `LAYA_MAX_CONCURRENT`;
  - `LAYA_MAX_TOKEN_BUDGET`, `LAYA_JEV_STRICT`.

  Ours add `LAYA_MODEL_DIR` (default `./models`), `LAYA_ORT_LIB` (an explicit library path) and
  `LAYA_ONNX_VARIANT` (`fp32` default, or `fp16`).
- **v1 scope:**
  - In: `POST /v1/systemone`, `GET /health`, all three checkpoints, and `task`/auto-task routing.
  - Out: `POST /v1/systemone/batch` and `GET /v1/models`, which return 404.
- **Downloads:**
  - Already approved: the English and multilingual upstream weights, for Python references.
  - Must be asked first: any ONNX model download (1.3–1.7 GB each), and the typed-decisions upstream
    weights. Task 17 is gated on that question.
- **Committing:** commit only when the user asks. Commit by path (`git commit -- <paths>`), never with
  `git add -A`. Never push.

## Review Focus

1. **Unusual Unicode in states and options.** Emoji (astral characters), combining marks, CRLF, tabs,
   NUL, zero-width spaces, runs of many spaces (English added tokens 50254–50279), and `[MASK]`/`<mask>`
   inside user text. Expected: token ids identical to HF `tokenizers`. Pinned by extra fixture strings in
   Task 2, tested in Tasks 5–6 and 8.
2. **JSON states with numbers.** Floats like `0.1`, `1e-05`, `1e16`, `-0.0`, integers above 2^53, nested
   non-ASCII, and keys in insertion order. Expected: the serialized state text equals Python's
   `json.dumps(..., ensure_ascii=False)`. Pinned in Task 3.
3. **Odd choice criteria.**
   - Labels that look numeric (`"1"`).
   - An empty-string label.
   - Labels sharing a long prefix, which collapse under the head budget.
   - The list form `["a","b"]`.
   - `null`/`""` descriptions.

   Expected: the same option text, the same `usage.options` report, and the original label echoed back.
   Pinned in Tasks 7–8.
4. **Load while inference is busy.** `/health` stays fast during a long forward pass. Request number 17
   gets 503 with `Retry-After: 1`. A slow-streaming body holds admission but not inference. Pinned in
   Task 15.
5. **Broken installs.** These must fail at startup with one clear sentence, never mid-request:
   - model files missing or truncated (a sha256 mismatch);
   - the ORT library missing;
   - an unknown `LAYA_DEFAULT_MODEL`.

   A failure inside inference returns 500 `inference failed`, with the cause only in the log. Pinned in
   Tasks 11, 13 and 15.

---

## File structure

```
pubspec.yaml  LICENSE  NOTICE  README.md  CHANGELOG.md  analysis_options.yaml  .gitignore
hook/build.dart                         # native assets: provides libonnxruntime for the target
hook/ort_release.dart                   # pinned ORT archive URLs + sha256 per OS/arch
lib/laya_server.dart                    # public library: exports
lib/src/json/python_json.dart           # Python json.dumps / float repr emulation
lib/src/tokenizer/tokenizer.dart        # LayaTokenizer (public), factory from tokenizer.json
lib/src/tokenizer/added_tokens.dart     # added/special token splitting (lstrip/rstrip/normalized)
lib/src/tokenizer/bpe.dart              # BPE merge engine (rank-ordered, heap for long words)
lib/src/tokenizer/byte_level.dart       # ModernBERT: NFC + GPT-2 regex + byte map
lib/src/tokenizer/metaspace.dart        # mmBERT: Replace + Metaspace + byte fallback
lib/src/tokenizer/nfc.dart              # Unicode NFC normaliser (data generated by tool/gen_nfc.dart)
lib/src/tokenizer/nfc_data.g.dart       # generated composition/decomposition tables
lib/src/model/question.dart             # typed questions, parsing, internal form
lib/src/model/render.dart               # render_criterion / render_options
lib/src/model/sequence.dart             # build_head / build_sequence / state_room
lib/src/model/checkpoint.dart           # CheckpointSpec, CheckpointConfig (temperatures, lens)
lib/src/model/decode.dart               # temperatures, softmax, answers, confidence, rounding
lib/src/model/agent.dart                # LayaAgent: one checkpoint, predict()
lib/src/routing/lang.dart               # port of laya/lang.py analyse()
lib/src/routing/router.dart             # names, aliases, workflows, route()
lib/src/inference/batch.dart            # InferenceBatch (collate) / InferenceOutput
lib/src/inference/backend.dart          # InferenceBackend interface + FakeBackend (public, for tests)
lib/src/inference/ort/ort_api.dart      # hand-written FFI structs for OrtApiBase/OrtApi
lib/src/inference/ort/ort_library.dart  # loading: LAYA_ORT_LIB path or native asset
lib/src/inference/ort/ort_session.dart  # Env/Session/Value wrappers with explicit release
lib/src/inference/ort_backend.dart      # worker-isolate backend running the Laya graph
lib/src/engine.dart                     # LayaEngine: checkpoints, LRU residency, routing+predict
lib/src/store/model_store.dart          # pinned file lists, verify, download from HF
lib/src/server/config.dart              # ServerConfig.fromEnvironment
lib/src/server/errors.dart              # ApiException(status, detail)
lib/src/server/limits.dart              # body cap, request limits, lone-surrogate walk
lib/src/server/strict.dart              # LAYA_JEV_STRICT projection
lib/src/server/handler.dart             # shelf Handler: routes, auth, admission, timing headers
bin/laya_server.dart                    # `dart run laya_server` — serve
bin/fetch.dart                          # `dart run laya_server:fetch` — download models
tool/gen_fixtures.dart                  # drives tool/reference/*.py → test/fixtures/*.json
tool/gen_nfc.dart                       # builds nfc_data.g.dart from UnicodeData (pinned URL+sha)
tool/pin_ort.dart                       # fetches ORT release digests → hook/ort_release.dart
tool/reference/*.py                     # thin calls into the Python reference implementation
tool/check.dart                         # runs format + analyze + test, each to a log with exit codes
test/...                                # mirrors lib/src; fixtures in test/fixtures/
example/main.dart
```

---

### Task 1: Package scaffold and gates

**Files:**
- Create: `pubspec.yaml`, `analysis_options.yaml`, `.gitignore`, `LICENSE`, `NOTICE`, `CHANGELOG.md`,
  `lib/laya_server.dart`, `tool/check.dart`, `test/smoke_test.dart`
- Modify: `README.md` (keep the existing three lines; add the "not affiliated" sentence)

**Interfaces:**
- Produces: `tool/check.dart`. Every later task runs it as its gate: `dart run tool/check.dart`. It writes
  `.tooling/logs/{format,analyze,test}.txt` and prints `format=<code> analyze=<code> test=<code>`.

- [ ] **Step 1: Create the pubspec**

```yaml
name: laya_server
description: >-
  Run the open Laya decision model locally with ONNX Runtime and serve it over a
  Jev-compatible (TypeSafe System One) HTTP API. Pure Dart, no Python.
version: 0.1.0-dev.1
repository: https://github.com/popupbits/laya_server
issue_tracker: https://github.com/popupbits/laya_server/issues
topics: [ai, onnx, classification, server, ffi]
environment:
  sdk: ^3.10.0
platforms:
  linux:
  macos:
  windows:
executables:
  laya_server: laya_server
  fetch: fetch
```

Then add the dependencies with `dart pub add`. This lets pub pick current versions:

```bash
dart pub add shelf ffi hooks code_assets archive crypto http args logging meta path
dart pub add dev:test dev:lints
```

The repository URL is a placeholder until the GitHub repository exists. Ask the user for the real URL
before the first publish (Task 16).

- [ ] **Step 2: Set up analysis options and ignores**

`analysis_options.yaml`:

```yaml
include: package:lints/recommended.yaml
analyzer:
  language:
    strict-casts: true
    strict-inference: true
    strict-raw-types: true
  exclude: [lib/src/tokenizer/nfc_data.g.dart]
linter:
  rules:
    - public_member_api_docs
    - prefer_final_locals
    - prefer_final_fields
    - unawaited_futures
    - always_declare_return_types
    - avoid_dynamic_calls
    - directives_ordering
```

`.gitignore`:

```
.dart_tool/
.tooling/
models/
build/
pubspec.lock
```

`pubspec.lock` is ignored because this is a library package.

- [ ] **Step 3: Add the licence files**

- `LICENSE`: the full Apache-2.0 text, fetched from `https://www.apache.org/licenses/LICENSE-2.0.txt`.
- `NOTICE`:

```
laya_server
Copyright 2026 PopupBits

This product includes software derived from "laya"
(https://github.com/NandhaKishorM/laya), Copyright Convai Innovations,
licensed under the Apache License, Version 2.0. The sequence format, answer
decoding, checkpoint router and language analysis in lib/src/model/,
lib/src/routing/ and lib/src/server/ are ports of laya's Python sources at
commit fa9a2a7070b1789912a49ae24603bbfb1a78b001, modified for Dart.

The Laya model weights are not distributed with this package. They are
published by Convai Innovations under Apache-2.0; the ONNX conversions used
are published by onnx-community under Apache-2.0.
```

- `CHANGELOG.md`: a `## 0.1.0-dev.1` heading with the line `- Initial development version.`

- [ ] **Step 4: Write the library entry and a smoke test**

`lib/laya_server.dart`:

```dart
/// Run the Laya decision model locally and serve it over a Jev-compatible API.
library;
```

`test/smoke_test.dart`:

```dart
import 'package:laya_server/laya_server.dart' as laya;
import 'package:test/test.dart';

void main() {
  test('library imports', () {
    expect(laya, isNotNull);
  });
}
```

- [ ] **Step 5: Write `tool/check.dart`**

```dart
// Runs the three gates, each into its own log file, and reports every exit code.
import 'dart:io';

Future<int> _gate(String name, List<String> args) async {
  final log = File('.tooling/logs/$name.txt')..createSync(recursive: true);
  final result = await Process.run(Platform.resolvedExecutable, args);
  log.writeAsStringSync('${result.stdout}${result.stderr}');
  return result.exitCode;
}

Future<void> main(List<String> args) async {
  final codes = {
    'format': await _gate('format', ['format', '--output=none', '--set-exit-if-changed', '.']),
    'analyze': await _gate('analyze', ['analyze', '--fatal-infos']),
    'test': await _gate('test', ['test', ...args]),
  };
  stdout.writeln(codes.entries.map((e) => '${e.key}=${e.value}').join(' '));
  exitCode = codes.values.any((c) => c != 0) ? 1 : 0;
}
```

- [ ] **Step 6: Run the gate**

Run `dart pub get`, then `dart run tool/check.dart`. Expect `format=0 analyze=0 test=0`. Read the logs if
any code is nonzero.

---

### Task 2: Golden fixtures from the Python reference

**Files:**
- Create: `tool/gen_fixtures.dart`
- Create: `tool/reference/tokenize.py`, `tool/reference/sequence.py`, `tool/reference/pyjson.py`,
  `tool/reference/lang.py`, `tool/reference/route.py`, `tool/reference/decode.py`,
  `tool/reference/cases.json`
- Create (generated, committed): `test/fixtures/tokenizer_english.json`,
  `test/fixtures/tokenizer_multilingual.json`, `test/fixtures/pyjson.json`, `test/fixtures/sequence.json`,
  `test/fixtures/lang.json`, `test/fixtures/route.json`, `test/fixtures/decode.json`

**Interfaces:**
- Produces fixtures. Each is a JSON object with `"generator"` (laya, tokenizers and Python versions),
  plus a `"cases"` list. The per-fixture shapes are given in Step 2.
- Tests read fixtures with `jsonDecode(File('test/fixtures/<name>.json').readAsStringSync())`.

- [ ] **Step 1: Write the shared input corpus, `tool/reference/cases.json`**

It holds `"texts"` (strings), `"states"` (JSON values) and `"requests"` (`{state, questions}` objects).
It must include every Review Focus input:

- `texts`, at least 60 strings:
  - plain English, contractions (`"don't"`, `"I'll"`), numbers (`"$1,234.56"`);
  - URLs and emails, code (`"if (x) { return y; }"`);
  - each of Hindi, Chinese, Japanese, Korean, Arabic, Russian, Greek, Thai, Spanish with accents,
    German umlauts and Vietnamese;
  - emoji including ZWJ sequences (`"👩‍👩‍👧"`), flags (`"🇳🇵"`), combining marks (`"é"`);
  - decomposed text that NFC changes (`"Café"`);
  - CRLF and tabs (`"a\r\nb\tc"`), NUL (`"a\u0000b"`), zero-width space (`"a​b"`);
  - runs of 1–30 spaces, leading and trailing spaces;
  - `"[MASK]"` and `"<mask>"` inside text, `"[CLS] x [SEP]"`;
  - the empty string, a 3,000-character paragraph, a 40-character single "word" (`"a"*40`), and
    `"|||IP_ADDRESS|||"`.
- `states`, at least 12:
  - plain strings;
  - `{"from":"a@b.c","subject":"Hi","body":"..."}`;
  - nested objects with floats `0.1, 1e-05, 1e16, 1e-7, 123456789.123, -0.0, 2.5, 1.0`;
  - an integer `9007199254740993`, booleans, nulls, non-ASCII keys;
  - a conversation list `[{"role":"user","text":"..."},…]` long enough to be truncated (more than
    600 tokens).
- `requests`, at least 25, each over a state from `states`:
  - all three types, criteria as a map, as a list (choice), structured criteria (object values);
  - noul with and without custom true/false criteria, and with boolean-ish keys;
  - score with 2, 5 and 10 levels;
  - choice with 2, 4, 8, 12, 60 and 100 options;
  - options sharing long prefixes, so they collapse;
  - a numeric-looking label `"1"`, an empty-string label, a null description;
  - instructions as an object;
  - question ids matching each of the 4 typed-decisions workflows.

- [ ] **Step 2: Write the reference scripts**

Each script reads `cases.json` and writes one fixture to stdout. They only call the reference
implementation and never reimplement it.

**`tokenize.py <tokenizer.json> <name>`**
- Uses `tokenizers.Tokenizer.from_file`.
- For each text, records `{"text", "ids": enc(text, add_special_tokens=False).ids}`.
- Also records the 48-truncated option encodings: `" " + text`, first 48 ids. It reproduces
  `common.py:215-221` by calling `transformers.PreTrainedTokenizerFast(tokenizer_file=...)` with
  `truncation=True, max_length=48`.

**`pyjson.py`**
- For each state, records `{"value": state, "dumps": json.dumps(state, ensure_ascii=False),
  "criterion": laya.common.render_criterion(state)}`.

**`sequence.py <english|multilingual|typed-decisions>`**
- Loads the tokenizer with `AutoTokenizer` from the pinned checkpoint's `tokenizer/` folder. That folder
  is downloaded by `gen_fixtures.dart`, tokenizer files only.
- For each request and question, converts the question with `ONNXAgent._to_internal(qdef)`, calls
  `laya.common.build_sequence(tok, state, q, max_len, head_max_len, truncate_left=isinstance(state, list),
  return_stats=True, return_truncation_stats=True)`, and records
  `{"request_index", "qid", "ids", "markers", "stats", "truncation"}`.
- Check `laya/onnx_agent.py:603-641` (`_encode_state`) for the exact `truncate_left` rule and any
  state-id reuse, and mirror it exactly.

**`lang.py`**
- For each text and each string state: `{"text", "analysis": laya.lang.analyse(text)}`.

**`route.py`**
- Constructs `Router(preload=False)` once with `auto_task_detection=False` and once with `True`. Check
  `laya/router.py` for the constructor's exact keyword names.
- For each request, calls `router.route(state, questions)` without loading models (see
  `examples/21_routing_without_running.py` in the upstream repo).
- Also calls it with `model=` each alias, `task=` each name, `lang="de"`, `lang="en-US"`, `lang=""` and
  `lang_guess="fr"`.
- Records the arguments plus the full returned decision dict.

**`decode.py`**
- Builds a decoder without loading weights:
  - `agent = object.__new__(ONNXAgent)`;
  - set `agent.cfg`, `agent.temperature`, `agent.temperature_by_options`, `agent.lang_temperatures = {}`
    from each checkpoint's `rl_agent_config.json`, clamped exactly as `onnx_agent.py:185-193` does.
- Seeds `numpy.random.default_rng(1234)` and creates 200 synthetic items: random type, random k in
  2..12, logits `rng.normal(0, 3, k)`, act logits `rng.normal(0, 2, 2)`.
- Includes edge rows: all-equal logits, one huge logit, `k=1` choice, logits with `-1e4` padding beyond
  k.
- Calls the real `_decode_answers` (`laya/onnx_agent.py:642-702`) with matching item metadata. Read that
  function for the exact `items` shape.
- Runs each item with `min_confidence` of `None`, `0.0` and `0.6`.
- Output shape:
  - `{"configs": {"<checkpoint>": <that checkpoint's onnx-community config.json>}, "cases": [...]}`.
  - Each case is `{"checkpoint", "question" (the request-form question JSON), "logits", "act_logits",
    "min_confidence", "answer"}`. `answer` is the exact dict `_decode_answers` returned for it.

- [ ] **Step 3: Write `tool/gen_fixtures.dart`**

```dart
// Regenerates test/fixtures/*.json from the Python reference in .tooling/venv.
// Usage: dart run tool/gen_fixtures.dart
import 'dart:io';

const _rev = '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851';
const _base = 'https://huggingface.co/convaiinnovations/laya/resolve/$_rev';
const _files = {
  'english': ['tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'rl_agent_config.json'],
  'multilingual': ['multilingual/tokenizer/tokenizer.json',
      'multilingual/tokenizer/tokenizer_config.json', 'multilingual/rl_agent_config.json'],
  'typed-decisions': ['typed-decisions/tokenizer/tokenizer.json',
      'typed-decisions/tokenizer/tokenizer_config.json', 'typed-decisions/rl_agent_config.json'],
};

String get _python => Platform.isWindows
    ? '.tooling/venv/Scripts/python.exe'
    : '.tooling/venv/bin/python';

Future<void> _download(String rel, String dest) async {
  final file = File(dest);
  if (file.existsSync()) return;
  file.parent.createSync(recursive: true);
  final client = HttpClient();
  try {
    final response = await (await client.getUrl(Uri.parse('$_base/$rel'))).close();
    if (response.statusCode != 200) {
      throw HttpException('GET $rel -> ${response.statusCode}');
    }
    await response.pipe(file.openWrite());
  } finally {
    client.close();
  }
}

Future<void> _run(List<String> args, String out) async {
  final result = await Process.run(_python, args,
      environment: {'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1'});
  if (result.exitCode != 0) {
    stderr.writeln(result.stderr);
    throw ProcessException(_python, args, 'exit ${result.exitCode}', result.exitCode);
  }
  File(out).writeAsStringSync(result.stdout as String);
  stdout.writeln('wrote $out');
}

Future<void> main() async {
  for (final entry in _files.entries) {
    for (final rel in entry.value) {
      final local = rel.replaceFirst('${entry.key}/', '');
      await _download(rel, '.tooling/checkpoints/${entry.key}/$local');
    }
  }
  const ref = 'tool/reference';
  await _run(['$ref/tokenize.py', '.tooling/checkpoints/english/tokenizer/tokenizer.json', 'english'],
      'test/fixtures/tokenizer_english.json');
  await _run(['$ref/tokenize.py',
      '.tooling/checkpoints/multilingual/tokenizer/tokenizer.json', 'multilingual'],
      'test/fixtures/tokenizer_multilingual.json');
  await _run(['$ref/pyjson.py'], 'test/fixtures/pyjson.json');
  await _run(['$ref/sequence.py'], 'test/fixtures/sequence.json');
  await _run(['$ref/lang.py'], 'test/fixtures/lang.json');
  await _run(['$ref/route.py'], 'test/fixtures/route.json');
  await _run(['$ref/decode.py'], 'test/fixtures/decode.json');
}
```

`sequence.py` writes all three checkpoints into one file, keyed by checkpoint.

- [ ] **Step 4: Generate the fixtures and sanity-check them**

1. Run `dart run tool/gen_fixtures.dart`.
2. Check that every fixture parses and has a nonzero `cases` count. Write a one-off check in Dart with
   `dart run`, not Python.
3. Spot-check the multilingual fixture against the two reference ids quoted in research.md §2.4. For
   example, `"choice question: Which department should handle this request?"` must give
   `[6241, 2872, 235292, 12236, 9888, 1412, 6589, 736, 3853, 235336]`.

If they differ, stop and report it. Either the corpus or the tokenizer revision is wrong.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 3: Python-compatible JSON serialisation

**Files:**
- Create: `lib/src/json/python_json.dart`
- Test: `test/json/python_json_test.dart`

**Interfaces:**
- Produces:

```dart
/// Python's `json.dumps(value, ensure_ascii=False)` with the given separators.
String pythonJsonDumps(Object? value, {String itemSeparator = ', ', String keySeparator = ': '});

/// Python's `repr(float)` / `json.dumps` text for a double.
String pythonFloatRepr(double value);
```

- [ ] **Step 1: Write the failing tests**

```dart
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/json/python_json.dart';
import 'package:test/test.dart';

void main() {
  group('pythonFloatRepr', () {
    final cases = <double, String>{
      0.1: '0.1', 1e-05: '1e-05', 1e16: '1e+16', 1e-7: '1e-07', 123456789.123: '123456789.123',
      -0.0: '-0.0', 2.5: '2.5', 1.0: '1.0', 1e15: '1000000000000000.0', 0.0001: '0.0001',
      1.5e300: '1.5e+300', double.infinity: 'Infinity', double.negativeInfinity: '-Infinity',
    };
    cases.forEach((value, text) {
      test('$value -> $text', () => expect(pythonFloatRepr(value), text));
    });
    test('NaN', () => expect(pythonFloatRepr(double.nan), 'NaN'));
  });

  test('matches json.dumps on every fixture state', () {
    final fixture = jsonDecode(File('test/fixtures/pyjson.json').readAsStringSync()) as Map;
    for (final c in (fixture['cases'] as List).cast<Map<String, Object?>>()) {
      expect(pythonJsonDumps(c['value']), c['dumps'], reason: '${c['value']}');
    }
  });

  test('escapes like Python with ensure_ascii=False', () {
    expect(pythonJsonDumps('a"b\\c\n\t\u0000\u001f é 😀'),
        r'"a\"b\\c\n\t\u0000\u001f é 😀"');
  });
}
```

`jsonDecode` turns `1.0` into an `int` in Dart only when the source has no fraction. Python keeps
`1.0`, so the fixture's `value` round-trips correctly. Make sure ints stay ints and doubles stay
doubles.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/json/python_json_test.dart > .tooling/logs/t3.txt 2>&1; echo "exit=$?"`. Expect a
nonzero exit, with "pythonFloatRepr isn't defined" in the log.

- [ ] **Step 3: Implement**

**`pythonFloatRepr`**
- Python uses the shortest round-trip digits (`repr`).
- If `1e-4 <= |v| < 1e16`, it prints positional notation with at least one fractional digit (`1.0`).
- Otherwise it prints scientific notation as `<mantissa>e<sign><exp at least 2 digits>`, with `.0`
  dropped from the mantissa when it is integral.
- Implementation:
  - Get the shortest digits from Dart's `v.toString()`. Dart also produces shortest round-trip
    digits, but formats them differently.
  - Parse Dart's output into a digit string plus a decimal exponent, then re-format with Python's rules.
  - Special-case `-0.0` as `'-0.0'`, and `NaN`/`Infinity`/`-Infinity`.

**`pythonJsonDumps`**
- `null`, `true` and `false` map to `null`, `true`, `false`.
- An `int` uses `toString()`.
- A `double` uses `pythonFloatRepr`.
- A `String` is quoted and escaped:
  - `"` → `\"`, `\` → `\\`, and `\n \r \t \b \f` as those escapes;
  - other characters below 0x20 → `\u00XX` (lowercase hex);
  - everything else is literal, including lone surrogates, exactly as Python leaves them.
- A `List` is `[` + items joined by `itemSeparator` + `]`.
- A `Map` is `{` + `"key"` + `keySeparator` + value, joined by `itemSeparator`, + `}`. Keys go through
  the string quoting; non-string keys are converted with `toString()`.
- Anything else throws `ArgumentError`.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 4: Tokenizer core — loading, added tokens, BPE engine

**Files:**
- Create: `lib/src/tokenizer/tokenizer.dart`, `lib/src/tokenizer/added_tokens.dart`,
  `lib/src/tokenizer/bpe.dart`
- Test: `test/tokenizer/bpe_test.dart`, `test/tokenizer/added_tokens_test.dart`

**Interfaces:**
- Produces:

```dart
/// A tokenizer loaded from a Hugging Face `tokenizer.json` (ByteLevel or Metaspace BPE).
abstract class LayaTokenizer {
  /// Parses [json] (the decoded tokenizer.json) and picks the implementation by pre_tokenizer.
  factory LayaTokenizer.fromJson(Map<String, Object?> json);
  /// Reads and parses a tokenizer.json file.
  static Future<LayaTokenizer> load(String path);
  /// Ids for [text] without special tokens (HF `add_special_tokens=False`).
  List<int> encode(String text);
  /// The CLS token id ([CLS] or <bos>).
  int get clsId;
  /// The SEP token id ([SEP] or <eos>).
  int get sepId;
  /// The PAD token id.
  int get padId;
  /// The MASK token id.
  int get maskId;
  /// The MASK token text ([MASK] or <mask>), which user text has replaced by a space.
  String get maskToken;
}

/// Rank-ordered BPE over pre-split words.
class BpeModel {
  BpeModel({required Map<String, int> vocab, required List<(String, String)> merges});
  /// Merges [symbols] (one word's initial symbols) and returns the final symbols.
  List<String> mergeWord(List<String> symbols);
  /// Looks up a token id; null when absent.
  int? idOf(String token);
}

/// One piece of input after added-token splitting: either raw text or a matched added token.
sealed class Piece {}
final class TextPiece extends Piece { TextPiece(this.text); final String text; }
final class TokenPiece extends Piece { TokenPiece(this.id); final int id; }

/// Splits text around added tokens the way HF's AddedVocabulary does.
class AddedTokenSplitter {
  AddedTokenSplitter(List<Map<String, Object?>> addedTokens, {required String Function(String) normalize});
  List<Piece> split(String text);
}
```

- [ ] **Step 1: Write the failing tests**

The test vectors are from `laya-ts/tests/bpe.test.ts:41-51` and `tokenizer-parity.test.ts:149-160`, as
cited in research.md §2.4.

```dart
import 'package:laya_server/src/tokenizer/added_tokens.dart';
import 'package:laya_server/src/tokenizer/bpe.dart';
import 'package:test/test.dart';

void main() {
  test('lowest rank merges first, leftmost on ties', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'b': 1, 'ab': 2, 'abab': 3, 'ba': 4},
      merges: [('a', 'b'), ('ab', 'ab'), ('b', 'a')],
    );
    expect(bpe.mergeWord(['a', 'b', 'a', 'b']), ['abab']);
    expect(bpe.mergeWord(['b', 'a', 'b']), ['b', 'ab']);
  });

  test('long words take the heap path and agree with the simple path', () {
    final bpe = BpeModel(
      vocab: {'a': 0, 'aa': 1, 'aaaa': 2},
      merges: [('a', 'a'), ('aa', 'aa')],
    );
    expect(bpe.mergeWord(List.filled(40, 'a')), List.filled(10, 'aaaa'));
  });

  test('lstrip on [MASK] eats the space before it', () {
    final splitter = AddedTokenSplitter([
      {'id': 50284, 'content': '[MASK]', 'lstrip': true, 'rstrip': false,
       'normalized': false, 'single_word': false, 'special': true},
    ], normalize: (s) => s);
    final pieces = splitter.split('a [MASK] b');
    expect(pieces.length, 3);
    expect((pieces[0] as TextPiece).text, 'a');
    expect((pieces[1] as TokenPiece).id, 50284);
    expect((pieces[2] as TextPiece).text, ' b');
  });
}
```

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/tokenizer > .tooling/logs/t4.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

**`BpeModel.mergeWord`**
- For words shorter than 32 symbols: repeatedly find the adjacent pair with the lowest merge rank,
  leftmost on ties, and merge every occurrence of that pair left to right in one pass, as HF's
  `Word::merge_all` does.
- For words of 32 symbols or more: use a priority queue keyed by `(rank, position)` over a
  doubly-linked symbol list, skipping stale entries.

Both paths must give identical output; Step 1 checks this. `laya-ts/src/tokenizer.ts:128-238` shows
both.

**`AddedTokenSplitter`** mirrors HF `AddedVocabulary::split_with_indices`:
- Phase 1 matches `normalized=false` tokens on the raw text.
- Phase 2 matches `normalized=true` tokens on the remaining text pieces, after `normalize`.
- Matching is leftmost-longest, done as a sorted alternation `RegExp` of escaped contents, longest
  first.
- `lstrip` removes trailing whitespace from the preceding text piece; `rstrip` removes leading
  whitespace from the following one.
- `single_word` is honoured: a match needs non-word characters or a boundary on both sides.
- Empty text pieces are dropped.

**`LayaTokenizer.fromJson`**
- Reads `model.vocab`. Merges come as `["a","b"]` pairs or `"a b"` strings; handle both.
- Reads `added_tokens`.
- Chooses `ByteLevelTokenizer` or `MetaspaceTokenizer` (Tasks 5–6) by `pre_tokenizer.type`.
- Special ids are resolved by content: `[CLS]`/`<bos>`, `[SEP]`/`<eos>`, `[PAD]`/`<pad>`,
  `[MASK]`/`<mask>`, using `tokenizer_config.json` semantics. Check them against the
  `post_processor.special_tokens` ids for the English file (research.md §2.4).
- Until Tasks 5–6 land, the factory throws `UnsupportedError` for each type.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 5: ByteLevel BPE (English / typed-decisions) with NFC

**Files:**
- Create: `lib/src/tokenizer/byte_level.dart`, `lib/src/tokenizer/nfc.dart`,
  `lib/src/tokenizer/nfc_data.g.dart`, `tool/gen_nfc.dart`
- Test: `test/tokenizer/byte_level_test.dart`, `test/tokenizer/nfc_test.dart`

**Interfaces:**
- Consumes: `BpeModel`, `AddedTokenSplitter` and `LayaTokenizer` from Task 4.
- Produces: `class ByteLevelTokenizer implements LayaTokenizer`, and `String nfc(String input)`.
- `encode` must equal HF `tokenizers` for every fixture string.

- [ ] **Step 1: Write the failing tests**

```dart
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/tokenizer/nfc.dart';
import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

void main() {
  late LayaTokenizer tok;
  late Map<String, Object?> fixture;
  setUpAll(() async {
    tok = await LayaTokenizer.load('.tooling/checkpoints/english/tokenizer/tokenizer.json');
    fixture = jsonDecode(File('test/fixtures/tokenizer_english.json').readAsStringSync())
        as Map<String, Object?>;
  });

  test('special ids', () {
    expect([tok.clsId, tok.sepId, tok.padId, tok.maskId], [50281, 50282, 50283, 50284]);
    expect(tok.maskToken, '[MASK]');
  });

  test('every fixture text matches HF tokenizers', () {
    for (final c in (fixture['cases'] as List).cast<Map<String, Object?>>()) {
      expect(tok.encode(c['text']! as String), c['ids'], reason: jsonEncode(c['text']));
    }
  });

  test('NFC composes', () {
    expect(nfc('Café'), 'Café');
    expect(nfc('가'), '가');
  });
}
```

The tokenizer files must not be read from a cache that only some machines have. Add a
`test/support/checkpoints.dart` helper:
- It returns the path under `.tooling/checkpoints`.
- When that is missing, it skips the test with the message
  `run dart run tool/gen_fixtures.dart first`.
- Use it from every test that needs a real tokenizer.

Also copy the two small tokenizer configs and the English `tokenizer.json` (3.5 MB) into
`test/fixtures/tokenizers/`, so pub's test run and CI work offline. The 34 MB multilingual file stays out
of the repository. Its test skips when the file is missing, and `tool/check.dart` prints a loud warning
in that case.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/tokenizer/byte_level_test.dart test/tokenizer/nfc_test.dart >
.tooling/logs/t5.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Write `tool/gen_nfc.dart` and generate `nfc_data.g.dart`**

1. The tool downloads `https://www.unicode.org/Public/15.1.0/ucd/UnicodeData.txt` and
   `CompositionExclusions.txt`. It verifies their sha256 against constants written into the tool the
   first time it runs, printed for review.
2. It emits these tables:
   - canonical decomposition, recursive;
   - canonical combining class;
   - primary composites, excluding composition exclusions and singletons;
   - the Hangul algorithmic constants.
3. Check which Unicode version Rust's `unicode-normalization` crate uses in `tokenizers` 0.23.2 and use
   that version. If the versions differ, record the difference in a comment in the tool.

`nfc(String)`:
1. Fully decompose, iterating by code point.
2. Canonically order runs of nonzero combining class (a stable sort by ccc).
3. Compose, using the standard algorithm with blocked-starter rules.
4. Fast path: return the input unchanged when every code point is below U+0300.

- [ ] **Step 4: Implement `ByteLevelTokenizer.encode`**

1. Split with `AddedTokenSplitter`. Its `normalize` is `nfc`, because the normalizer is NFC.
2. For each `TextPiece`:
   1. Apply `nfc`.
   2. Split with the GPT-2 regex, using `unicode: true`:
      `r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"`.
   3. Map each piece's UTF-8 bytes through the GPT-2 bytes→unicode table
      (`laya-ts/src/tokenizer.ts:42-55`).
   4. Run `mergeWord` on its characters.
   5. Map the resulting symbols to ids. A missing symbol throws `StateError`: it cannot happen with a
      full byte vocab, so it signals a bug.
3. A `TokenPiece` contributes its id.

Before trusting this split, check the regex `\s` against HF. HF's regex is Oniguruma's `\s`; Dart's is
JavaScript's. They differ on U+0085, U+180E and U+FEFF; add these three to `cases.json`. If the fixtures
disagree, replace `\s` with an explicit character class matching HF's.

- [ ] **Step 5: Run the tests and check they pass**

Run the same command; expect `exit=0`. Any fixture mismatch is a bug in the tokenizer, never in the
fixture.

- [ ] **Step 6: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 6: Metaspace BPE with byte fallback (multilingual)

**Files:**
- Create: `lib/src/tokenizer/metaspace.dart`
- Test: `test/tokenizer/metaspace_test.dart`

**Interfaces:**
- Consumes: Task 4.
- Produces: `class MetaspaceTokenizer implements LayaTokenizer`.

- [ ] **Step 1: Write the failing tests**

These mirror Task 5's tests, with these changes:
- Load `.tooling/checkpoints/multilingual/tokenizer/tokenizer.json`.
- Expect special ids `cls=2, sep=1, pad=0, mask=4` and mask token `<mask>`.
- Iterate `test/fixtures/tokenizer_multilingual.json`.
- Add the two literal laya-ts vectors from research.md §2.4 as explicit `expect`s.

Also add a test on a tiny inline synthetic tokenizer.json for byte fallback, built in the test: vocab
`{"<unk>":0, "<0xF0>":1, "<0x9F>":2, "<0x98>":3, "<0x80>":4, "▁a":5, "b":6, ...}`. It asserts that
`"a😀b"` falls back to the four byte tokens, in the shape of `tokenizer-parity.test.ts:92`.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/tokenizer/metaspace_test.dart > .tooling/logs/t6.txt 2>&1; echo "exit=$?"` and
expect a nonzero exit.

- [ ] **Step 3: Implement**

Follow HF's semantics from the JSON exactly, not laya-ts's simplifications:

1. Split with `AddedTokenSplitter`. Its `normalize` applies the `Replace` normalizer(s) in order.
2. Apply the normalizer to each `TextPiece`.
3. Run the Metaspace pre-tokenizer:
   - `prepend_scheme: "always"` adds `▁` to the start of every piece that does not already start with
     it.
   - `split: true` splits before each `▁`, keeping it as the prefix of the next word.
4. BPE each word over its characters.
5. With `byte_fallback`, a symbol missing from the vocab becomes one `<0xNN>` per UTF-8 byte (uppercase
   hex). If any byte token is missing, the whole symbol becomes `unk`.
6. With `fuse_unk`, consecutive `unk`s are fused into one.

HF applies the pre-tokenizer per added-token-free segment. Confirm with the fixtures that a segment
following a special token also gets the `▁` prefix; the `"a <t> b"` → `[3,0,8,4]` vector says so.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 7: Questions — typed models, validation, rendering

**Files:**
- Create: `lib/src/model/question.dart`, `lib/src/model/render.dart`, `lib/src/server/errors.dart`
- Test: `test/model/question_test.dart`, `test/model/render_test.dart`

**Interfaces:**
- Consumes: `pythonJsonDumps` (Task 3).
- Produces:

```dart
/// An HTTP-mappable failure: [status] and the client-safe [detail].
class ApiException implements Exception {
  const ApiException(this.status, this.detail, {this.headers = const {}});
  final int status;
  final String detail;
  final Map<String, String> headers;
}

/// Question type, with the head's type id.
enum QuestionType { choice(0), score(1), noul(2); const QuestionType(this.id); final int id; }

/// One question in internal form (laya's `_to_internal`).
sealed class Question {
  QuestionType get type;
  /// Instructions as text: strings pass through, anything else is Python json.dumps.
  String get instructions;
  /// Option texts in label-index order (laya `render_options`).
  List<String> renderOptions();
  /// Parses one question value from a request; throws [ApiException] 422 like laya.
  static Question parse(String id, Object? json);
}
final class ChoiceQuestion extends Question { /* labels: List<String>, criteria: List<Object?> */ }
final class ScoreQuestion extends Question { /* levels: List<Object?> */ }
final class NoulQuestion extends Question { /* trueCriterion, falseCriterion, labels */ }

/// laya `render_criterion`: strings pass through, else compact Python JSON.
String renderCriterion(Object? value);
```

- [ ] **Step 1: Write the failing tests**

```dart
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/server/errors.dart';
import 'package:test/test.dart';

void main() {
  test('choice renders key: criterion, bare key for null/empty', () {
    final q = Question.parse('q', {
      'type': 'choice', 'instructions': 'Which team?',
      'criteria': {'billing': 'refunds', 'tech': null, '1': '', 'x': {'d': 1}},
    });
    expect(q.renderOptions(), ['billing: refunds', 'tech', '1', 'x: {"d": 1}']);
  });

  test('choice list criteria become labels', () {
    final q = Question.parse('q', {'type': 'choice', 'instructions': 'i', 'criteria': ['a', 'b']});
    expect(q.renderOptions(), ['a', 'b']);
  });

  test('score levels', () {
    final q = Question.parse('q', {'type': 'score', 'instructions': 'i', 'criteria': ['low', 'high']});
    expect(q.renderOptions(), ['level 0: low', 'level 1: high']);
  });

  test('noul defaults and custom criteria, false first', () {
    final q = Question.parse('q', {'type': 'noul', 'instructions': 'i'});
    expect(q.renderOptions(), [
      'false: no, the statement does not hold', 'true: yes, the statement holds']);
    final c = Question.parse('q', {'type': 'noul', 'instructions': 'i',
        'criteria': {'True': 'yes it is', 'false': ''}});
    expect(c.renderOptions(), ['false: no, the statement does not hold', 'true: yes it is']);
  });

  test('object instructions are Python json', () {
    final q = Question.parse('q', {'type': 'noul', 'instructions': {'a': [1, 2.0]}});
    expect(q.instructions, '{"a": [1, 2.0]}');
  });

  test('unknown type is 422 naming the question', () {
    expect(() => Question.parse('q1', {'type': 'rank', 'instructions': 'i'}),
        throwsA(isA<ApiException>().having((e) => e.status, 'status', 422)
            .having((e) => e.detail, 'detail', contains('q1'))));
  });
}
```

Add a second test file that runs every question in `test/fixtures/sequence.json`'s source requests. It
asserts that `renderOptions()` equals the Python `render_options` output. Add that output to
`sequence.py` as an `"options"` field per case, then regenerate.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/model > .tooling/logs/t7.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

- Port `laya/onnx_agent.py:223-239` (`_to_internal`) and `laya/common.py:81-133` (`render_criterion`,
  `_resolve_noul_labels`, `render_options`), line by line.
- The 422 texts are the `ValueError` messages laya raises. Find each `raise ValueError` reachable from
  `predict` for question validation:
  - `grep -n "raise ValueError" .tooling/venv/Lib/site-packages/laya/agent.py
    .tooling/venv/Lib/site-packages/laya/common.py`;
  - reproduce each message text exactly, including `labels is only supported for noul questions` and
    the noul labels message.
- Python's `str(k)` on a choice label is already a string in JSON input, so keep labels as their
  original `String`.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 8: Sequence building

**Files:**
- Create: `lib/src/model/sequence.dart`, `lib/src/model/checkpoint.dart`
- Test: `test/model/sequence_test.dart`

**Interfaces:**
- Consumes: `LayaTokenizer` (Tasks 4–6), `Question` (Task 7), `pythonJsonDumps` (Task 3).
- Produces:

```dart
/// Static facts about one checkpoint.
class CheckpointSpec {
  const CheckpointSpec({required this.name, required this.repo, required this.onnxRepo,
      required this.onnxRevision, required this.maxLen, required this.headMaxLen});
  final String name;          // english | multilingual | typed-decisions
  final String repo;          // public HF id reported in routing.repo
  final String onnxRepo;      // onnx-community/...
  final String onnxRevision;
  final int maxLen;
  final int headMaxLen;
  /// The three shipped checkpoints, keyed by name.
  static const Map<String, CheckpointSpec> all = { /* values from research.md §2.1 + Global Constraints */ };
}

/// One encoded row: a question over a state.
class SequenceRow {
  final List<int> ids;
  final List<int> markers;          // [MASK] position per option
  final OptionStats options;        // options, distinct, tokensPerOption
  final TruncationStats truncation; // stateTokens, used, dropped, truncated
}

/// laya `build_sequence` (return_stats + return_truncation_stats).
SequenceRow buildSequence(LayaTokenizer tok, Object state, Question q,
    {required int maxLen, required int headMaxLen, bool truncateLeft = false, List<int>? stateIds});

/// Tokenizes the serialized state once (mask token replaced by a space).
List<int> encodeState(LayaTokenizer tok, Object state);

/// laya `state_room`.
int stateRoom(LayaTokenizer tok, Question q, {required int maxLen, required int headMaxLen});
```

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/model/checkpoint.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/sequence.dart';
import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

import '../support/checkpoints.dart';

void main() {
  final fixture = jsonDecode(File('test/fixtures/sequence.json').readAsStringSync()) as Map;
  final requests = (fixture['requests'] as List).cast<Map<String, Object?>>();
  for (final name in ['english', 'multilingual', 'typed-decisions']) {
    test('$name rows match laya build_sequence', () async {
      final tok = await LayaTokenizer.load(tokenizerPath(name));
      final spec = CheckpointSpec.all[name]!;
      for (final c in (fixture[name] as List).cast<Map<String, Object?>>()) {
        final req = requests[c['request_index']! as int];
        final state = req['state']!;
        final qid = c['qid']! as String;
        final q = Question.parse(qid, (req['questions']! as Map)[qid]);
        final row = buildSequence(tok, state, q, maxLen: spec.maxLen, headMaxLen: spec.headMaxLen,
            truncateLeft: state is List);
        final why = '$name request ${c['request_index']} $qid';
        expect(row.ids, c['ids'], reason: why);
        expect(row.markers, c['markers'], reason: why);
        expect(row.options.toJson(), c['stats'], reason: why);
        expect(row.truncation.toJson(), c['truncation'], reason: why);
      }
    });
  }
}
```

- [ ] **Step 2: Run the test and check it fails**

Run `dart test test/model/sequence_test.dart > .tooling/logs/t8.txt 2>&1; echo "exit=$?"` and expect a
nonzero exit.

- [ ] **Step 3: Implement**

Port `laya/common.py:136-259` exactly:

- The head text is `"${q.type.name} question: ${ins}"`, with the mask token replaced by `" "`.
- Each option is `tok.encode(" " + opt.replaceAll(maskToken, " "))`, taking the first 48 ids. Prefix it
  with `maskId`.
- Budget: `optBudget = headMaxLen - sum(len)`.
  - If `optBudget < 16`, set `per = max(4, (headMaxLen - 16) ~/ max(1, n))` and cut each option to
    `per`.
  - Then cut the head ids to `max(8, optBudget)`.
- Assemble `[cls] + head + [sep] + options + [sep]`. Markers are the index of each option's mask.
- `room = max(0, maxLen - len - 1)`.
- The state is clamped to `room`: from the start, or from the end when `truncateLeft`.
- Append `[sep]`. Clamp the whole row to `maxLen`, and drop markers at or beyond `maxLen`.
- `options.distinct` counts distinct option id lists after capping.

`serialize_state`:
- A `String` is used as-is.
- Anything else goes through `pythonJsonDumps`.
- Then replace the mask token with `" "`.

Put the three `CheckpointSpec` constants in `checkpoint.dart`, with the pinned onnx revisions from
Global Constraints.

- [ ] **Step 4: Run the test and check it passes**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 9: Decoding — temperatures, answers, confidence, gating, strict projection

**Files:**
- Create: `lib/src/model/decode.dart`, `lib/src/server/strict.dart`
- Modify: `lib/src/model/checkpoint.dart` (add `CheckpointConfig`)
- Test: `test/model/decode_test.dart`, `test/server/strict_test.dart`

**Interfaces:**
- Consumes: `Question` (Task 7), `SequenceRow` (Task 8).
- Produces:

```dart
/// Calibration and windows read from a checkpoint's config (onnx-community config.json `laya` section).
class CheckpointConfig {
  CheckpointConfig({required this.maxLen, required this.headMaxLen,
      required List<double> temperature, required Map<String, double> temperatureByOptions});
  factory CheckpointConfig.fromOnnxConfig(Map<String, Object?> configJson);
  final int maxLen;
  final int headMaxLen;
  /// Clamped to [0.5, 5.0]; non-numbers become 1.0.
  final List<double> temperature;
  final Map<String, double> temperatureByOptions;
  /// Temperature for a question type with [k] options (bucket, then type, then 1.0).
  double temperatureFor(QuestionType type, int k);
}

/// Decodes one question's head output into its Jev answer map (full laya payload keys).
Map<String, Object?> decodeAnswer(Question q, List<double> logits, List<double> actLogits,
    CheckpointConfig cfg, {double? minConfidence});

/// Python round(x, 4).
double round4(double x);

/// LAYA_JEV_STRICT projection of a full response (laya serve.py:128-163).
Map<String, Object?> projectJevStrict(Map<String, Object?> result);
```

- [ ] **Step 1: Write the failing tests**

```dart
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/model/checkpoint.dart';
import 'package:laya_server/src/model/decode.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:test/test.dart';

void main() {
  final fixture = jsonDecode(File('test/fixtures/decode.json').readAsStringSync()) as Map;
  test('decodeAnswer equals laya _decode_answers on every fixture row', () {
    for (final c in (fixture['cases'] as List).cast<Map<String, Object?>>()) {
      final cfg = CheckpointConfig.fromOnnxConfig(
          (fixture['configs'] as Map)[c['checkpoint']] as Map<String, Object?>);
      final q = Question.parse('q', c['question']);
      final got = decodeAnswer(q, (c['logits']! as List).cast<num>().map((e) => e.toDouble()).toList(),
          (c['act_logits']! as List).cast<num>().map((e) => e.toDouble()).toList(), cfg,
          minConfidence: (c['min_confidence'] as num?)?.toDouble());
      expect(jsonEncode(got), jsonEncode(c['answer']), reason: jsonEncode(c));
    }
  });

  test('round4 is Python round', () {
    expect(round4(0.12345), 0.1235); // 0.12345 is stored just above .12345
    expect(round4(2.675e-1), 0.2675);
    expect(round4(1.00005), 1.0);    // stored below the midpoint
  });

  test('temperature clamp neutralises choice:11+ = 0.1006', () {
    final cfg = CheckpointConfig(maxLen: 512, headMaxLen: 192, temperature: [1, 1, 1],
        temperatureByOptions: {'choice:11+': 0.10058280825614929});
    expect(cfg.temperatureFor(QuestionType.choice, 12), 0.5);
  });
}
```

Comparing JSON text also pins key order. That matters for byte-compatible output: Jev clients do not
care, but the `laya-serve` parity test in Task 17 does.

`strict_test.dart` builds a full result with a choice, a score and a noul answer. It asserts the exact
strict output from research.md §1.2, and that an unknown answer type passes through unchanged.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/model/decode_test.dart test/server/strict_test.dart > .tooling/logs/t9.txt 2>&1;
echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

Port `laya/onnx_agent.py:642-702` together with `common.py:664-725` (`answer_confidence`,
`confidence_from_probs`, `temp_bucket`, `clamp_temperature`), and `laya/confidence.py` for the
abstention fields:
- `abstention` is `passed`, `abstained` or `unevaluated`;
- `abstention_threshold`;
- `low_confidence` appears only when abstained.

Notes:
- The softmax is max-subtracted and runs in `double`. The reference computes it in float32 numpy or
  torch; check the dtype in `_decode_answers`. If it is float32, emulate it with `Float32List` for the
  logits/T and softmax steps, so 4-decimal rounding matches. Verify against the fixtures; they decide.
- `round4`: Python's `round(x, 4)` is correctly rounded on the exact binary value. Implement it as:
  1. Take `x.toStringAsFixed(4)`. Dart's version rounds the exact binary value, half away from zero.
  2. A tie at exactly .5 ulp at 4 decimals cannot occur for binary doubles except for exactly
     representable values like 0.00005 × 2^k. Handle exact ties with half-even, by checking
     `(x * 10000)` exact equality to `n + 0.5` using a `BigInt` rational check.
  3. Parse the result back with `double.parse`.

  Test the values above.
- Key order of a choice answer: `type, choice, probabilities, confidence, answer_confidence, action`.
  Take the exact orders for all types from the decode fixture, not from this note.
- The action entry is `{"act_probability": round4(softmax(act)[0])}`.

`projectJevStrict`: port `laya/serve.py:128-163`.

`CheckpointConfig.fromOnnxConfig` reads the `laya` section of the onnx-community `config.json`:
`max_len`, `head_max_len`, `temperature`, `temperature_by_options`. Clamp with `temperature_clamp`, and
fall back to `[0.5, 5.0]` when it is absent. Add a test asserting that each onnx-community config's
temperatures equal the upstream `rl_agent_config.json` for that checkpoint:
- Both files are in `.tooling/checkpoints/`.
- Add the three onnx `config.json` files to `gen_fixtures.dart`'s downloads; they are small.

If typed-decisions' values differ, upstream's `rl_agent_config.json` wins. Make `CheckpointConfig`
prefer it when present, and record the finding in research.md.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 10: Language analysis and routing

**Files:**
- Create: `lib/src/routing/lang.dart`, `lib/src/routing/router.dart`
- Test: `test/routing/lang_test.dart`, `test/routing/router_test.dart`

**Interfaces:**
- Produces:

```dart
/// laya.lang.analyse(text) result; toJson() matches the Python dict key-for-key.
class LanguageAnalysis {
  final String script;               // 'latin', 'unknown', ...
  final Map<String, double> scriptProfile;
  final String? language;
  final bool isEnglish;
  final bool languageUndecided;
  final double diacriticRate;
  final double nonLatinFraction;
  final Object? mixedSegment;
  Map<String, Object?> toJson();
}
LanguageAnalysis analyse(String text);

/// Resolves a checkpoint name or alias; throws ArgumentError like normalise_name.
String normaliseCheckpointName(String name);

/// The routing decision; toJson() is the response's `routing` block.
class RouteDecision {
  final String model; final String repo; final String reason;
  final LanguageAnalysis? detection; final String? workflow;
  Map<String, Object?> toJson();
}

/// Routes like laya Router._route.
class CheckpointRouter {
  CheckpointRouter({String defaultModel = 'english', bool autoTaskDetection = false});
  RouteDecision route(Object state, Map<String, Object?> questions,
      {String? model, String? task, String? lang, String? langGuess});
}
```

- [ ] **Step 1: Write the failing tests**

```dart
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/routing/lang.dart';
import 'package:laya_server/src/routing/router.dart';
import 'package:test/test.dart';

void main() {
  test('analyse equals laya.lang.analyse', () {
    final f = jsonDecode(File('test/fixtures/lang.json').readAsStringSync()) as Map;
    for (final c in (f['cases'] as List).cast<Map<String, Object?>>()) {
      expect(jsonEncode(analyse(c['text']! as String).toJson()), jsonEncode(c['analysis']),
          reason: jsonEncode(c['text']));
    }
  });

  test('route equals Router.route', () {
    final f = jsonDecode(File('test/fixtures/route.json').readAsStringSync()) as Map;
    for (final c in (f['cases'] as List).cast<Map<String, Object?>>()) {
      final args = c['args']! as Map<String, Object?>;
      final router = CheckpointRouter(autoTaskDetection: args['auto_task'] == true);
      final got = router.route(args['state']!, (args['questions']! as Map).cast(),
          model: args['model'] as String?, task: args['task'] as String?,
          lang: args['lang'] as String?, langGuess: args['lang_guess'] as String?);
      expect(jsonEncode(got.toJson()), jsonEncode(c['decision']), reason: jsonEncode(args));
    }
  });

  test('aliases', () {
    expect(normaliseCheckpointName(' ML '), 'multilingual');
    expect(normaliseCheckpointName('typed_decisions'), 'typed-decisions');
    expect(() => normaliseCheckpointName('jev-1'), throwsArgumentError);
  });
}
```

Floats in `analysis` (`diacritic_rate` and others) must match Python's exact values. The reference
computes them as Python floats (doubles). Port the arithmetic in the same order, so the results are
bit-identical.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/routing > .tooling/logs/t10.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

- Port `laya/lang.py` in full: 701 lines, including the stopword tables, script ranges, thresholds at
  `:355-371` and `analyse` at `:644`. Python `str` iterates code points, so iterate `text.runes`, never
  UTF-16 units. Python's `unicodedata` calls need equivalents. List each one the port needs:
  - category, which separates letters from marks;
  - decomposition, used for diacritics.

  Build exactly those tables in `tool/gen_nfc.dart`'s output, and do not invent heuristics.
- Port `laya/router.py`:
  - the `_ALIASES` and `DEFAULT_MODELS` block, `:47-118`;
  - the `_TYPED_DECISION_WORKFLOWS` block;
  - `_route`, `:869-960`, with each `reason` format string reproduced exactly. Python's `%r` of a str is
    `'value'`, with single quotes and Python escaping; write `pyRepr(String)` for it.
- `repo` per checkpoint is the `STANDALONE_MODELS` value for that name, `router.py:56-60`. Check which
  one the decision dict reports; the fixture decides.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 11: ONNX Runtime FFI bindings and native asset

**Files:**
- Create: `hook/build.dart`, `hook/ort_release.dart`, `tool/pin_ort.dart`,
  `lib/src/inference/ort/ort_api.dart`, `lib/src/inference/ort/ort_library.dart`,
  `lib/src/inference/ort/ort_session.dart`
- Create (generated by Step 1, committed): `test/fixtures/ort/echo.onnx`
- Test: `test/inference/ort_session_test.dart`

**Interfaces:**
- Produces:

```dart
/// Thrown for any failure reported by ONNX Runtime; [message] is ORT's text.
class OrtException implements Exception { OrtException(this.code, this.message); final int code; final String message; }

/// Loads the ORT shared library: [explicitPath] (LAYA_ORT_LIB) or the bundled native asset.
/// Throws [OrtException] with an actionable message when neither is available.
OrtRuntime loadOrtRuntime({String? explicitPath});

/// The loaded runtime; one per isolate.
class OrtRuntime {
  String get version;                          // e.g. "1.30.0"
  OrtSession openSession(String modelPath, {int? intraOpThreads});
}

/// A loaded graph. Not thread-safe across isolates; close() releases native memory.
class OrtSession {
  List<String> get inputNames;
  List<String> get outputNames;
  /// Runs with named inputs; returns float32 outputs by name with their shapes.
  Map<String, OrtTensorF32> run(Map<String, OrtInput> inputs);
  void close();
}
sealed class OrtInput {}
final class OrtInt64Input extends OrtInput { OrtInt64Input(this.data, this.shape); final Int64List data; final List<int> shape; }
final class OrtBoolInput extends OrtInput { OrtBoolInput(this.data, this.shape); final Uint8List data; final List<int> shape; }
class OrtTensorF32 { OrtTensorF32(this.data, this.shape); final Float32List data; final List<int> shape; }
```

- [ ] **Step 1: Create a tiny test model with the reference ORT tooling**

Write `tool/reference/echo_model.py` and run it from `gen_fixtures.dart`.
- It uses `onnx.helper` to build a graph with these inputs:
  - `ids`, int64 `[b,s]`;
  - `mask`, bool `[b,k]`.
- And these outputs:
  - `ids_f = Cast(ids, float)`;
  - `mask_f = Cast(mask, float)`.
- Opset 18. Save it as `test/fixtures/ort/echo.onnx`, which is a few hundred bytes.

- [ ] **Step 2: Write the failing test**

```dart
import 'dart:typed_data';

import 'package:laya_server/src/inference/ort/ort_library.dart';
import 'package:laya_server/src/inference/ort/ort_session.dart';
import 'package:test/test.dart';

void main() {
  late OrtRuntime ort;
  setUpAll(() => ort = loadOrtRuntime());

  test('version is the pinned release', () => expect(ort.version, '1.30.0'));

  test('runs int64 and bool inputs', () {
    final s = ort.openSession('test/fixtures/ort/echo.onnx', intraOpThreads: 1);
    addTearDown(s.close);
    expect(s.inputNames, ['ids', 'mask']);
    final out = s.run({
      'ids': OrtInt64Input(Int64List.fromList([1, 2, 3, 4, 5, 6]), [2, 3]),
      'mask': OrtBoolInput(Uint8List.fromList([1, 0, 0, 1]), [2, 2]),
    });
    expect(out['ids_f']!.shape, [2, 3]);
    expect(out['ids_f']!.data, [1, 2, 3, 4, 5, 6]);
    expect(out['mask_f']!.data, [1, 0, 0, 1]);
  });

  test('bad path raises OrtException with ORT text', () {
    expect(() => ort.openSession('missing.onnx'), throwsA(isA<OrtException>()));
  });

  test('explicit wrong library path explains how to fix it', () {
    expect(() => loadOrtRuntime(explicitPath: 'nope.dll'),
        throwsA(isA<OrtException>().having((e) => e.message, 'message', contains('LAYA_ORT_LIB'))));
  });
}
```

- [ ] **Step 3: Run the test and check it fails**

Run `dart test test/inference/ort_session_test.dart > .tooling/logs/t11.txt 2>&1; echo "exit=$?"` and
expect a nonzero exit.

- [ ] **Step 4: Write `tool/pin_ort.dart` and generate `hook/ort_release.dart`**

1. Query `https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v1.30.0`.
2. For these assets, take `browser_download_url` and `digest`:
   - `onnxruntime-win-x64-1.30.0.zip`;
   - `onnxruntime-win-arm64-1.30.0.zip`;
   - `onnxruntime-linux-x64-1.30.0.tgz`;
   - `onnxruntime-linux-aarch64-1.30.0.tgz`;
   - `onnxruntime-osx-arm64-1.30.0.tgz`;
   - `onnxruntime-osx-x86_64-1.30.0.tgz`.

   GitHub provides `sha256:...` in `digest`. If that field is absent, download each file and compute
   the hash with `package:crypto`.
3. Emit a `const` map `{(os, arch): (url, sha256, libPathInArchive)}`. Print it for review.

Also download the matching `include/onnxruntime_c_api.h` out of the win-x64 archive into
`.tooling/ort/` (not committed). It is the reference for Step 5.

- [ ] **Step 5: Write `ort_api.dart` against the v1.30.0 header**

- `OrtApiBase` is a struct of two function pointers:
  - `GetApi(uint32_t version)` returning `const OrtApi*`;
  - `GetVersionString()` returning `const char*`.
- `OrtApi` is a struct whose fields are function pointers in exactly the header's declaration order.
  - Declare every field from the first up to the last one we use. Unused ones are
    `external Pointer<Void>`.
  - Name used ones by their C name.
  - Request `ORT_API_VERSION` 23 or whatever the header defines, and check `GetApi` returns non-null.
- Functions we use:
  - `CreateStatus`, `GetErrorCode`, `GetErrorMessage`, `ReleaseStatus`;
  - `CreateEnv`, `ReleaseEnv`;
  - `CreateSessionOptions`, `SetIntraOpNumThreads`, `SetSessionGraphOptimizationLevel`,
    `ReleaseSessionOptions`;
  - `CreateSession`, `ReleaseSession`;
  - `SessionGetInputCount`, `SessionGetOutputCount`, `SessionGetInputName`, `SessionGetOutputName`;
  - `GetAllocatorWithDefaultOptions`, `AllocatorFree`;
  - `CreateCpuMemoryInfo`, `ReleaseMemoryInfo`;
  - `CreateTensorWithDataAsOrtValue`;
  - `Run`;
  - `GetTensorMutableData`, `GetTensorTypeAndShape`, `GetDimensionsCount`, `GetDimensions`,
    `ReleaseTensorTypeAndShapeInfo`, `ReleaseValue`.
- `ORTCHAR_T` is `wchar_t` (UTF-16) on Windows and `char` elsewhere. `CreateSession`'s path parameter
  therefore needs `toNativeUtf16()` on Windows and `toNativeUtf8()` elsewhere.
- Every call returning `OrtStatus*` goes through one `_check(Pointer status)` helper. It throws
  `OrtException` with ORT's message and releases the status.

Write a comment above the `OrtApi` struct stating the header version. Also add a test in
`ort_session_test.dart` that calls `GetVersionString` before `GetApi`, so a struct-order mistake shows
up as a version mismatch rather than a crash. Do this as the first test.

- [ ] **Step 6: Write `ort_library.dart` and `hook/build.dart`**

`loadOrtRuntime`:
1. If `explicitPath` (or `LAYA_ORT_LIB`) is set, `DynamicLibrary.open` it. On failure, throw
   `OrtException(-1, 'could not load ONNX Runtime from <path>: <os error>. Set LAYA_ORT_LIB to the
   onnxruntime shared library, version 1.30.x')`.
2. Otherwise resolve `OrtGetApiBase` through the native asset. Use
   `@Native<Pointer<OrtApiBaseStruct> Function()>(symbol: 'OrtGetApiBase', assetId:
   'package:laya_server/onnxruntime')`.

`hook/build.dart`, using `package:hooks` and `package:code_assets`:
1. If `input.config.buildCodeAssets` is false, return.
2. Find the target OS/arch in `ortRelease`.
3. Download to `input.outputDirectoryShared`, skipping when it is already there with a matching sha256.
4. Verify the sha256 and extract only the library file, using `package:archive`.
5. Emit `CodeAsset(package: 'laya_server', name: 'onnxruntime', linkMode: DynamicLoadingBundled(),
   file: <extracted lib>)`.
6. An unsupported target throws a `BuildError` naming the supported list and `LAYA_ORT_LIB`.

Check the exact API names of `hooks`/`code_assets` against the version `dart pub add` resolved. The
hooks API changed between 0.x releases; read the package's example in
`.dart_tool/../pub-cache/hosted/pub.dev/hooks-*/example/`.

- [ ] **Step 7: Write `ort_session.dart`**

- Session: `CreateEnv(ORT_LOGGING_LEVEL_WARNING, "laya")` once per `OrtRuntime`, plus
  `ORT_ENABLE_ALL` and the given thread count.
- `run`:
  1. Allocate native buffers with `calloc`.
  2. Copy the `Int64List`/`Uint8List` data in.
  3. `CreateTensorWithDataAsOrtValue` with element types `ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64` (7) and
     `_BOOL` (9).
  4. `Run`.
  5. Copy outputs into a `Float32List`, checking that the element type is FLOAT (1).
  6. Release everything in `finally`.
- `close` is idempotent. A `Finalizer` releases the session if `close` is never called.

- [ ] **Step 8: Run the tests and check they pass**

Run `dart test test/inference/ort_session_test.dart > .tooling/logs/t11.txt 2>&1; echo "exit=$?"` and
expect `exit=0`. This is the first step that downloads ORT (about 70–130 MB), through the build hook,
which was approved as tooling. The run must not touch any system directory.

- [ ] **Step 9: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 12: Batching and the inference backend (worker isolate)

**Files:**
- Create: `lib/src/inference/batch.dart`, `lib/src/inference/backend.dart`,
  `lib/src/inference/ort_backend.dart`
- Test: `test/inference/batch_test.dart`, `test/inference/ort_backend_test.dart`

**Interfaces:**
- Consumes: `SequenceRow` (Task 8), `QuestionType` (Task 7), `OrtRuntime`/`OrtSession` (Task 11).
- Produces:

```dart
/// Collated model inputs for N rows (laya common.collate_items).
class InferenceBatch {
  factory InferenceBatch.collate(List<SequenceRow> rows, List<QuestionType> types, {required int padId});
  final int rows; final int seqLen; final int markers;
  final Int64List inputIds; final Int64List attentionMask;
  final Int64List markerPos; final Uint8List markerMask; final Int64List qtype;
}

/// Per-row outputs.
class InferenceOutput {
  InferenceOutput(this.logits, this.actLogits, this.markers);
  final Float32List logits; final Float32List actLogits; final int markers;
  List<double> logitsFor(int row, int k);
  List<double> actFor(int row);
}

/// Runs a checkpoint's graph. Implementations must not block the caller's isolate.
abstract interface class InferenceBackend {
  Future<InferenceOutput> run(InferenceBatch batch);
  Future<void> close();
}

/// Deterministic stand-in for tests and examples: logits from a caller-supplied function.
class FakeBackend implements InferenceBackend {
  FakeBackend(this.logits, {this.delay = Duration.zero});
  final List<double> Function(int row, int markers) logits;
  final Duration delay;
}

/// ORT-backed implementation owning one session on a dedicated isolate.
class OrtBackend implements InferenceBackend {
  static Future<OrtBackend> open(String modelPath, {int? intraOpThreads, String? ortLibraryPath});
}
```

- [ ] **Step 1: Write the failing tests**

`batch_test.dart` uses two rows of different lengths, with 2 and 3 markers. It asserts:
- padding with `padId`;
- attention ones then zeros;
- marker positions padded with 0;
- mask padded with 0;
- `qtype` per row.

These are the exact arrays for the example. Compute them by hand in the test from
`laya/common.py:775-816`.

`ort_backend_test.dart` (tagged `ort`) opens `test/fixtures/ort/echo.onnx`. It cannot be a Laya graph,
so add an `OrtBackend.openRaw` test hook that returns raw outputs. Through the isolate, it checks:
- a round-trip works;
- concurrent `run` calls are serialised and each gets its own result;
- `close()` shuts the isolate down;
- a later `run` throws `StateError('backend closed')`.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/inference > .tooling/logs/t12.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

- `OrtBackend` spawns an isolate with `Isolate.spawn`. The worker loads the runtime and opens the
  session.
- Requests carry `TransferableTypedData` for the five tensors and get a reply port.
- The worker processes one request at a time, which mirrors `laya-serve`'s single inference worker.
- Errors are sent back as `(code, message)` and rethrown as `OrtException`.
- The input feed names are fixed by the onnx-community graph: `input_ids`, `attention_mask`,
  `marker_pos`, `marker_mask`, `qtype`. Outputs are `logits` and `act_logits`.
- On open, the worker checks that the session's input names equal this set. If not, it fails with a
  message naming the expected and actual names. This catches a split `encoder.onnx` or another export
  being pointed at by mistake.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 13: Agent and engine — predict end to end with a fake backend

**Files:**
- Create: `lib/src/model/agent.dart`, `lib/src/engine.dart`
- Modify: `lib/laya_server.dart` (export the public API)
- Test: `test/model/agent_test.dart`, `test/engine_test.dart`

**Interfaces:**
- Consumes: Tasks 7–12.
- Produces:

```dart
/// Options a request may set (laya Router.predict body controls).
class PredictOptions {
  const PredictOptions({this.model, this.task, this.lang, this.langGuess,
      this.maxLen, this.headMaxLen, this.minConfidence});
  final String? model, task, lang, langGuess; final int? maxLen, headMaxLen; final double? minConfidence;
}

/// One loaded checkpoint: tokenizer + config + backend.
class LayaAgent {
  LayaAgent({required this.spec, required this.tokenizer, required this.config, required this.backend});
  /// Full laya result: {model, answers, usage} (routing added by the engine).
  Future<Map<String, Object?>> predict(Object state, Map<String, Question> questions,
      {int? maxLen, int? headMaxLen, double? minConfidence});
}

/// Loads checkpoints on demand (LRU, maxLoaded), routes, and predicts.
class LayaEngine {
  LayaEngine({required Future<LayaAgent> Function(String checkpoint) loader,
      int maxLoaded = 2, String defaultModel = 'english', bool autoTaskDetection = false});
  /// Builds the loader for real files under [modelDir] (see Task 14 layout).
  static Future<LayaAgent> Function(String) fileLoader(String modelDir,
      {String variant = 'fp32', int? threads, String? ortLibraryPath});
  Future<void> preload(Iterable<String> names);
  List<String> get loaded;
  Future<Map<String, Object?>> predict(Object state, Map<String, Object?> questions, PredictOptions options);
  Future<void> close();
}
```

- [ ] **Step 1: Write the failing tests**

`agent_test.dart` uses the real English tokenizer (from the fixture copy), a fixed config, and a
`FakeBackend` returning logits `[0, 1, 2, …]` per row. It asserts:
- `answers` has one entry per question, in request order;
- the choice is the last option, because its logit is highest;
- `usage.input_tokens` equals the sum of row lengths;
- `usage.output_tokens == 0`;
- `state_tokens` equals the state's token count;
- `truncated` is false for a short state;
- a 3,000-word state gives `truncated == true`, `truncated_questions` listing every question, and
  `state_tokens_dropped > 0`;
- a question whose options collapse produces `usage.options[qid]` with `total`, `distinct` and
  `tokens_per_option`.

Port the usage assembly from `laya/onnx_agent.py:704-790` and `agent.ts:600,689-697`. The exact key set
and order are `input_tokens`, `output_tokens`, `state_tokens`, `state_tokens_dropped`, `truncated`,
`truncated_questions`, and optionally `options`. Assert them with `jsonEncode`.

`engine_test.dart` uses a loader that counts loads and returns fake agents. It asserts:
- routing picks `multilingual` for a Hindi state and `english` for English;
- `routing` is attached with `reason`;
- `maxLoaded: 2` evicts the least recently used checkpoint on the third distinct checkpoint, and
  `close()` is called on the evicted backend;
- an explicit `model: 'jev-latest'` auto-routes;
- `model: 'ml'` forces multilingual;
- an unknown `task` gives `ApiException` 422, with laya's message from `normalise_name`;
- concurrent `predict` calls on one checkpoint never load it twice.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/model/agent_test.dart test/engine_test.dart > .tooling/logs/t13.txt 2>&1;
echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

`LayaAgent.predict`:
1. `encodeState` once.
2. For each question, `buildSequence`. Use `truncateLeft` when the state is a `List`, and resolve
   `maxLen`/`headMaxLen` from the overrides or the config.
3. Collate one batch.
4. Run `backend.run`.
5. For each row, `decodeAnswer`.
6. Assemble `{model: 'laya-rl-agent', answers, usage}`.

Laya checks that the marker count equals the option count (`onnx_agent.py:603-641`). Port that check
and its 422 message, raised when options do not fit `head_max_len`.

`LayaEngine.predict`:
1. Parse the questions with `Question.parse`.
2. Route with `CheckpointRouter`. `_resolve_model` semantics come from research.md §1.2: an unknown
   model name means auto.
3. Get or load the agent.
4. Predict, then attach `routing`.

Loads are memoised per name with a `Future` map so concurrent calls share one load.

`fileLoader` reads `<modelDir>/<checkpoint>/`:
- `tokenizer.json`;
- `config.json`;
- `onnx/model.onnx` (or `model_fp16.onnx`), plus its `_data` file.

It builds `OrtBackend.open`. A missing file gives
`StateError('checkpoint <name> is not installed in <dir>; run: dart run laya_server:fetch <name>')`.

Export the public API from `lib/laya_server.dart`:
- `LayaEngine`, `LayaAgent`, `PredictOptions`;
- `LayaTokenizer`;
- `InferenceBackend`, `FakeBackend`, `OrtBackend`;
- `CheckpointSpec`;
- `ApiException`;
- `createHandler` and `ServerConfig` (Task 15);
- `ModelStore` (Task 14).

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 14: Model store and the `fetch` CLI

**Files:**
- Create: `lib/src/store/model_store.dart`, `bin/fetch.dart`
- Test: `test/store/model_store_test.dart`

**Interfaces:**
- Produces:

```dart
/// A pinned file of a checkpoint on the Hugging Face hub.
class PinnedFile { const PinnedFile(this.repo, this.revision, this.path, this.size, this.sha256); ... }

/// Installs and verifies checkpoint files under [root]/<checkpoint>/.
class ModelStore {
  ModelStore(this.root, {http.Client? client, Uri? hubBase});
  /// Files a checkpoint needs for [variant] ('fp32' | 'fp16').
  List<PinnedFile> filesFor(String checkpoint, {String variant = 'fp32'});
  /// Downloads missing files (resuming partial ones) and verifies sha256; reports progress.
  Future<void> install(String checkpoint, {String variant = 'fp32', void Function(String path, int done, int total)? onProgress});
  /// True when every file is present with the pinned size (fast check used at startup).
  bool isInstalled(String checkpoint, {String variant = 'fp32'});
  /// Full sha256 verification of installed files; returns the mismatching paths.
  Future<List<String>> verify(String checkpoint, {String variant = 'fp32'});
}
```

- [ ] **Step 1: Pin file sizes and hashes**

Get them from `https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=true`, where LFS files
carry `lfs.oid` (the sha256) and `lfs.size`. Write a `tool/pin_models.dart` that prints the Dart
constants for the three onnx-community repos at their pinned revisions.

Files per checkpoint:
- `tokenizer.json`;
- `config.json`;
- `onnx/model.onnx` and `onnx/model.onnx_data`, or `onnx/model_fp16.onnx` and
  `onnx/model_fp16.onnx_data`.

Non-LFS small files have no `oid` in that sense. Compute their sha256 by downloading them in the tool.

- [ ] **Step 2: Write the failing tests**

The tests use a local `HttpServer` serving fake files, with `hubBase` pointed at it and tiny pinned
files built in the test. Assert:
- `install` downloads into `<root>/<checkpoint>/...` and verifies;
- a partial file is resumed with an HTTP `Range` request, and the server records it;
- a corrupted file makes `install` throw `StateError` naming the path, and leaves no file at the final
  path (write to `.part`, verify, then rename);
- `isInstalled` is false until install completes;
- `verify` reports a byte-flipped file.

- [ ] **Step 3: Run the tests and check they fail**

Run `dart test test/store > .tooling/logs/t14.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 4: Implement `ModelStore` and `bin/fetch.dart`**

`bin/fetch.dart` uses `package:args`:

```
dart run laya_server:fetch [--dir models] [--variant fp32|fp16] <english|multilingual|typed-decisions|all>...
```

- It prints the total size before starting.
- It shows progress on one line.
- Exit code 0 means success; 1 means failure, with one sentence to stderr.

Do not run `fetch` against the real hub in this task. Real downloads are Task 17's gate.

- [ ] **Step 5: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 6: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 15: HTTP server — config, handler, limits, errors

**Files:**
- Create: `lib/src/server/config.dart`, `lib/src/server/limits.dart`, `lib/src/server/handler.dart`,
  `bin/laya_server.dart`
- Test: `test/server/handler_test.dart`, `test/server/limits_test.dart`, `test/server/config_test.dart`,
  `test/server/client_compat_test.dart`

**Interfaces:**
- Consumes: `LayaEngine`, `PredictOptions`, `ApiException` and `projectJevStrict`.
- Produces:

```dart
/// Server settings from LAYA_* environment variables (laya-serve names and defaults).
class ServerConfig {
  factory ServerConfig.fromEnvironment(Map<String, String> env);
  final String host; final int port; final String rootPath; final String modelDir;
  final List<String> preload; final bool preloadEnabled; final int? threads;
  final bool autoTask; final String defaultModel; final int maxLoaded; final String? apiKey;
  final int maxConcurrent; final int maxTokenBudget; final bool jevStrict;
  final String variant; final String? ortLibraryPath;
}

/// The shelf handler serving GET /health and POST /v1/systemone.
Handler createHandler(LayaEngine engine, ServerConfig config, {Logger? logger});
```

- [ ] **Step 1: Write the failing tests**

`handler_test.dart` builds the handler over an engine with fake agents (`FakeBackend`). It drives
requests through `handler(Request(...))` directly, without sockets. The example bodies come from
research.md §1.2, from `docs/http-api.md:95-105` in upstream, and from `laya-railway`'s README.

Each case below asserts status, the full JSON body and the relevant headers.

**Auth**
1. `GET /health` without a key configured returns the full payload, with `loaded`, `revisions`
   (checkpoint → onnx revision), `device: "cpu"`, `device_is_preference`, `checkpoint_devices` and
   `cpu_fallbacks`, all keyed consistently.
2. With `LAYA_API_KEY=k`:
   - `GET /health` without a bearer returns `{"status":"ok"}` only;
   - `POST` without a bearer returns 401 `{"detail":"invalid or missing bearer token"}`;
   - with `Bearer k` it returns 200;
   - a header with a non-ASCII byte (`Bearer s\xe9cret`, built from latin-1 bytes) returns 401, never
     500.

**400s**
3. `{"detail":"request body must be valid JSON"}`.
4. `{"detail":"request body must be an object with a 'questions' field"}`, for an array body and for a
   missing `questions`.
5. `{"detail":"'state' is required"}`, for a missing state and for `"state": null`.
6. `{"detail":"'questions' must be an object"}`.
7. The lone-surrogate detail, for the body `{"state":"\udc00","questions":{}}`. Dart's `jsonDecode`
   yields a lone UTF-16 surrogate: detect code units 0xD800–0xDFFF that are not part of a valid pair.

**413s, each with the exact text from `laya/serve.py`**
8. A body over 2 MiB, sent both with a truthful `Content-Length` and as a chunked stream with no
   length: `request body too large`.
9. 65 questions; 101 choice options; 33 score levels; 513 total options.
10. A 50,001-character string state.
11. A JSON state whose serialized form exceeds 50,000 characters because of escaping
    (`{"body": "\"" * 30000}`), giving `state too large (60014 > 50000 chars)`. Compute the expected
    number with `pythonJsonDumps`.

**422s**
12. `max_len: 0`, `max_len: "5"`, `max_len: true`, and `max_len: 9000` over the default budget 8192
    (`max_len exceeds server limit (9000 > 8192)`).
13. `lang: true`.
14. `min_confidence: 2`, with the message from `laya/confidence.py` `check_min_confidence`. Port that
    validator into `limits.dart`.
15. `hooks: []` and `on_predict_start: "x"`, with the refusal message from `serve.py:263-268`. The keys
    are listed sorted.
16. A score `null` level, with the `serve.py:579-584` message.
17. Unknown `task: "nope"`.

**200s**
18. The `queue`/`urgency` example. The response contains `model`, `answers` (choice and score key
    orders), `usage` and `routing`. It carries the `Server-Timing: inference;dur=<x.xx>` and
    `X-Inference-Time-Ms` headers, matching `^\d+\.\d{2}$`.
19. `LAYA_JEV_STRICT=1`: the same request gives the strict projection, with no `routing`.

**Failures and load**
20. A backend that throws gives 500 `{"detail":"inference failed"}`. The exception is logged at
    `SEVERE` with its stack trace; capture it with a `Logger` listener.
21. `LAYA_MAX_CONCURRENT=1`:
    - a slow `FakeBackend` (`delay: 300ms`) holds the slot;
    - a second concurrent POST gets 503 `server busy, try again later` with `Retry-After: 1`;
    - `GET /health` during that time answers in under 50 ms (Review Focus 4).
22. `POST /v1/systemone/batch` and `GET /v1/models` give 404 `{"detail":"Not Found"}`, which is
    FastAPI's text.
23. `LAYA_ROOT_PATH=/laya` with a request to `/laya/v1/systemone` gives 404. The root path is only for
    URL generation; the proxy strips it. Requests are still served at `/v1/systemone`.

`limits_test.dart` unit-tests the lone-surrogate walk, covering keys too, and the state-length lower
bound.

`config_test.dart` covers:
- defaults;
- invalid `LAYA_PORT` throws `ArgumentError('invalid LAYA_PORT ...')`;
- invalid `LAYA_MAX_CONCURRENT`/`LAYA_MAX_LOADED`/`LAYA_MAX_TOKEN_BUDGET` fall back to defaults, as
  upstream does;
- an unknown `LAYA_DEFAULT_MODEL` throws `ArgumentError('invalid LAYA_DEFAULT_MODEL ...')`.

`client_compat_test.dart` (tagged `compat`):
1. Add `jev_dart` as a dev dependency.
2. Serve the handler with `shelf_io.serve` on port 0, backed by the fake engine.
3. Point `jev_dart`'s client at it with base URL and key.
4. Send the same request through the client's typed API, and assert the decoded choice/score/noul
   answers. Read `jev_dart`'s README for its constructor and request types.
5. Run it twice, in strict mode and in full mode. Both must decode, because Jev clients must ignore
   extra fields. If `jev_dart` rejects the full payload, that is exactly what `LAYA_JEV_STRICT` exists
   for. Record the finding in the README.

- [ ] **Step 2: Run the tests and check they fail**

Run `dart test test/server > .tooling/logs/t15.txt 2>&1; echo "exit=$?"` and expect a nonzero exit.

- [ ] **Step 3: Implement**

**Routing**
- A `shelf` handler with manual routing on `(method, path)`. No `shelf_router` dependency is needed for
  two routes.
- An unknown path gives 404 `{"detail":"Not Found"}`. A known path with the wrong method gives 405
  `{"detail":"Method Not Allowed"}`, as FastAPI does.

**Admission**
- A counter checked before the body is read. When full, return 503 with `Retry-After: 1`.
- Admission is released in `finally`.

**Inference gate**
- The backend isolate already serialises per checkpoint.
- Across checkpoints, `laya-serve` also serialises globally, with a single worker. Mirror that with one
  async mutex around `engine.predict`, so CPU threads are not oversubscribed. A comment states this
  choice.

**Body reading**
- Read `request.read()` chunk by chunk, counting bytes. Abort with 413 once past 2 MiB.
- Check `Content-Length` first.
- Decode as UTF-8 with `allowMalformed: false`. Malformed UTF-8 gives the invalid-JSON 400.

**Order of checks** (exactly `serve.py:876-916`)
1. Parse.
2. Object and `questions` check.
3. `_check_request_limits`.
4. Refusals.
5. Lone surrogates.
6. Model, budget, task, lang and `min_confidence` validation.

Then predict.

**Errors**
- `ApiException` becomes its status and detail.
- Any other exception becomes 500 `inference failed`, logged with the model name sanitised of CR/LF.

**Responses**
- `jsonEncode` of the result.
- `content-type: application/json`.
- The timing headers measure only the `engine.predict` time, formatted with `toStringAsFixed(2)`.

`bin/laya_server.dart`:
1. Build `ServerConfig.fromEnvironment(Platform.environment)`. A config error prints the message to
   stderr and exits 2.
2. Construct `LayaEngine` with `fileLoader`.
3. If preload is enabled, preload the checkpoints. A missing checkpoint prints the fetch command and
   exits 2 (Review Focus 5).
4. `shelf_io.serve` with `shared: false`.
5. Disable Nagle with `server.autoCompress = false` and `HttpServer`'s default TCP_NODELAY. Check that
   Dart sets TCP_NODELAY by default, and note it in a comment.
6. Handle `SIGINT`/`SIGTERM` by closing the server and `engine.close()`.

- [ ] **Step 4: Run the tests and check they pass**

Run the same command; expect `exit=0`.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 16: Documentation, example, publish readiness

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `lib/laya_server.dart` (library docs)
- Create: `example/main.dart`, `doc/api.md` (the full wire protocol, moved from research.md §1.2 and
  edited for users)

**Interfaces:**
- Consumes: the public API from Task 13.

- [ ] **Step 1: Write the README**

The README covers:
- what this is;
- an "Independent port, not affiliated with Convai Innovations or TypeSafe" line;
- quick start: `dart pub global activate laya_server`, then `fetch english`, then `laya_server`, then a
  curl example;
- a configuration table of every env var;
- the API summary, linking to `doc/api.md`;
- differences from `laya-serve`:
  - batch and models routes are not in v1;
  - `device` is always `cpu`;
  - fp16 is opt-in;
- memory needs per checkpoint (fp32 sizes);
- licence and credits;
- how to regenerate fixtures.

- [ ] **Step 2: Write `example/main.dart`**

The example shows the library use: build a `LayaEngine` with `fileLoader('models')` and call `predict`
on the queue/urgency example. When the models are absent, it prints the fetch hint and exits 0, so pub's
example analysis passes.

- [ ] **Step 3: Check publish readiness**

1. Run `dart pub publish --dry-run > .tooling/logs/publish.txt 2>&1; echo "exit=$?"`. Expect `exit=0`
   with no warnings. The package must not include `.tooling/`, `test/fixtures/tokenizers/` beyond the
   small files, or the ORT archives; use `.pubignore` if needed.
2. Run `dart pub global run pana --no-warning . > .tooling/logs/pana.txt 2>&1; echo "exit=$?"`, after
   `dart pub global activate pana`. Read the score and fix every lost point that is fixable.

Ask the user for the repository URL. **Do not publish.**

- [ ] **Step 4: Run the gate**

Run `dart run tool/check.dart` and expect all zeros.

---

### Task 17: End-to-end parity against `laya-serve` (gated on downloads)

**Files:**
- Create: `tool/reference/serve_parity.py`, `tool/parity.dart`, `test/parity/parity_test.dart` (tagged
  `model`)
- Create (generated, committed): `test/fixtures/parity_<checkpoint>.json`

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: Ask the user before downloading**

Ask in one question, listing exact sizes:
- the ONNX fp32 models: English 1.69 GB, multilingual 1.29 GB, typed-decisions 1.69 GB;
- the upstream torch weights for the Python reference: English 0.84 GB, multilingual 0.64 GB (already
  approved), typed-decisions 0.84 GB (needs asking).

Offer the options "all three", "English and multilingual only" and "not now". Proceed only with what is
approved. Skipped checkpoints keep their tests skipped, and the skip is reported.

- [ ] **Step 2: Generate the parity references**

`serve_parity.py` runs `laya.serve.create_app()` in-process with FastAPI's `TestClient`, pinned to the
checkpoint by `model`. It POSTs every request in `cases.json` and records the status and JSON body.

Run it through `tool/parity.dart`, with `HF_HUB_OFFLINE=0` only for this step.

- [ ] **Step 3: Write the parity test**

The test starts the real Dart handler with `fileLoader` over the fetched models and posts each request.
It asserts:
- the status is equal;
- the key sets are equal at every level;
- strings are equal;
- each probability, score and confidence is within `2e-4` of the reference. ONNX vs torch fp32 drift is
  below 1e-4 before 4-decimal rounding, per the onnx-community card;
- `choice` is equal except where the reference's top two probabilities are within `2e-4` (a reported
  near-tie);
- `usage` and `routing` are exactly equal.

- [ ] **Step 4: Run it and record the results**

Run `dart test --tags model test/parity > .tooling/logs/parity.txt 2>&1; echo "exit=$?"`. Record the
max |Δp| per checkpoint and the latency per request (median of 20, with `LAYA_THREADS` at the physical
core count) in `doc/api.md` and in research.md. Say plainly which checkpoints were not tested.

- [ ] **Step 5: Run the gate**

Run `dart run tool/check.dart`, which excludes the `model` tag by default (`dart_test.yaml`:
`tags: {model: {skip: ...}, compat: {}}`). Expect all zeros.

---

## Self-review notes

**Coverage against the brief**
- Shelf server and typed models: Tasks 7, 9, 13 and 15.
- Jev-compatible endpoints: Task 15. The batch and models routes are excluded by the v1 decision.
- Tokeniser with reference-id tests: Tasks 2 and 4–6.
- API-compatibility tests against `laya-serve` and Jev examples: Tasks 15 and 17.
- Dart scripts: every `tool/*.dart`. Python only wraps the reference implementation.
- Gates written to files with exit codes: `tool/check.dart` and every step.

**Fixtures are authoritative over the notes in this plan.** Where a note here (a key order, a dtype) and
a fixture disagree, the fixture is right. Fix the code, and correct this plan.

**Uncertainties that tasks must resolve, not assume**
- The float32-vs-float64 softmax in `_decode_answers` (Task 9).
- The Oniguruma-vs-JavaScript `\s` difference (Task 5).
- The Unicode version for NFC (Task 5).
- The `hooks` API names (Task 11).
- The `OrtApi` field order: checked against the header and by the version test (Task 11).
- The `jev_dart` API (Task 15).
- Typed-decisions temperatures in the onnx-community config (Task 9).
