# laya_server HTTP API

`laya_server` serves the same HTTP API as `laya-serve` (Laya's FastAPI server),
which is compatible with the TypeSafe System One ("Jev") contract. Two routes
are served:

| Method and path | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | open | Liveness, and what is loaded |
| `POST /v1/systemone` | bearer, if `LAYA_API_KEY` is set | One state, many questions |

`POST /v1/systemone/batch` and `GET /v1/models` are **not** in v1; they answer
404. Every other path is 404 `Not Found`, another method on a known path is 405
`Method Not Allowed` (with an `Allow` header), and a known path with its
trailing slash added or removed is a 307 redirect.

Error bodies are always `{"detail": "<message>"}`.

## Authentication

When `LAYA_API_KEY` is set, `POST /v1/systemone` needs the header
`Authorization: Bearer <key>`. The comparison is constant-time. A wrong or
missing token is 401 `invalid or missing bearer token`.

`GET /health` is never refused. Without the bearer it returns only
`{"status":"ok"}`; with it (or when no key is set) it returns the full payload
below.

## `GET /health`

```json
{
  "status": "ok",
  "loaded": ["english"],
  "revisions": {"english": "42e2a6e3b3708c8ff5f61f4aa75c314b1449199b"},
  "device": "cpu",
  "device_is_preference": false,
  "checkpoint_devices": {"english": "cpu"},
  "cpu_fallbacks": {"english": {"count": 0, "last_reason": null}}
}
```

`loaded` lists the resident checkpoints, least recently used first. `revisions`
is the pinned revision of each one's ONNX export. `device` is always `cpu`, and
`device_is_preference` is true only while nothing is loaded.

## `POST /v1/systemone`

### Request

A JSON object. The body may be UTF-8 (with or without a BOM), UTF-16 or UTF-32,
as Python's `json.loads` accepts.

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | string, object or array | What to classify. Required, not null. An object or array is serialized as JSON text |
| `questions` | object | Question id to question definition. Required |
| `model` | string | Optional. A checkpoint name, alias or public id picks that checkpoint; anything else (`jev-1`, `jev-latest`, `convaiinnovations/laya`) auto-routes |
| `task` | string | Optional checkpoint, by name or alias. An unknown one is 422 |
| `lang`, `lang_guess` | string or null | Optional language code (`"de"`) for routing |
| `max_len`, `head_max_len` | positive integer | Optional token budgets, each capped by `LAYA_MAX_TOKEN_BUDGET` |
| `min_confidence` | number or object | Optional abstention threshold; see below |

Checkpoint names are `english`, `multilingual` and `typed-decisions`. Aliases
such as `en`, `ml`, `multi` and `typed` are accepted, as are the public ids
`convaiinnovations/laya-multilingual` and
`convaiinnovations/laya-typed-decisions`.

The hook controls `hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`
and `hooks_timeout` run inside the server process, so they are refused with 422.

#### Questions

Every question has a `type`, `instructions` (the text the model should answer)
and `criteria`:

```json
{
  "queue": {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments", "tech": "bugs", "other": "anything else"}
  },
  "urgency": {
    "type": "score",
    "instructions": "How upset is the customer?",
    "criteria": ["calm", "firm", "angry", "furious"]
  }
}
```

- `choice`: `criteria` is an object of label to description, or a list of labels.
- `score`: `criteria` is a list of level descriptions, index 0 first.
- `noul`: a yes/no question; `criteria` is an object keyed `true` and `false`,
  optionally with `labels`.

#### `min_confidence`

A float in [0.0, 1.0] applies to every answer. An object maps bucket names such
as `"choice:3-5"` to floats, one threshold per option-count bucket. An answer
below its threshold is flagged as abstaining; see the response.

### Request limits

| Limit | Value | Status |
| --- | --- | --- |
| Body size | 2 MiB, enforced while streaming | 413 |
| Serialized state | 50,000 characters | 413 |
| Questions | 64 | 413 |
| Options per choice question | 100 | 413 |
| Levels per score question | 32 | 413 |
| Options across all questions | 512 | 413 |
| `max_len`, `head_max_len` | `LAYA_MAX_TOKEN_BUDGET` (default 8192) | 422 |
| JSON nesting | 2993 levels | 400 |
| Requests in flight | `LAYA_MAX_CONCURRENT` (default 16) | 503 |

There is no body-read timeout, as in `laya-serve`: a slow client holds one of the
concurrent slots until it finishes or disconnects. When the server is exposed to
untrusted clients, put a reverse proxy with timeouts in front of it.

### Response

200 with `Content-Type: application/json` and the headers
`Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms: <ms>`.

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "queue": {
      "type": "choice", "choice": "billing",
      "probabilities": {"billing": 0.9519, "tech": 0.0327, "other": 0.0154},
      "confidence": 0.797, "answer_confidence": 0.9519,
      "action": {"act_probability": 1.0}
    },
    "urgency": {
      "type": "score", "score": 1.6994,
      "legend": {"0": "calm", "1": "firm", "2": "angry", "3": "furious"},
      "probabilities": {"0": 0.0249, "1": 0.4136, "2": 0.3985, "3": 0.1629},
      "confidence": 0.1925, "answer_confidence": 0.4136,
      "action": {"act_probability": 1.0}
    }
  },
  "usage": {
    "input_tokens": 83, "output_tokens": 0, "state_tokens": 12,
    "state_tokens_dropped": 0, "truncated": false, "truncated_questions": []
  },
  "routing": {
    "model": "english", "repo": "convaiinnovations/laya",
    "reason": "English Latin text",
    "detection": {
      "script": "latin", "script_profile": {"latin": 1.0}, "language": "en",
      "is_english": true, "language_undecided": false, "diacritic_rate": 0.0,
      "non_latin_fraction": 0.0, "mixed_segment": null
    },
    "workflow": null
  }
}
```

- A `noul` answer carries `noul`, `confidence`, `answer_confidence` and `action`.
- `usage.options` appears only when options collapsed under the head token
  budget.
- `abstention`, `abstention_threshold` and `low_confidence` appear on every
  answer, but only when `min_confidence` was sent.
- `routing.model` is the checkpoint that answered.

#### Strict Jev responses

With `LAYA_JEV_STRICT=1` the response is projected onto the strict Jev contract:
no `routing`; a choice keeps `type`, `choice`, `confidence` and `probabilities`;
a score keeps `type`, `score`, `confidence`, `probabilities` and `legend`; a
noul keeps `type` and `noul`; `usage` keeps `input_tokens` and `output_tokens`.

### Errors

| Status | When |
| --- | --- |
| 400 | The body is not JSON (the `NaN` and `Infinity` literals included), not an object, or has no `questions`; `state` is missing or null; `questions` is not an object; the body holds an unpaired `\udXXX` surrogate escape; JSON nested deeper than 2993 levels |
| 401 | The bearer token is wrong or missing |
| 413 | A request limit above was exceeded |
| 422 | An unknown question type or a malformed question; options exceed the head token budget; a null score level; an unknown `task`; a bad `lang`, `lang_guess`, `min_confidence`, `max_len` or `head_max_len`; a hook control was sent |
| 500 | Anything else, with `inference failed` and nothing leaked; the cause is logged. A NaN or infinity in a result is also a 500 |
| 503 | `LAYA_MAX_CONCURRENT` requests are already in flight. Comes with `Retry-After: 1`. Load is refused, not queued |

### Example

```sh
curl -s http://localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "I was charged twice this month and I am furious.",
    "questions": {
      "queue": {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "payments", "tech": "bugs", "other": "else"}}
    }
  }'
```

## Concurrency

Admission is checked before the request body is read, so an overloaded server
refuses with 503 without buffering. Admitted requests run one inference at a
time, in the order they finished reading their bodies.

## `LAYA_ROOT_PATH`

Behind a reverse proxy mounted at a prefix, set `LAYA_ROOT_PATH=/laya`. A
request path carrying the prefix is served with the prefix removed
(`/laya/v1/systemone`), and one without it is served as is (`/v1/systemone`).
Redirects keep the prefix.
