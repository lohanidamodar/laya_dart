"""Reference HTTP behaviour of `laya-serve` (`laya.serve.create_app`) on crafted requests.

The app is the real FastAPI app `create_app` builds, driven through Starlette's `TestClient`, so
every status, `{"detail": ...}` text, check order and header below comes from `serve.py` itself.
Only the checkpoint is fake: the app gets a real `laya.router.Router` whose `load` returns a stub
agent instead of building torch weights. `Router.predict` therefore still runs for real --
`check_min_confidence`, routing (and its unknown-`task` ValueError) and `apply_confidence_gate` --
and the stub's `system_one` returns a fixed answer set in place of a forward pass.

For each case the fixture records the request (method, path, headers, body), the response status,
its JSON body (or null for an empty body), the `response_headers` a client can observe beyond the
body, and the arguments `Router.predict` received when the request reached it. A body is given as
text, as base64 when it is not UTF-8, or as a list of `[text, repeat]` parts for the large ones.

`depth` records how deep a JSON body `json.loads` accepts inside the app: the deepest nesting that
still parses, found by bisection, and the response one level deeper.
"""
import base64
import json
import os

from fastapi.testclient import TestClient

from laya import serve
from laya.router import Router

from _shared import emit

BASE_ENV = {"LAYA_DEVICE": "cpu", "LAYA_PRELOAD": "0"}
ENV_KEYS = ("LAYA_API_KEY", "LAYA_JEV_STRICT", "LAYA_MAX_TOKEN_BUDGET", "LAYA_ROOT_PATH",
            "LAYA_MAX_CONCURRENT")

CANNED_ANSWER = {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9,
                 "action": {"act_probability": 1.0}}


class StubAgent:
    device = "cpu"
    revision = "stub-revision"
    cpu_fallback_count = 0
    last_fallback_reason = None

    def system_one(self, state, questions, lang=None, **overrides):
        return {"model": "laya-rl-agent",
                "answers": {qid: dict(CANNED_ANSWER) for qid in questions},
                "usage": {"input_tokens": 7, "output_tokens": 0}}


class StubRouter(Router):
    """A real Router whose checkpoints are stub agents: routing and gating stay laya's."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def load(self, name):
        with self._lock:
            if name not in self._agents:
                self._agents[name] = StubAgent()
                self._order.append(name)
            return self._agents[name]

    def predict(self, state, questions, model=None, **kwargs):
        self.calls.append({"model": model, "kwargs": kwargs})
        return super().predict(state, questions, model=model, **kwargs)


QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Payments and invoices", "tech": "Bugs and outages",
                           "other": "Anything else"}},
    "urgency": {"type": "score", "instructions": "How upset is the customer?",
                "criteria": ["calm", "firm", "angry", "furious"]},
}
STATE = "I was charged twice for my March invoice and nobody answers my emails."


def body(obj):
    return {"text": json.dumps(obj, ensure_ascii=False)}


def ok_body(**extra):
    return body({"state": STATE, "questions": QUESTIONS, **extra})


def choice(n, as_list=False):
    labels = ["o%d" % i for i in range(n)]
    crit = labels if as_list else {label: label for label in labels}
    return {"type": "choice", "instructions": "Pick one.", "criteria": crit}


def score(n, null_at=None):
    levels = ["level %d" % i for i in range(n)]
    if null_at is not None:
        levels[null_at] = None
    return {"type": "score", "instructions": "Rate it.", "criteria": levels}


NOUL = {"type": "noul", "instructions": "Is it urgent?"}


def cases():
    out = []

    def add(name, env=None, method="POST", path="/v1/systemone", headers=None, data=None,
            chunked=False):
        out.append({"name": name, "env": env or {}, "method": method, "path": path,
                    "headers": headers or {}, "body": data, "chunked": chunked})

    key = {"LAYA_API_KEY": "k"}
    # --- routing and methods
    add("health", method="GET", path="/health")
    add("health head", method="HEAD", path="/health")
    add("health post", path="/health", data=ok_body())
    add("health put", method="PUT", path="/health")
    add("systemone get", method="GET")
    add("systemone delete", method="DELETE")
    add("health trailing slash", method="GET", path="/health/")
    add("systemone trailing slash", path="/v1/systemone/", data=ok_body())
    add("percent-encoded path", method="GET", path="/%68ealth")
    add("query string", path="/v1/systemone?x=1", data=ok_body())
    add("models", method="GET", path="/v1/models")
    add("unknown path", method="GET", path="/nope")
    add("root", method="GET", path="/")
    add("double slash", method="GET", path="//health")
    add("batch", path="/v1/systemone/batch", data=body({"states": [STATE], "questions": QUESTIONS}))
    # --- auth
    add("auth health no bearer", key, "GET", "/health")
    add("auth health bearer", key, "GET", "/health", {"authorization": "Bearer k"})
    add("auth health wrong bearer", key, "GET", "/health", {"authorization": "Bearer x"})
    add("auth post no bearer", key, data=ok_body())
    add("auth post bearer", key, headers={"authorization": "Bearer k"}, data=ok_body())
    add("auth post lowercase scheme", key, headers={"authorization": "bearer k"}, data=ok_body())
    add("auth post bare key", key, headers={"authorization": "k"}, data=ok_body())
    add("auth post latin-1 byte", key, headers={"authorization": "Bearer sécret"},
        data=ok_body())
    add("auth before body", key, data={"text": "{"})
    add("auth before body size", key, data={"parts": [["x", 2 * 1024 * 1024 + 1]]})
    add("auth wrong method", key, "GET")
    add("auth unknown path", key, "GET", "/v1/models")
    # --- 400s
    add("invalid json", data={"text": "{"})
    add("empty body", data={"text": ""})
    add("trailing garbage", data={"text": '{"state":"x","questions":{}} x'})
    add("array body", data=body([]))
    add("string body", data=body("x"))
    add("missing questions", data=body({"state": "x"}))
    add("missing state", data=body({"questions": {}}))
    add("null state", data=body({"state": None, "questions": {}}))
    add("null state before questions type", data=body({"state": None, "questions": []}))
    add("questions list", data=body({"state": "x", "questions": []}))
    add("questions null", data=body({"state": "x", "questions": None}))
    add("invalid utf-8", data={"b64": base64.b64encode(b'{"state":"\xff","questions":{}}').decode()})
    add("utf-8 surrogate bytes",
        data={"b64": base64.b64encode(b'{"state":"\xed\xa0\x80","questions":{}}').decode()})
    add("utf-8 bom", data={"b64": base64.b64encode(
        b"\xef\xbb\xbf" + json.dumps({"state": "x", "questions": {}}).encode()).decode()})
    add("utf-16 bom", data={"b64": base64.b64encode(
        json.dumps({"state": "x", "questions": {}}).encode("utf-16")).decode()})
    add("utf-16-le no bom", data={"b64": base64.b64encode(
        json.dumps({"state": "x", "questions": {}}).encode("utf-16-le")).decode()})
    add("utf-16-be no bom", data={"b64": base64.b64encode(
        json.dumps({"state": "x", "questions": {}}).encode("utf-16-be")).decode()})
    add("utf-32-le no bom", data={"b64": base64.b64encode(
        json.dumps({"state": "x", "questions": {}}).encode("utf-32-le")).decode()})
    add("utf-32 bom", data={"b64": base64.b64encode(
        json.dumps({"state": "x", "questions": {}}).encode("utf-32")).decode()})
    add("utf-16 odd length", data={"b64": base64.b64encode(
        json.dumps({"state": "x", "questions": {}}).encode("utf-16-le") + b"\x20").decode()})
    add("raw control character", data={"text": '{"state":"a\u0001b","questions":{}}'})
    add("lone surrogate state", data={"text": '{"state":"\\udc00","questions":{}}'})
    add("lone high surrogate", data={"text": '{"state":"x\\ud800y","questions":{}}'})
    add("reversed pair", data={"text": '{"state":"\\ude00\\ud83d","questions":{}}'})
    add("paired surrogate", data={"text": '{"state":"\\ud83d\\ude00","questions":{}}'})
    add("lone surrogate in question key",
        data={"text": '{"state":"x","questions":{"\\udc00":{"type":"noul","instructions":"y"}}}'})
    add("lone surrogate in unrelated field",
        data={"text": '{"state":"x","questions":{},"extra":["\\ud800"]}'})
    add("lone surrogate before max_len",
        data={"text": '{"state":"\\udc00","questions":{},"max_len":0}'})
    add("NaN literal", data={"text": '{"state":"x","questions":{},"min_confidence":NaN}'})
    add("Infinity in state", data={"text": '{"state":{"v":Infinity},"questions":{}}'})
    # --- 413s
    big = 2 * 1024 * 1024
    add("body over 2 MiB", data={"parts": [['{"state":"', 1], ["x", big], ['","questions":{}}', 1]]})
    add("body over 2 MiB chunked", chunked=True,
        data={"parts": [['{"state":"', 1], ["x", big], ['","questions":{}}', 1]]})
    add("body exactly 2 MiB", data={"parts": [['{"state":"x","questions":{}}', 1],
                                              [" ", big - len('{"state":"x","questions":{}}')]]})
    add("65 questions", data=body({"state": "x", "questions": {"q%d" % i: NOUL for i in range(65)}}))
    add("64 questions", data=body({"state": "x", "questions": {"q%d" % i: NOUL for i in range(64)}}))
    add("101 choice options", data=body({"state": "x", "questions": {"q": choice(101)}}))
    add("101 choice options list", data=body({"state": "x", "questions": {"q": choice(101, True)}}))
    add("100 choice options", data=body({"state": "x", "questions": {"q": choice(100)}}))
    add("33 score levels", data=body({"state": "x", "questions": {"q": score(33)}}))
    add("513 total options", data=body({"state": "x", "questions": {
        "a": choice(100), "b": choice(100), "c": choice(100), "d": choice(100), "e": choice(100),
        "f": score(13)}}))
    add("512 total options", data=body({"state": "x", "questions": {
        "a": choice(100), "b": choice(100), "c": choice(100), "d": choice(100), "e": choice(100),
        "f": score(12)}}))
    add("question id repr", data=body({"state": "x", "questions": {"it's": choice(101)}}))
    add("50001-char state", data=body({"state": "x" * 50001, "questions": {}}))
    add("50000-char state", data=body({"state": "x" * 50000, "questions": {}}))
    add("escaped state", data=body({"state": {"body": '"' * 30000}, "questions": {}}))
    add("lower-bound state", data=body({"state": {"a": "x" * 50001}, "questions": {}}))
    add("lower-bound many strings", data=body({"state": ["y" * 1000] * 70, "questions": {}}))
    add("many numbers state", data=body({"state": [1] * 20000, "questions": {}}))
    add("number state", data=body({"state": 5, "questions": {}}))
    add("empty string state", data=body({"state": "", "questions": {}}))
    add("too many questions before null state", data=body(
        {"state": None, "questions": {"q%d" % i: NOUL for i in range(65)}}))
    add("too many questions before refusal", data=body(
        {"state": "x", "questions": {"q%d" % i: NOUL for i in range(65)}, "hooks": []}))
    add("state size before refusal", data=body({"state": "x" * 50001, "questions": {}, "hooks": 1}))
    # --- 422s from serve.py
    add("score null level", data=body({"state": "x", "questions": {"q": score(4, null_at=2)}}))
    add("33 levels before null level", data=body({"state": "x", "questions": {"q": score(33, 1)}}))
    add("null level before state size", data=body(
        {"state": "x" * 50001, "questions": {"q": score(4, null_at=0)}}))
    add("refusal hooks", data=ok_body(hooks=[]))
    add("refusal null is fine", data=ok_body(hooks=None))
    add("refusal sorted", data=ok_body(on_predict_start="x", hooks_timeout=1, hooks_raise=False))
    add("refusal before surrogate", data={"text": '{"state":"\\udc00","questions":{},"hooks":0}'})
    add("max_len 0", data=ok_body(max_len=0))
    add("max_len string", data=ok_body(max_len="5"))
    add("max_len true", data=ok_body(max_len=True))
    add("max_len float", data=ok_body(max_len=5.0))
    add("max_len over budget", data=ok_body(max_len=9000))
    add("max_len at budget", data=ok_body(max_len=8192))
    add("max_len null", data=ok_body(max_len=None))
    add("head_max_len negative", data=ok_body(head_max_len=-1))
    add("max_len before head_max_len", data=ok_body(max_len=0, head_max_len="x"))
    add("budget env", {"LAYA_MAX_TOKEN_BUDGET": "100"}, data=ok_body(max_len=101))
    add("budget env invalid", {"LAYA_MAX_TOKEN_BUDGET": "abc"}, data=ok_body(max_len=9000))
    add("budget env zero", {"LAYA_MAX_TOKEN_BUDGET": "0"}, data=ok_body(head_max_len=8193))
    add("lang true", data=ok_body(lang=True))
    add("lang number", data=ok_body(lang=5))
    add("lang_guess list", data=ok_body(lang_guess=[]))
    add("lang null", data=ok_body(lang=None))
    add("lang code", data=ok_body(lang="de"))
    add("budget before lang", data=ok_body(max_len="x", lang=True))
    add("lang before min_confidence", data=ok_body(lang=1, min_confidence=2))
    add("min_confidence 2", data=ok_body(min_confidence=2))
    add("min_confidence negative", data=ok_body(min_confidence=-0.1))
    add("min_confidence true", data=ok_body(min_confidence=True))
    add("min_confidence string", data=ok_body(min_confidence="0.5"))
    add("min_confidence list", data=ok_body(min_confidence=[0.5]))
    add("min_confidence huge", data={"text": json.dumps(
        {"state": STATE, "questions": QUESTIONS})[:-1] + ',"min_confidence":1e400}'})
    add("min_confidence empty map", data=ok_body(min_confidence={}))
    add("min_confidence map bad value", data=ok_body(min_confidence={"choice:2": 2}))
    add("min_confidence map bool value", data=ok_body(min_confidence={"default": True}))
    add("min_confidence map string value", data=ok_body(min_confidence={"default": "x"}))
    add("min_confidence 0", data=ok_body(min_confidence=0))
    add("min_confidence 0.5", data=ok_body(min_confidence=0.5))
    add("min_confidence map", data=ok_body(min_confidence={"choice:3-5": 0.9, "default": 0.2}))
    add("min_confidence null", data=ok_body(min_confidence=None))
    add("min_confidence before task", data=ok_body(min_confidence=2, task="nope"))
    # --- 422s from routing (Router.predict)
    add("task unknown", data=ok_body(task="nope"))
    add("task number", data=ok_body(task=5))
    add("task bool", data=ok_body(task=True))
    add("task alias", data=ok_body(task="typed"))
    add("task typed_decisions", data=ok_body(task="Typed_Decisions"))
    add("model beats task", data=ok_body(model="english", task="nope"))
    add("model jev", data=ok_body(model="jev-1"))
    add("model number", data=ok_body(model=5))
    add("model published id", data=ok_body(model="convaiinnovations/laya-multilingual"))
    add("task null", data=ok_body(task=None))
    # --- 200s and env
    add("ok", data=ok_body())
    add("ok text/plain", headers={"content-type": "text/plain"}, data=ok_body())
    add("ok no questions", data=body({"state": "x", "questions": {}}))
    add("ok extra fields", data=ok_body(foo={"bar": [1, 2]}, stream=True))
    add("strict", {"LAYA_JEV_STRICT": "1"}, data=ok_body())
    add("strict yes", {"LAYA_JEV_STRICT": "yes"}, data=ok_body())
    add("strict off", {"LAYA_JEV_STRICT": "no"}, data=ok_body())
    add("root path prefixed", {"LAYA_ROOT_PATH": "/laya"}, path="/laya/v1/systemone", data=ok_body())
    add("root path stripped", {"LAYA_ROOT_PATH": "/laya"}, data=ok_body())
    add("root path health", {"LAYA_ROOT_PATH": "/laya"}, "GET", "/laya/health")
    add("root path trailing slash", {"LAYA_ROOT_PATH": "/laya"}, "GET", "/laya/health/")
    return out


def body_bytes(spec):
    if spec is None:
        return b""
    if "text" in spec:
        return spec["text"].encode("utf-8", "surrogatepass")
    if "b64" in spec:
        return base64.b64decode(spec["b64"])
    return b"".join(text.encode("utf-8") * n for text, n in spec["parts"])


def set_env(env):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(BASE_ENV)
    os.environ.update(env)


def run_case(case):
    set_env(case["env"])
    router = StubRouter()
    app = serve.create_app(router)
    data = body_bytes(case["body"])
    headers = {k: v.encode("latin-1") for k, v in case["headers"].items()}
    with TestClient(app, base_url="http://testserver", follow_redirects=False) as client:
        if case["chunked"]:
            content = (data[i:i + 65536] for i in range(0, len(data), 65536))
        else:
            content = data if case["body"] is not None else None
        resp = client.request(case["method"], case["path"], content=content, headers=headers)
    out = {"status": resp.status_code}
    out["json"] = json.loads(resp.content) if resp.content else None
    seen = {}
    for name in ("content-type", "allow", "retry-after", "www-authenticate", "location"):
        if name in resp.headers:
            seen[name] = resp.headers[name]
    if "allow" in seen:
        seen["allow"] = ", ".join(sorted(m.strip() for m in seen["allow"].split(",")))
    for name in ("server-timing", "x-inference-time-ms"):
        seen[name + "?"] = name in resp.headers
    out["response_headers"] = seen
    out["predict"] = [{"model": c["model"], "kwargs": c["kwargs"]} for c in router.calls]
    return out


def depth_case(d):
    set_env({})
    app = serve.create_app(StubRouter())
    text = '{"state":' + "[" * d + "]" * d + ',"questions":{}}'
    with TestClient(app) as client:
        resp = client.post("/v1/systemone", content=text.encode())
    return resp.status_code, resp.json()


def depth():
    # Body nesting = state nesting + 1.
    def accepted(d):
        return depth_case(d)[1] != {"detail": "request body must be valid JSON"}

    lo, hi = 100, 20000
    assert accepted(lo) and not accepted(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if accepted(mid):
            lo = mid
        else:
            hi = mid
    return {"max_state_depth": lo,
            "at_max": dict(zip(("status", "json"), depth_case(lo))),
            "over_max": dict(zip(("status", "json"), depth_case(lo + 1)))}


def main():
    recorded = []
    for case in cases():
        recorded.append({**case, **run_case(case)})
    emit(recorded, depth=depth(), questions=QUESTIONS, state=STATE)


if __name__ == "__main__":
    main()
