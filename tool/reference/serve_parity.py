"""End-to-end reference: `laya-serve` on the real upstream torch weights.

The app is `laya.serve.create_app(router)` under Starlette's `TestClient`, with a real
`laya.router.Router` on CPU loading the pinned snapshot from the local Hugging Face cache
(`HF_HUB_OFFLINE=1`; passing the snapshot `revision` lets the cache resolve without a `refs/`
entry, and keeps the router's `routing.repo` the bundle repo id the Dart port reports).

Usage:
  serve_parity.py fixture <checkpoint>        fixture JSON on stdout
  serve_parity.py bench <checkpoint> <request> <n>  latency JSON on stdout

`fixture` sends every request in `cases.json` (its `state` is an index into `states`) twice:
with no `model` (auto-routed) and with `model: <checkpoint>`. An auto-routed request is recorded
in the fixture of the checkpoint the router picks, so it is kept here only when that is
`<checkpoint>`; one that routes to typed-decisions is listed under `skipped` instead, because
those weights are not part of this run. Each case records the exact body text sent, the status,
the raw response text and its parsed JSON.

`bench` posts one request `n` times after a warm-up and reports the per-request wall times and
the process's peak working set. `LAYA_THREADS` caps torch's intra-op threads, as laya-serve does.
"""
import ctypes
import ctypes.wintypes
import json
import os
import statistics
import sys
import time

from fastapi.testclient import TestClient

from laya import serve
from laya.router import Router

from _shared import emit, load_cases

REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
CHECKPOINTS = ("english", "multilingual")


def make_client():
    serve._apply_thread_limit()
    router = Router(device="cpu", revision=REVISION)
    app = serve.create_app(router)
    return router, TestClient(app, base_url="http://testserver")


def request_body(cases, index, model):
    req = dict(cases["requests"][index])
    body = {"state": cases["states"][req.pop("state")], "questions": req.pop("questions")}
    body.update(req)
    if model is not None:
        body["model"] = model
    return body


def post(client, text):
    resp = client.post("/v1/systemone", content=text.encode("utf-8"),
                       headers={"content-type": "application/json"})
    raw = resp.content.decode("utf-8")
    return {"status": resp.status_code, "text": raw, "json": json.loads(raw) if raw else None}


def fixture(checkpoint):
    cases = load_cases()
    router, client = make_client()
    recorded, skipped = [], []
    with client:
        for i in range(len(cases["requests"])):
            for variant in ("auto", "model"):
                body = request_body(cases, i, checkpoint if variant == "model" else None)
                if variant == "auto":
                    routed = router.route(body["state"], body["questions"])["model"]
                    if routed == "typed-decisions":
                        skipped.append({"request": i, "variant": variant,
                                        "reason": "auto-routes to typed-decisions"})
                    if routed != checkpoint:
                        continue
                text = json.dumps(body, ensure_ascii=False)
                recorded.append({"request": i, "variant": variant, "body": text,
                                 **post(client, text)})
    emit(recorded, checkpoint=checkpoint, revision=REVISION, skipped=skipped)


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_rss():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi")
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_void_p,
                                           ctypes.wintypes.DWORD]
    psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                               counters.cb)
    return counters.PeakWorkingSetSize


def bench(checkpoint, index, n):
    import torch

    cases = load_cases()
    _, client = make_client()
    text = json.dumps(request_body(cases, index, checkpoint), ensure_ascii=False)
    with client:
        warm = post(client, text)
        assert warm["status"] == 200, warm
        samples = []
        for _ in range(n):
            start = time.perf_counter()
            resp = post(client, text)
            samples.append((time.perf_counter() - start) * 1000.0)
            assert resp["status"] == 200, resp
    print(json.dumps({"checkpoint": checkpoint, "request": index, "n": n,
                      "threads": torch.get_num_threads(),
                      "median_ms": statistics.median(samples), "samples_ms": samples,
                      "peak_rss_bytes": peak_rss()}))


def main(argv):
    if len(argv) >= 2 and argv[0] == "fixture" and argv[1] in CHECKPOINTS:
        fixture(argv[1])
    elif len(argv) == 4 and argv[0] == "bench" and argv[1] in CHECKPOINTS:
        bench(argv[1], int(argv[2]), int(argv[3]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
