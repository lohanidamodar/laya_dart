"""Reference `Router.route` decisions, without loading any checkpoint.

Two routers are built with `preload=False`: one with `auto_task_detection=False` (the default)
and one with `True`. Every request is routed by both. The explicit overrides (`model=` each name
and alias, `task=` each name, `lang=` and `lang_guess=`) are applied to a few representative
requests, and every corpus text is routed as a bare state by the default router.
"""
import os

from laya.router import _ALIASES, DEFAULT_MODELS, Router

from _shared import emit, error_of, load_cases

# A deployment-wide digest map would not change routing, but keep the input fixed anyway.
os.environ.pop("LAYA_SHA256_DIGESTS", None)

OVERRIDE_REQUESTS = (0, 3, 19, 26)


def overrides():
    for name in list(DEFAULT_MODELS) + list(_ALIASES) + ["EN", " Multi ", "gpt"]:
        yield {"model": name}
    for name in list(DEFAULT_MODELS) + ["typed_decisions", "Typed-Decisions", "multi", "gpt"]:
        yield {"task": name}
    for code in ("de", "en-US", "", "en_US.UTF-8", "C", "und", "  "):
        yield {"lang": code}
    yield {"lang_guess": "fr"}
    yield {"lang_guess": "en"}
    yield {"lang_guess": "C.UTF-8"}
    yield {"lang": "en", "model": "multilingual"}


def main():
    corpus = load_cases()
    states = corpus["states"]
    routers = {
        "default": Router(preload=False, auto_task_detection=False),
        "auto_task_detection": Router(preload=False, auto_task_detection=True),
    }
    calls = []
    for index, req in enumerate(corpus["requests"]):
        for router in routers:
            calls.append((router, {"request_index": index}, states[req["state"]],
                          req["questions"], {}))
    for index in OVERRIDE_REQUESTS:
        req = corpus["requests"][index]
        for kwargs in overrides():
            for router in routers:
                calls.append((router, {"request_index": index}, states[req["state"]],
                              req["questions"], kwargs))
    for text in corpus["texts"]:
        calls.append(("default", {"state": text}, text, None, {}))

    cases = []
    for router, where, state, questions, kwargs in calls:
        case = {"router": router, **where, "kwargs": kwargs}
        try:
            case["decision"] = dict(routers[router].route(state, questions, **kwargs))
        except Exception as exc:
            case["error"] = error_of(exc)
        cases.append(case)
    emit(cases, routers={
        "default": {"preload": False, "auto_task_detection": False},
        "auto_task_detection": {"preload": False, "auto_task_detection": True},
    })


if __name__ == "__main__":
    main()
