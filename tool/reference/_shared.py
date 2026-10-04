"""Shared plumbing for the reference scripts: corpus loading and fixture output.

Nothing here computes a reference value; that is always a call into laya, tokenizers or
transformers in the script that records it.
"""
import json
import os
import sys
from importlib.metadata import version

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CHECKPOINTS = os.path.join(ROOT, ".tooling", "checkpoints")
NAMES = ("english", "multilingual", "typed-decisions")


def load_cases():
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        return json.load(f)


def generator():
    return {
        "laya": version("laya"),
        "tokenizers": version("tokenizers"),
        "transformers": version("transformers"),
        "numpy": version("numpy"),
        "python": sys.version.split()[0],
    }


def error_of(exc):
    return "%s: %s" % (type(exc).__name__, exc)


def emit(cases, **extra):
    """Write `{"generator", **extra, "cases": [...]}` to stdout, one case per line."""
    head = {"generator": generator(), **extra}
    out = sys.stdout
    dumps = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    out.write("{")
    for key, value in head.items():
        out.write("%s:%s,\n" % (dumps(key), dumps(value)))
    out.write('"cases":[\n')
    out.write(",\n".join(dumps(c) for c in cases))
    out.write("\n]}\n")
