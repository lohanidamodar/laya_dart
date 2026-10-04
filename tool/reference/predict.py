"""Reference full predict results ({model, answers, usage}) on synthetic logits, all checkpoints.

No weights are loaded. As in decode.py, the agent is `object.__new__(ONNXAgent)` with the
checkpoint's real tokenizer, its `rl_agent_config.json` (temperatures clamped as
`ONNXAgent.__init__` clamps them) and a stub session. The stub returns seeded random float32
logits shaped `[rows, K]` and act logits `[rows, 2]` for whatever batch it is handed, and records
them so the Dart test can feed back exactly the same values.

Every corpus request (plus an empty-questions one) goes through the real
`ONNXAgent.predict_batch([state], questions, ...)` -- the path `ONNXAgent.system_one` takes --
under a few token-budget and `min_confidence` variants. That runs the real `_check_question`,
`_encode_state` (and its marker-count check), `collate_items`, `_decode_answers`, the usage
assembly and `apply_confidence_gate`. A request laya rejects is recorded with its error.

`ONNXAgent` reports `model` as "laya-rl-agent-onnx"; the torch `Agent` that `laya-serve` runs
reports "laya-rl-agent". Everything else in the payload is the same code on both backends.

Also recorded: `laya.serve._resolve_model` for a list of client `model` values.
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import clamp_temperature, resolve_lang_temperatures
from laya.onnx_agent import ONNXAgent
from laya.serve import _resolve_model

from _shared import CHECKPOINTS, NAMES, emit, error_of, load_cases

VARIANTS = (
    {},
    {"max_len": 128},
    {"max_len": 48, "head_max_len": 32},
    {"min_confidence": 0.6},
    {"min_confidence": {"choice:2": 0.9, "choice:3-5": 0.5, "default": 0.75}},
)

MODEL_VALUES = (
    None, "", "english", "EN", " Multi ", "ml", "typed", "laya", "default",
    "multilingual", "typed-decisions", "typed_decisions",
    "convaiinnovations/laya", "convaiinnovations/laya-multilingual",
    "Convaiinnovations/Laya-Typed-Decisions ", "convaiinnovations/laya-typed-decisions",
    "jev-1", "jev-latest", "gpt-4", "laya-auto",
)


def f32(v):
    """A float32 as the shortest decimal that rounds back to it."""
    out = float(str(np.float32(v)))
    assert np.float32(out) == np.float32(v)
    return out


class StubSession:
    """Stands in for `onnxruntime.InferenceSession`: seeded logits for any batch, recorded."""

    def __init__(self):
        self.rng = None
        self.calls = []

    def run(self, names, inputs):
        assert names == ["logits", "act_logits"]
        rows = inputs["input_ids"].shape[0]
        k = inputs["marker_pos"].shape[1]
        logits = self.rng.normal(0, 3, (rows, k)).astype(np.float32)
        act = self.rng.normal(0, 2, (rows, 2)).astype(np.float32)
        self.calls.append({
            "rows": rows,
            "markers": k,
            "logits": [[f32(v) for v in row] for row in logits],
            "act_logits": [[f32(v) for v in row] for row in act],
        })
        return [logits, act]


def load_agent(name):
    model_dir = os.path.join(CHECKPOINTS, name)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    _fix_tokenizer_config(model_dir)
    agent = object.__new__(ONNXAgent)
    agent.cfg = cfg
    agent.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    # onnx_agent.py:185-193, without the warning bookkeeping.
    agent.temperature_raw = cfg.get("temperature", [1.0, 1.0, 1.0])
    agent.temperature_by_options_raw = cfg.get("temperature_by_options", {})
    agent.temperature = [clamp_temperature(t) for t in agent.temperature_raw]
    agent.temperature_by_options = {k: clamp_temperature(v)
                                    for k, v in agent.temperature_by_options_raw.items()}
    agent.lang_temperatures = resolve_lang_temperatures({}, agent.temperature_raw)
    agent.session = StubSession()
    return agent


def main():
    corpus = load_cases()
    states = corpus["states"]
    requests = list(corpus["requests"]) + [{"state": 0, "questions": {}}]
    configs = {}
    cases = []
    seed = 0
    for name in NAMES:
        with open(os.path.join(CHECKPOINTS, name, "onnx_config.json"), encoding="utf-8") as f:
            configs[name] = json.load(f)
        agent = load_agent(name)
        for index, req in enumerate(requests):
            for variant in VARIANTS:
                seed += 1
                agent.session.rng = np.random.default_rng(seed)
                agent.session.calls = []
                case = {"checkpoint": name, "request_index": index, "variant": variant}
                try:
                    result = agent.predict_batch([states[req["state"]]], req["questions"],
                                                 **variant)[0]
                    case["result"] = result
                except Exception as exc:  # recorded: the port must fail the same way
                    case["error"] = error_of(exc)
                assert len(agent.session.calls) <= 1
                case["run"] = agent.session.calls[0] if agent.session.calls else None
                cases.append(case)
    resolve = [{"model": m, "resolved": _resolve_model(m)} for m in MODEL_VALUES]
    emit(cases, configs=configs, requests=requests, states=states, resolve_model=resolve)


if __name__ == "__main__":
    main()
