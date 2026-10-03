"""Reference decoding of synthetic logits into answers, on all three checkpoints' temperatures.

No weights are loaded. The agent is `object.__new__(ONNXAgent)` with:

* `cfg`, `temperature`, `temperature_by_options` and `lang_temperatures` set from the upstream
  `rl_agent_config.json`, clamped as `ONNXAgent.__init__` clamps them (`clamp_temperature`,
  `resolve_lang_temperatures`);
* the checkpoint's real tokenizer, loaded as `ONNXAgent.__init__` loads it;
* `session` replaced by a stub whose `run` returns the synthetic `logits` and `act_logits`.

Each item then goes through the real `ONNXAgent.predict_batch`, so the real `_infer_batch`
(collation, the act softmax), `_decode_answers` and `apply_confidence_gate` produce the answer.
`_decode_answers` takes the act *probabilities*, which `_infer_batch` computes from the act
logits, and `min_confidence` is applied after it by `predict_batch`; calling `_decode_answers`
alone would have meant reimplementing both.

Logits are float32, as ONNX Runtime returns them. The recorded `logits` / `act_logits` are those
float32 values in their shortest float32 round-trip form: read them into a `Float32List` to get
back exactly the values the reference decoded.
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import clamp_temperature, resolve_lang_temperatures
from laya.onnx_agent import ONNXAgent

from _shared import CHECKPOINTS, NAMES, emit

STATE = "Synthetic state for decode fixtures."
MIN_CONFIDENCES = (None, 0.0, 0.6)
TYPES = ("choice", "score", "noul")


class StubSession:
    """Stands in for `onnxruntime.InferenceSession`; returns the logits it was given."""

    def __init__(self):
        self.outputs = None

    def run(self, names, inputs):
        assert names == ["logits", "act_logits"]
        rows = inputs["input_ids"].shape[0]
        logits, act_logits = self.outputs
        assert logits.shape[0] == rows and act_logits.shape[0] == rows
        return [logits, act_logits]


def f32(v):
    """A float32 as the shortest decimal that rounds back to it."""
    out = float(str(np.float32(v)))
    assert np.float32(out) == np.float32(v)
    return out


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


def question(qtype, k, rng_labels):
    """A request-form question with `k` options of type `qtype`."""
    if qtype == "choice":
        labels = ["option_%d" % i for i in range(k)]
        if rng_labels:
            return {"type": "choice", "instructions": "Pick one option.",
                    "criteria": {label: "Description of %s" % label for label in labels}}
        return {"type": "choice", "instructions": "Pick one option.", "criteria": labels}
    if qtype == "score":
        return {"type": "score", "instructions": "Rate this.",
                "criteria": ["level %d" % i for i in range(k)]}
    return {"type": "noul", "instructions": "The statement holds."}


def items():
    """(question, logits row, act logits) triples: 200 seeded random ones, then edge rows."""
    rng = np.random.default_rng(1234)
    out = []
    for _ in range(200):
        qtype = TYPES[int(rng.integers(0, 3))]
        k = int(rng.integers(2, 13)) if qtype != "noul" else 2
        as_map = bool(rng.integers(0, 2))
        logits = rng.normal(0, 3, k)
        act = rng.normal(0, 2, 2)
        out.append((question(qtype, k, as_map), logits, act))
    act = np.array([0.0, 0.0])
    for qtype, k in (("choice", 4), ("score", 5), ("noul", 2), ("choice", 12)):
        out.append((question(qtype, k, False), np.zeros(k), act))         # all equal
        out.append((question(qtype, k, False), np.full(k, 7.25), act))    # all equal, nonzero
        huge = np.zeros(k)
        huge[k - 1] = 1e4
        out.append((question(qtype, k, False), huge, np.array([1e4, -1e4])))   # one huge logit
        padded = np.full(12 if k < 12 else 16, -1e4)
        padded[:k] = np.linspace(-2.0, 3.0, k)
        out.append((question(qtype, k, False), padded, np.array([0.5, -0.5])))  # -1e4 padding
    out.append((question("choice", 1, False), np.array([2.5]), np.array([1.0, 2.0])))       # k=1
    out.append((question("choice", 1, True), np.array([2.5, -1e4, -1e4]), np.array([0.0, 3.0])))
    out.append((question("noul", 2, False), np.array([-30.0, 30.0]), np.array([-3.0, 3.0])))
    out.append((question("score", 11, False), np.arange(11.0), np.array([2.0, 0.0])))
    return out


def main():
    configs = {}
    temperatures = {}
    cases = []
    for name in NAMES:
        with open(os.path.join(CHECKPOINTS, name, "onnx_config.json"), encoding="utf-8") as f:
            configs[name] = json.load(f)
        agent = load_agent(name)
        temperatures[name] = {"temperature": agent.temperature,
                              "temperature_by_options": agent.temperature_by_options}
        for qdef, logits, act in items():
            row = np.asarray(logits, dtype=np.float32)[None, :]
            act_row = np.asarray(act, dtype=np.float32)[None, :]
            for mc in MIN_CONFIDENCES:
                agent.session.outputs = (row, act_row)
                result = agent.predict_batch([STATE], {"q": qdef}, min_confidence=mc)[0]
                cases.append({
                    "checkpoint": name,
                    "question": qdef,
                    "logits": [f32(v) for v in row[0]],
                    "act_logits": [f32(v) for v in act_row[0]],
                    "min_confidence": mc,
                    "answer": result["answers"]["q"],
                })
    emit(cases, configs=configs, temperatures=temperatures, state=STATE)


if __name__ == "__main__":
    main()
