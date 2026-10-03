"""Reference token sequences for every request question, on all three checkpoints.

The tokenizer is loaded exactly as `ONNXAgent.__init__` loads it (`_fix_tokenizer_config`, then
`AutoTokenizer` on the checkpoint's `tokenizer/` folder). Each question goes through the real
`Agent._check_question`, `ONNXAgent._to_internal` and `ONNXAgent._encode_state`; the latter calls
`build_sequence` with the state tokenized once (mask token replaced), `truncate_left` for a list
state and the question's `option_order`. The agent is built with `object.__new__`, so no weights
or ONNX session are loaded: `_encode_state` only reads `self.tok`.
"""
import json
import os

from transformers import AutoTokenizer

from laya.agent import Agent, _fix_tokenizer_config
from laya.common import render_options
from laya.onnx_agent import ONNXAgent

from _shared import CHECKPOINTS, NAMES, emit, error_of, load_cases


def load_agent(name):
    model_dir = os.path.join(CHECKPOINTS, name)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    _fix_tokenizer_config(model_dir)
    agent = object.__new__(ONNXAgent)
    agent.cfg = cfg
    agent.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    return agent


def main():
    corpus = load_cases()
    states = corpus["states"]
    cases = []
    checkpoints = {}
    for name in NAMES:
        agent = load_agent(name)
        # The same defaults `_infer_batch` applies.
        max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        tok = agent.tok
        checkpoints[name] = {
            "max_len": max_len,
            "head_max_len": head_max_len,
            "cls_token_id": tok.cls_token_id,
            "sep_token_id": tok.sep_token_id,
            "pad_token_id": tok.pad_token_id,
            "mask_token_id": tok.mask_token_id,
            "mask_token": tok.mask_token,
        }
        for index, req in enumerate(corpus["requests"]):
            state = states[req["state"]]
            for qid, qdef in req["questions"].items():
                case = {"checkpoint": name, "request_index": index, "qid": qid}
                try:
                    Agent._check_question(qid, qdef)
                    q = ONNXAgent._to_internal(qdef)
                    case["internal"] = q
                    case["options"] = render_options(q)
                    # One question per call: `_encode_state` tokenizes the state once and builds
                    # each question's row independently, so this is the same row it builds in a
                    # multi-question request.
                    item = agent._encode_state(state, [qid], {qid: q}, max_len, head_max_len)[0]
                    case.update({
                        "truncate_left": isinstance(state, list),
                        "ids": item["ids"],
                        "markers": item["markers"],
                        "stats": item["options"],
                        "truncation": item["state_stats"],
                    })
                except Exception as exc:  # recorded: the port must fail the same way
                    case["error"] = error_of(exc)
                cases.append(case)
    emit(cases, checkpoints=checkpoints)


if __name__ == "__main__":
    main()
