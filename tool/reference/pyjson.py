"""Python JSON rendering: `json.dumps(ensure_ascii=False)` (what `serialize_state` uses) and
`laya.common.render_criterion`, for every corpus state and every criterion value in the requests.
"""
import json

from laya.common import render_criterion, serialize_state

from _shared import emit, load_cases


def criterion_values(requests):
    for req in requests:
        for qdef in req["questions"].values():
            crit = qdef.get("criteria")
            if isinstance(crit, dict):
                yield from crit.values()
            elif isinstance(crit, list):
                yield from crit


def main():
    corpus = load_cases()
    cases = []
    values = [("state", s) for s in corpus["states"]]
    values += [("criterion", v) for v in criterion_values(corpus["requests"])]
    for source, value in values:
        cases.append({
            "source": source,
            "value": value,
            "dumps": json.dumps(value, ensure_ascii=False),
            "serialized": serialize_state(value),
            "criterion": render_criterion(value),
        })
    emit(cases)


if __name__ == "__main__":
    main()
