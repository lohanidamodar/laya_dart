"""Reference `laya.lang.analyse` results for every corpus text and every corpus state.

Text cases carry `"text"`; state cases carry `"state"` (any JSON value, including strings).
"""
from laya.lang import analyse

from _shared import emit, load_cases


def main():
    corpus = load_cases()
    cases = [{"text": t, "analysis": analyse(t)} for t in corpus["texts"]]
    cases += [{"state": s, "analysis": analyse(s)} for s in corpus["states"]]
    emit(cases)


if __name__ == "__main__":
    main()
