"""Reference token ids for every corpus text.

Named tokenizer_ids.py, not tokenize.py: a tokenize.py beside these scripts shadows the
standard-library `tokenize` module that inspect, linecache and huggingface_hub import.

Usage: tokenizer_ids.py <tokenizer.json> <name>

`ids` is `tokenizers.Tokenizer.encode(text, add_special_tokens=False).ids`. `option_ids` is the
option encoding `common.build_head` uses: `" " + text` through a fast tokenizer with
`truncation=True, max_length=48`.
"""
import sys

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from _shared import emit, load_cases


def main():
    path, name = sys.argv[1], sys.argv[2]
    tok = Tokenizer.from_file(path)
    fast = PreTrainedTokenizerFast(tokenizer_file=path)
    cases = []
    for text in load_cases()["texts"]:
        ids = tok.encode(text, add_special_tokens=False).ids
        option_ids = fast(" " + text, add_special_tokens=False, truncation=True,
                          max_length=48)["input_ids"]
        cases.append({"text": text, "ids": ids, "option_ids": option_ids})
    emit(cases, tokenizer=name)


if __name__ == "__main__":
    main()
