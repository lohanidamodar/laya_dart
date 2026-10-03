"""Reference per-code-point behaviour of the Python `str`, `re` and `unicodedata` calls that
`laya.lang` and `%r` depend on, for every code point (surrogates included).

Usage: unicode_props.py

Each bit is one observed behaviour of this Python (3.12, Unicode 15.0.0), not a UCD field:

  1     ch.isalpha()                   64    ("A" + ch).isupper() is False
  2     ch.isdecimal()                 128   unicodedata.combining(ch) != 0
  4     re.match(r"\\w", ch)            256   ch.isprintable()
  8     ch.isspace()                   512   ("A" + ch + SIGMA).lower() ends in final sigma
  16    ch.islower()                   1024  (ch + SIGMA).lower() ends in final sigma
  32    ch.isupper()                   2048  ("A" + SIGMA + ch).lower()[1] is final sigma
  4096  re.match(r"\\d", ch)

`runs` holds `[start, bits]` for each run of equal bits. `lower` holds `[cp, [lowered...]]` for
every code point whose `ch.lower()` differs from `ch`.
"""
import re
import sys
import unicodedata

from _shared import emit

SIGMA, FINAL = "Σ", "ς"
WORD, DIGIT = re.compile(r"\w"), re.compile(r"\d")


def bits(ch):
    b = 0
    b |= 1 if ch.isalpha() else 0
    b |= 2 if ch.isdecimal() else 0
    b |= 4 if WORD.match(ch) else 0
    b |= 8 if ch.isspace() else 0
    b |= 16 if ch.islower() else 0
    b |= 32 if ch.isupper() else 0
    b |= 64 if not ("A" + ch).isupper() else 0
    b |= 128 if unicodedata.combining(ch) else 0
    b |= 256 if ch.isprintable() else 0
    b |= 512 if ("A" + ch + SIGMA).lower().endswith(FINAL) else 0
    b |= 1024 if (ch + SIGMA).lower().endswith(FINAL) else 0
    b |= 2048 if ("A" + SIGMA + ch).lower()[1] == FINAL else 0
    b |= 4096 if DIGIT.match(ch) else 0
    return b


def main():
    runs, lower, last = [], [], None
    for cp in range(0x110000):
        ch = chr(cp)
        b = bits(ch)
        if b != last:
            runs.append([cp, b])
            last = b
        low = ch.lower()
        if low != ch:
            lower.append([cp, [ord(c) for c in low]])
    emit([{"runs": runs, "lower": lower}], unicode=unicodedata.unidata_version)


if __name__ == "__main__":
    sys.exit(main())
