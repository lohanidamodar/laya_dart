"""Reference NFC from Hugging Face tokenizers' own normalizer.

Usage: nfc.py

`tokenizers.normalizers.NFC` is the normalizer both ByteLevel checkpoints declare. The inputs
probe every code point that has a canonical decomposition or a nonzero combining class in this
Python's unicodedata (a later Unicode than the tokenizers crate implements, so the cases also show
what it leaves alone), plus Hangul syllables and jamo. unicodedata only picks the inputs; every
recorded output is `NFC().normalize_str(input)`.
"""
import sys
import unicodedata

from tokenizers.normalizers import NFC

from _shared import emit

ACUTE, DOT_BELOW = chr(0x0301), chr(0x0323)


def inputs():
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        ch = chr(cp)
        decomposition = unicodedata.decomposition(ch)
        if decomposition and not decomposition.startswith("<"):
            parts = "".join(chr(int(h, 16)) for h in decomposition.split())
            yield ch
            yield parts
            yield "a" + parts + DOT_BELOW
        if unicodedata.combining(ch):
            # Reordering against acute (230) and dot below (220), and composition across it.
            yield "a" + ACUTE + ch
            yield "a" + ch + DOT_BELOW + ACUTE
            yield ch + "a"
    for lead in (0x1100, 0x1112, 0x1113):
        for vowel in (0x1161, 0x1175, 0x1176):
            for trail in (None, 0x11A7, 0x11A8, 0x11C2, 0x11C3):
                s = chr(lead) + chr(vowel) + ("" if trail is None else chr(trail))
                yield s
                yield s + ACUTE
    yield chr(0xAC00) + chr(0x11A8)  # LV + T composes
    yield chr(0xAC01) + chr(0x11A8)  # LVT + T does not
    yield chr(0xD7A3) + chr(0x11A8)
    yield chr(0x1100) + ACUTE + chr(0x1161)  # a mark between L and V blocks the composition


def main():
    nfc = NFC()
    seen = set()
    cases = []
    for text in inputs():
        if text in seen:
            continue
        seen.add(text)
        cases.append({"text": text, "nfc": nfc.normalize_str(text)})
    emit(cases, unicode=unicodedata.unidata_version)


if __name__ == "__main__":
    sys.exit(main())
