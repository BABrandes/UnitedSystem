from typing import Final

_PREFIX_PAIRS: Final[dict[str, float]] = {
    "Y": 10**24,
    "Z": 10**21,
    "E": 10**18,
    "P": 10**15,
    "T": 10**12,
    "G": 10**9,
    "M": 10**6,
    "k": 10**3,
    "h": 10**2,
    "da": 10**1,
    "d": 10**-1,
    "c": 10**-2,
    "m": 10**-3,
    "µ": 10**-6,  # Micro Sign (U+00B5)
    "μ": 10**-6,  # Greek Small Letter Mu (U+03BC) - same value as µ
    "n": 10**-9,
    "p": 10**-12,
    "f": 10**-15,
    "a": 10**-18,
    "z": 10**-21,
    "y": 10**-24,
}
