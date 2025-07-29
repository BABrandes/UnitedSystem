from enum import Enum

class UnitPrefix(Enum):
    value: tuple[str, list[str], float] # type: ignore

    YOTTA = ("yotta", ["Y"], 10**24)
    ZETTA = ("zetta", ["Z"], 10**21)
    EXA = ("exa", ["E"], 10**18)
    PETA = ("peta", ["P"], 10**15)
    TERA = ("tera", ["T"], 10**12)
    GIGA = ("giga", ["G"], 10**9)
    MEGA = ("mega", ["M"], 10**6)
    KILO = ("kilo", ["k"], 10**3)
    HECTO = ("hecto", ["h"], 10**2)
    DEKA = ("deka", ["da"], 10**1)
    DECI = ("deci", ["d"], 10**-1)
    CENTI = ("centi", ["c"], 10**-2)
    MILLI = ("milli", ["m"], 10**-3)
    MICRO = ("micro", ["µ", "μ"], 10**-6)
    NANO = ("nano", ["n"], 10**-9)
    PICO = ("pico", ["p"], 10**-12)
    FEMTO = ("femto", ["f"], 10**-15)
    ATTO = ("atto", ["a"], 10**-18)
    ZEPTO = ("zepto", ["z"], 10**-21)
    YOCTO = ("yocto", ["y"], 10**-24)

    @property
    def name(self) -> str:
        return self.value[0]

    @property
    def prefix_string(self) -> str:
        return self.value[1][0]
    
    @property
    def prefixes(self) -> list[str]:
        return self.value[1]
    
    @property
    def factor(self) -> float:
        return self.value[2]

    @classmethod
    def get_prefix(cls, prefix: str) -> "UnitPrefix":
        for unit_prefix in cls:
            if prefix in unit_prefix.prefixes:
                return unit_prefix
        raise ValueError(f"Prefix {prefix} not found")