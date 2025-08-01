from typing import TypeVar, Generic
from .._utils.general import VALUE_TYPE

PT = TypeVar("PT", bound=VALUE_TYPE)

class BaseScalar(Generic[PT]):
    
    canonical_value: PT

    def __init__(self, value: PT):
        self.canonical_value = value

    def value(self) -> PT:
        return self.canonical_value

    def __str__(self) -> str:
        return str(self.canonical_value)