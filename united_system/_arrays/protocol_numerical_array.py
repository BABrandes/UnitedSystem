from typing import Generic, Protocol, runtime_checkable, TypeVar
from .._utils.value_type import VALUE_TYPE

PT = TypeVar("PT", bound=VALUE_TYPE, covariant=True)

@runtime_checkable
class ProtocolNumericalArray(Protocol, Generic[PT]):

    def sum(self) -> PT:
        ...

    def mean(self) -> PT:
        ...

    def std(self) -> PT:
        ...

    def min(self) -> PT:
        ...

    def max(self) -> PT:
        ...
