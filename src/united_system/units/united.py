from typing import Generic, TypeVar, Optional
from .base_classes.base_unit import BaseUnit
from .base_classes.base_dimension import BaseDimension

DT = TypeVar("DT", bound=BaseDimension)
UT = TypeVar("UT", bound=BaseUnit)

class United(Generic[DT, UT]):

    dimension: DT
    display_unit: Optional[UT] = None