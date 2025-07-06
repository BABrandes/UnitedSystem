from typing import Generic, TypeVar, Optional
from .base_classes.base_unit import BaseUnit
from .base_classes.base_dimension import BaseDimension
from abc import ABC

DT = TypeVar("DT", bound=BaseDimension)
UT = TypeVar("UT", bound=BaseUnit)

class United(ABC, Generic[DT, UT]):

    dimension: DT
    display_unit: Optional[UT] = None