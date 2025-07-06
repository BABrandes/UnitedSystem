from typing import Protocol, Callable, Generic, TypeVar
from dataclasses import dataclass
from pandas._typing import Dtype
from ..unit import Unit
from ..dimension import Dimension
from .column_type import ColumnType
from .column_key import ColumnKey

@dataclass(frozen=True, slots=True)
class ColumnInformation():
    dimension: Dimension|None
    column_type: ColumnType
    display_unit: Unit|None

    def __postinit__(self):
        if self.dimension is None:
            if self.display_unit is not None:
                raise ValueError(f"When the unit quantity is None, the display unit must also be None")
        else:
            if self.display_unit is None:
                self.display_unit = self.dimension.canonical_unit()
            if not self.display_unit.compatible_to(self.dimension):
                raise ValueError(f"Display unit {self.display_unit} is not compatible with unit quantity {self.dimension}")

    def internal_dataframe_column_name(self, column_key: ColumnKey|str, internal_column_name_formatter: "InternalDataFrameNameFormatter[CK]" = "SIMPLE_INTERNAL_NAME_FORMATTER") -> str:
        return internal_column_name_formatter.create_internal_dataframe_column_name(column_key, self)

    @classmethod
    def create(
        cls,
        dimension: Dimension|None,
        column_type: ColumnType,
        display_unit: Unit|None=None) -> "ColumnInformation":
        return cls(dimension, column_type, display_unit)

CK = TypeVar("CK", bound=ColumnKey|str, default=str)
class InternalDataFrameNameFormatter(Protocol, Generic[CK]):
    def create_internal_dataframe_column_name(self, column_key: CK, column_information: ColumnInformation[CK]) -> str:
        ...
    @classmethod
    def retrieve_from_internal_dataframe_column_name(cls, internal_dataframe_column_name: str, dtype: Dtype, column_key_constructor: Callable[[str], CK]|None=None) -> tuple[CK, ColumnInformation[CK]]:
        ...

def x(internal_dataframe_column_name: str, dtype: Dtype, column_key_constructor: Callable[[str], CK]|None=None) -> tuple[CK, ColumnInformation[CK]]:
    # Find the indices of '[' and ']' in the internal_dataframe_column_name, looking from the end of the string
    internal_dataframe_column_name = internal_dataframe_column_name.strip()
    index_bracket_close: int = internal_dataframe_column_name.rfind(']')
    index_bracket_open: int = internal_dataframe_column_name[index_bracket_close:].rfind('[')
    display_unit: str = internal_dataframe_column_name[index_bracket_open+1:index_bracket_close]
    # Make sure there is a space before the '['
    if index_bracket_open > 0 and internal_dataframe_column_name[index_bracket_open-1] != ' ':
        raise ValueError(f"Invalid internal dataframe column name: {internal_dataframe_column_name}")
    # Get the rest of ths string, but without space
    column_key_str = internal_dataframe_column_name[:index_bracket_open-1]
    if display_unit == "-":
        display_unit: Unit|None = None
    else:
        display_unit: Unit = Unit.parse_string(display_unit)
    if CK == str:
        column_key: CK = column_key_str
    else:
        column_key: CK = column_key_constructor

    # Get the column type
    column_type: ColumnType = ColumnType.from_dtype(dtype)

    column_information: ColumnInformation = ColumnInformation.create(
        unit_quantity=display_unit.unit_quantity() if display_unit is not None else None,
        column_type=column_type,
        display_unit=display_unit if display_unit is not None else None
    )
    return column_key, column_information
SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER: InternalDataFrameNameFormatter[CK] = InternalDataFrameNameFormatter[CK](
    create_internal_dataframe_column_name=lambda column_key, column_information: f"{column_key} [{column_information.display_unit}]" if column_information.display_unit != None else f"{column_key} [-]",
    retrieve_from_internal_dataframe_column_name=lambda internal_dataframe_column_name, dtype, column_key_constructor: x(internal_dataframe_column_name, dtype, column_key_constructor))