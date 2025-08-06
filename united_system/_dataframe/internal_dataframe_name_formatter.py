from typing import Protocol, Callable, Optional
from .._units_and_dimension.unit import Unit
from .._dataframe.column_key import ColumnKey

class InternalDataFrameColumnNameFormatter(Protocol):
    """
    Protocol for creating and retrieving internal dataframe column names.
    The column key is the key of the column in the UnitedDataframe, the unit is the unit of the column.
    The column key is a string, the unit is a Unit object.
    """

    def create_internal_dataframe_column_name(self, column_key: ColumnKey|str, unit: Optional[Unit]) -> str:
        ...
    @classmethod
    def retrieve_from_internal_dataframe_column_name(cls, internal_dataframe_column_name: str, column_key_constructor: Callable[[str], ColumnKey|str]|None=None) -> tuple[ColumnKey|str, Optional[Unit]]:
        ...

# Concrete implementation of the protocol
class SimpleInternalDataFrameNameFormatter(InternalDataFrameColumnNameFormatter):
    """
    Simple implementation of the InternalDataFrameColumnNameFormatter protocol.
    It creates and retrieves internal dataframe column names in the format "<column_key> [<unit>]" or "<column_key> [-]" if the unit is None.
    The column key is the key of the column in the UnitedDataframe, the unit is the unit of the column.
    The column key is a string, the unit is a Unit object.
    """

    def create_internal_dataframe_column_name(self, column_key: ColumnKey|str, unit: Optional[Unit]) -> str:
        return f"{column_key} [{unit}]" if unit != None else f"{column_key} [-]"
    @classmethod
    def retrieve_from_internal_dataframe_column_name(cls, internal_dataframe_column_name: str, column_key_constructor: Callable[[str], ColumnKey|str]|None=None) -> tuple[ColumnKey|str, Optional[Unit]]:
        return cls._retrieving_method(internal_dataframe_column_name, column_key_constructor)
    @staticmethod
    def _retrieving_method(internal_dataframe_column_name: str, column_key_constructor: Callable[[str], ColumnKey|str]|None=None) -> tuple[ColumnKey|str, Optional[Unit]]:
        # Find the indices of '[' and ']' in the internal_dataframe_column_name, looking from the end of the string
        internal_dataframe_column_name = internal_dataframe_column_name.strip()
        index_bracket_close: int = internal_dataframe_column_name.rfind(']')
        index_bracket_open: int = internal_dataframe_column_name.rfind('[')  # Fixed: search in full string, not substring
        unit_str: str = internal_dataframe_column_name[index_bracket_open+1:index_bracket_close]
        # Make sure there is a space before the '['
        if index_bracket_open > 0 and internal_dataframe_column_name[index_bracket_open-1] != ' ':
            raise ValueError(f"Invalid internal dataframe column name: {internal_dataframe_column_name}")
        # Get the rest of ths string, but without space
        column_key_str = internal_dataframe_column_name[:index_bracket_open-1]
        if unit_str == "-":
            unit: Optional[Unit] = None
        else:
            unit: Optional[Unit] = Unit(unit_str)
        if column_key_constructor is not None:
            column_key: ColumnKey|str = column_key_constructor(column_key_str)
        else:
            column_key: ColumnKey|str = column_key_str

        return column_key, unit


SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER: SimpleInternalDataFrameNameFormatter = SimpleInternalDataFrameNameFormatter()