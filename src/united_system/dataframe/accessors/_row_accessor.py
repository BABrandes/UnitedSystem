from typing import Generic, Iterator, TypeVar
from ...united_dataframe import UnitedDataframe, ColumnKey
from ...real_united_scalar import RealUnitedScalar
from ..column_type import SCALAR_TYPE

CK = TypeVar("CK", bound=ColumnKey|str)

class _RowAccessor(Generic[CK]):
    """
    Internal class for row-based access to cell values.
    """
    def __init__(self, parent: UnitedDataframe[CK], row_index: int):
        self._parent: UnitedDataframe[CK] = parent
        self._row_index: int = row_index

    def __getitem__(self, column_key: CK) -> SCALAR_TYPE:
        return self._parent.get_cell_value(self._row_index, column_key)
    
    def __setitem__(self, column_key: CK, value: RealUnitedScalar):
        self._parent.set_cell_value(self._row_index, column_key, value)
    
    def __len__(self) -> int:
        return len(self._parent)
    
    def __iter__(self) -> Iterator[SCALAR_TYPE]:
        return self._parent.get_iterator_for_row(self._row_index)
    
    def __contains__(self, value: SCALAR_TYPE) -> bool:
        return value in self._parent.get_iterator_for_row(self._row_index)