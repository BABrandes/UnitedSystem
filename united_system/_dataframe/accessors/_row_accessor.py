from typing import Generic, Iterator, TypeVar, TYPE_CHECKING, Sequence, overload, Union

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe
    from ..._dataframe.column_key import ColumnKey

from ..._utils.scalar_type import SCALAR_TYPE, SCALAR_TYPE_RUNTIME
from ..._utils.value_type import VALUE_TYPE, VALUE_TYPE_RUNTIME

CK = TypeVar("CK", bound="ColumnKey|str")

class RowAccessor(Generic[CK]):
    """
    Internal class for row-based access to cell values.
    """

    def __init__(self, parent: "UnitedDataframe[CK]", row_index: int, column_keys: Sequence[CK]|None = None):
        self._parent: "UnitedDataframe[CK]" = parent
        self._row_index: int = row_index
        if column_keys is None:
            self._column_keys: Sequence[CK] = self._parent.colkeys
        else:
            self._column_keys: Sequence[CK] = column_keys

    @overload
    def __getitem__(self, column_key: CK) -> SCALAR_TYPE: ...
    @overload
    def __getitem__(self, column_key: Sequence[CK]) -> "RowAccessor[CK]": ...
    def __getitem__(self, column_key: Union[CK, Sequence[CK]]) -> Union[SCALAR_TYPE, "RowAccessor[CK]"]:
        if isinstance(column_key, Sequence):
            return RowAccessor[CK](self._parent, self._row_index, column_key) # type: ignore[no-any-return]
        else:
            return self._parent.cell_get_value(self._row_index, column_key)
    
    @overload
    def __setitem__(self, column_key: CK, value: VALUE_TYPE) -> None: ...
    @overload
    def __setitem__(self, column_key: CK, value: SCALAR_TYPE) -> None: ...
    def __setitem__(self, column_key: CK, value: VALUE_TYPE|SCALAR_TYPE) -> None:
        if isinstance(value, VALUE_TYPE_RUNTIME):
            assert isinstance(value, VALUE_TYPE)
            self._parent.cell_set_value(self._row_index, column_key, value)
        elif isinstance(value, SCALAR_TYPE_RUNTIME):
            assert isinstance(value, SCALAR_TYPE)
            self._parent.cell_set_scalar(self._row_index, column_key, value)
        else:
            raise ValueError(f"Invalid value type: {type(value)}")
    
    def __len__(self) -> int:
        return len(self._parent.colkeys)
    
    def __iter__(self) -> Iterator[CK]:
        return iter(self._column_keys)
    
    def __contains__(self, value: SCALAR_TYPE) -> bool:
        raise NotImplementedError("Row contains not yet implemented")
    
    def as_dict(self) -> dict[CK, SCALAR_TYPE]:
        return {column_key: self._parent.cell_get_value(self._row_index, column_key) for column_key in self._column_keys}