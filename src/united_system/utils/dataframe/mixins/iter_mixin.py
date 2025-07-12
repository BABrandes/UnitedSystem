"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the core access patterns for the dataframe including
iteration, indexing, and item assignment through magic methods.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Iterator, TYPE_CHECKING
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ....unit import Unit
from ....dimension import Dimension
from ..column_type import ColumnType

if TYPE_CHECKING:
    from ..accessors._row_accessor import RowAccessor # type: ignore
    from ..accessors._column_accessor import ColumnAccessor # type: ignore

class IterMixin(UnitedDataframeProtocol[CK]):
    """
    Mixin providing magic methods for dataframe item assignment.
    
    This mixin implements:
    - __iter__: Iterate over column names
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    def __iter__(self) -> Iterator[CK]:
        """
        Iterate over column names.
        """
        with self._rlock:
            return iter(self._column_keys)
        
    def __next__(self) -> CK:
        """
        Get the next column name.
        """
        with self._rlock:
            next_column_key: CK = next(self._column_keys) # type: ignore[no-any-return]
            return next_column_key # type: ignore[no-any-return]
        
    def iter_rows(self) -> "Iterator[RowAccessor[CK]]":
        """
        Iterate over row accessors.

        Returns:
            Iterator[RowAccessor[CK]]
        """
        for row_index in range(self._number_of_rows()):
            yield self._row_get_as_row_accessor(row_index)

    def iter_columns(self) -> "Iterator[ColumnAccessor[CK]]":
        """
        Iterate over column keys.

        Returns:
            Iterator[CK]
        """
        for column_key in self._column_keys:
            yield self._column_get_as_column_accessor(column_key)

    def iter_units(self) -> Iterator[Unit|None]:
        """
        Iterate over units.
        """
        for column_key in self._column_keys:
            yield self._unit_get(column_key)

    def iter_dimensions(self) -> Iterator[Dimension|None]:
        """
        Iterate over (column_key, dimension) pairs.
        """
        for column_key in self._column_keys:
            unit: Unit|None = self._column_units[column_key]
            if unit is not None:
                yield unit.dimension
            else:
                yield None

    def iter_coltypes(self) -> Iterator[ColumnType]:
        """
        Iterate over column types.
        """
        for column_key in self._column_keys:
            yield self._column_types[column_key]

    def column_items(self) -> "Iterator[tuple[CK, ColumnAccessor[CK]]]":
        """
        Iterate over (column_key, column_accessor) pairs.

        Returns:
            Iterator[Tuple[CK, _ColumnAccessor[CK]]]
        """
        for column_key in self._column_keys:
            yield column_key, self._column_get_as_column_accessor(column_key)

    def row_items(self) -> "Iterator[tuple[int, RowAccessor[CK]]]":
        """
        Iterate over (row_index, row_accessor) pairs.

        Returns:
            Iterator[Tuple[int, _RowAccessor[CK]]]
        """
        for row_index in range(self._number_of_rows()):
            yield row_index, self._row_get_as_row_accessor(row_index)

    def unit_items(self) -> Iterator[tuple[CK, Unit|None]]:
        """
        Iterate over (column_key, unit) pairs.

        Returns:
            Iterator[Tuple[CK, Unit|None]]
        """
        for column_key in self._column_keys:
            yield column_key, self._unit_get(column_key)
    
    def dimension_items(self) -> Iterator[tuple[CK, Dimension]]:
        """
        Iterate over (column_key, dimension) pairs.
        Only over columns with units.

        Returns:
            Iterator[Tuple[CK, Dimension|None]]
        """
        for column_key in self._columns_get_with_units():
            unit: Unit|None = self._column_units[column_key]
            if unit is None:
                raise ValueError(f"Column {column_key} has no unit.")
            else:
                yield column_key, unit.dimension

    def coltype_items(self) -> Iterator[tuple[CK, ColumnType]]:
        """
        Iterate over (column_key, column_type) pairs.

        Returns:
            Iterator[Tuple[CK, ColumnType]]
        """
        for column_key in self._column_keys:
            yield column_key, self._column_types[column_key]
        
        