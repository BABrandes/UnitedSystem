"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the __getitem__ magic method for the dataframe.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, Any, overload, Union
from collections.abc import Sequence
import numpy as np
import pandas as pd
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._dataframe.column_key import ColumnKey
from ..._arrays.bool_array import BoolArray
from ..._utils.general import VALUE_TYPE

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe
    from ..accessors._row_accessor import RowAccessor # type: ignore
    from ..accessors._column_accessor import ColumnAccessor # type: ignore

class AccessorGetitemMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Mixin providing magic methods for dataframe access patterns.
    
    This mixin implements:
    - __getitem__: Comprehensive indexing (columns, rows, cells, slices)
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    ################### Overloads for single indexing #########################################################

    @overload
    def __getitem__(self, key: CK) -> "ColumnAccessor[CK]":
        """
        Single indexing (e.g., df[column]).

        Args:
            key (CK): A column key.

        Returns:
            _ColumnAccessor[CK]: The column accessor.
        """
        ...

    @overload
    def __getitem__(self, key: int) -> "RowAccessor[CK]":
        """
        Single indexing (e.g., df[row]).
        """
        ...

    @overload
    def __getitem__(self, key: slice) -> "UnitedDataframe[CK]":
        """
        Single indexing (e.g., df[slice]).
        """
        ...

    @overload
    def __getitem__(self, key: Sequence[CK]) -> "UnitedDataframe[CK]":
        """
        Single indexing (e.g., df[sequence of columns]).
        """
        ...

    ################### Overloads for tuple indexing #########################################################

    @overload
    def __getitem__(self, key: tuple[CK, int]) -> VALUE_TYPE:
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[CK, int]): A tuple of indexing arguments.

        Returns:
            PYTHON_SCALAR_TYPE: The value at the given cell position.
        """
        ...

    @overload
    def __getitem__(self, key: tuple[int, CK]) -> VALUE_TYPE:
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[int, CK]): A tuple of indexing arguments.

        Returns:
            PYTHON_SCALAR_TYPE: The value at the given cell position.
        """
        ...
    
    @overload
    def __getitem__(self, key: tuple[CK, slice]) -> "ColumnAccessor[CK]":
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[CK, slice]): A tuple of indexing arguments.

        Returns:
            _ColumnAccessor[CK]: The column accessor.
        """
        ...

    @overload
    def __getitem__(self, key: tuple[slice, CK]) -> "ColumnAccessor[CK]":
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[slice, CK]): A tuple of indexing arguments.

        Returns:
            _RowAccessor[CK]: The row accessor.
        """
        ...

    @overload
    def __getitem__(self, key: tuple[int, Sequence[CK]]) -> "RowAccessor[CK]":
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[int, sequence[CK]]): A tuple of indexing arguments.

        Returns:
            _ColumnAccessor[CK]: The column accessor.
        """
        ...

    @overload
    def __getitem__(self, key: tuple[Sequence[CK], int]) -> "RowAccessor[CK]":
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[sequence[CK], int]): A tuple of indexing arguments.

        Returns:
            PYTHON_SCALAR_TYPE: The value at the given cell position.
        """
        ...

    @overload
    def __getitem__(self, key: tuple[Sequence[CK], slice]) -> "UnitedDataframe[CK]":
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[sequence[CK], slice]): A tuple of indexing arguments.

        Returns:
            PYTHON_SCALAR_TYPE: The value at the given cell position.
        """
        ...
    
    @overload
    def __getitem__(self, key: tuple[slice, Sequence[CK]]) -> "UnitedDataframe[CK]":
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[slice, sequence[CK]]): A tuple of indexing arguments.

        Returns:
            UnitedDataframe[CK]: The dataframe.
        """
        ...

    ################### Overloads for other #########################################################

    @overload
    def __getitem__(self, key: pd.Series) -> "UnitedDataframe[CK]": # type: ignore[no-any-return]
        """
        Boolean row filtering.

        Args:
            key (pd.Series): A boolean Series with the same length as the DataFrame.

        Returns:
            UnitedDataframe[CK]: The dataframe.
        """
        ...

    @overload
    def __getitem__(self, key: BoolArray) -> "UnitedDataframe[CK]":
        """
        Boolean row filtering.

        Args:
            key (BoolArray): A BoolArray with the same length as the DataFrame.

        Returns:
            UnitedDataframe[CK]: The dataframe.
        """
        ...

    @overload
    def __getitem__(self, key: np.ndarray) -> "UnitedDataframe[CK]":
        """
        Boolean row filtering.

        Args:
            key (np.ndarray[np.bool_]): A boolean numpy array with the same length as the DataFrame.

        Returns:
            UnitedDataframe[CK]: The dataframe.
        """
        ...

    ################### Implementation #########################################################

    def __getitem__(self, key: Union[ # type: ignore[no-any-return]
        CK,
        int,
        slice,
        Sequence[CK],
        tuple[CK, int],
        tuple[int, CK],
        tuple[CK, slice],
        tuple[slice, CK],
        tuple[int, Sequence[CK]],
        tuple[Sequence[CK], int],
        tuple[Sequence[CK], slice],
        tuple[slice, Sequence[CK]],
        pd.Series, # type: ignore[no-any-return]
        BoolArray,
        np.ndarray,
        ]
    ) -> Union[VALUE_TYPE, "ColumnAccessor[CK]", "RowAccessor[CK]", "UnitedDataframe[CK]"]:
        """
        Tuple indexing (e.g., df[rows, columns]).

        Args:
            key (tuple[slice, slice] | tuple[slice, sequence[str]] | tuple[sequence[str], slice] | tuple[sequence[str], sequence[str]]): A tuple of indexing arguments.

        Returns:
            PYTHON_SCALAR_TYPE | _ColumnAccessor[CK] | _RowAccessor[CK] | UnitedDataframe[CK]: The result of the indexing.
        """

        def _ensure_column_keys(raw: Sequence[Any]) -> Sequence[CK]:
            keys: Sequence[CK] = []
            for key in raw:
                if not isinstance(key, (ColumnKey, str)):
                    raise ValueError(f"Invalid column key: {key}")
                keys.append(key) # type: ignore
            return keys
        
        def _ensure_valid_slice(s: slice) -> slice:
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else self._number_of_rows()
            step = s.step if s.step is not None else 1

            if not isinstance(start, int) or not isinstance(stop, int) or not isinstance(step, int):
                raise ValueError(f"Invalid slice values: {s}")

            start = max(0, min(self._number_of_rows(), start))
            stop = max(0, min(self._number_of_rows(), stop))

            return slice(start, stop, step)

        with self._rlock:

            if isinstance(key, ColumnKey|str):
                column_key: CK = key # type: ignore
                return self._column_get_as_column_accessor(column_key)
            elif isinstance(key, int):
                return self._row_get_as_row_accessor(key)
            elif isinstance(key, slice):
                return self._crop_dataframe(row_indices=key)
            elif isinstance(key, Sequence):
                column_keys: Sequence[CK] = _ensure_column_keys(key) # type: ignore
                return self._crop_dataframe(column_keys=column_keys)
            elif isinstance(key, tuple):

                key_0: Any = key[0]
                key_1: Any = key[1]

                if isinstance(key_0, ColumnKey|str) and isinstance(key_1, int):
                    column_key: CK = key_0 # type: ignore
                    return self._cell_get_lowlevel_value(key_1, column_key)
                elif isinstance(key_0, int) and isinstance(key_1, ColumnKey|str):
                    column_key: CK = key_1 # type: ignore
                    return self._cell_get_lowlevel_value(key_0, column_key)
                elif isinstance(key_0, ColumnKey|str) and isinstance(key_1, slice):
                    column_key: CK = key_0 # type: ignore
                    _slice: slice = _ensure_valid_slice(key_1)
                    return self._column_get_as_column_accessor(column_key, _slice)
                elif isinstance(key_0, slice) and isinstance(key_1, ColumnKey|str):
                    _slice: slice = _ensure_valid_slice(key_0)
                    column_key: CK = key_1 # type: ignore
                    return self._column_get_as_column_accessor(column_key, _slice)
                elif isinstance(key_0, int) and isinstance(key_1, Sequence):
                    column_keys: Sequence[CK] = _ensure_column_keys(key_1) # type: ignore
                    return self._row_get_as_row_accessor(key_0, column_keys)
                elif isinstance(key_0, Sequence) and isinstance(key_1, int):
                    column_keys: Sequence[CK] = _ensure_column_keys(key_0)
                    return self._row_get_as_row_accessor(key_1, column_keys)
                elif isinstance(key_0, Sequence) and isinstance(key_1, slice):
                    column_keys: Sequence[CK] = _ensure_column_keys(key_0)
                    return self._crop_dataframe(column_keys, key_1)
                elif isinstance(key_0, slice) and isinstance(key_1, Sequence):
                    column_keys: Sequence[CK] = _ensure_column_keys(key_1) # type: ignore
                    _slice: slice = _ensure_valid_slice(key_0)
                    return self._crop_dataframe(column_keys, _slice)
                else:
                    raise ValueError(f"Invalid key: {key}")
                
            elif isinstance(key, pd.Series):
                # Get sequence of row indices based on boolean Series
                if len(key) != self._number_of_rows(): # type: ignore[arg-type]
                    raise ValueError(f"Boolean Series must have the same length as the number of rows in the DataFrame. Got {len(key)} rows, expected {self._number_of_rows()}.") # type: ignore[arg-type]
                row_indices: Sequence[int] = []
                for i in range(self._number_of_rows()):
                    if key[i]:
                        row_indices.append(i)
                return self._crop_dataframe(row_indices=row_indices)
            elif isinstance(key, BoolArray):
                if len(key) != self._number_of_rows():
                    raise ValueError(f"BoolArray must have the same length as the number of rows in the DataFrame. Got {len(key)} rows, expected {self._number_of_rows()}.")
                row_indices: Sequence[int] = []
                for i in range(self._number_of_rows()):
                    if key[i]:
                        row_indices.append(i)
                return self._crop_dataframe(row_indices=row_indices)
            else:
                raise ValueError(f"Invalid key: {key}")
