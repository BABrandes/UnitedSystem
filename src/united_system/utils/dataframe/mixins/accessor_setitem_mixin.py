"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the core access patterns for the dataframe including
iteration, indexing, and item assignment through magic methods.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import overload, TYPE_CHECKING, Any, Union
from collections.abc import Sequence
from ..column_key import ColumnKey
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..column_type import LOWLEVEL_TYPE, ARRAY_TYPE, SCALAR_TYPE
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe
    from ..accessors._row_accessor import RowAccessor # type: ignore
    from ..accessors._column_accessor import ColumnAccessor # type: ignore

class AccessorSetitemMixin(UnitedDataframeProtocol[CK]):
    """
    Mixin providing magic methods for dataframe item assignment.
    
    This mixin implements:
    - __setitem__: Cell value assignment using tuple syntax
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    ################### Overloads #########################################################

    @overload
    def __setitem__(self, key: CK, value: ARRAY_TYPE | Sequence[LOWLEVEL_TYPE] | Sequence[SCALAR_TYPE] | np.ndarray | pd.Series[Any]) -> None: ...
    @overload
    def __setitem__(self, key: int, value: dict[CK, LOWLEVEL_TYPE] | dict[CK, SCALAR_TYPE]) -> None: ...
    @overload
    def __setitem__(self, key: Sequence[CK], value: ARRAY_TYPE | Sequence[LOWLEVEL_TYPE] | Sequence[SCALAR_TYPE] | np.ndarray | pd.Series[Any]) -> None: ...
    @overload
    def __setitem__(self, key: slice, value: dict[CK, LOWLEVEL_TYPE] | dict[CK, SCALAR_TYPE]) -> None: ...
    @overload
    def __setitem__(self, key: tuple[int, CK], value: LOWLEVEL_TYPE) -> None: ...
    @overload
    def __setitem__(self, key: tuple[CK, int], value: LOWLEVEL_TYPE) -> None: ...
    @overload
    def __setitem__(self, key: tuple[CK, slice], value: ARRAY_TYPE | Sequence[LOWLEVEL_TYPE] | Sequence[SCALAR_TYPE] | np.ndarray | pd.Series[Any]) -> None: ...
    @overload
    def __setitem__(self, key: tuple[slice, CK], value: ARRAY_TYPE | Sequence[LOWLEVEL_TYPE] | Sequence[SCALAR_TYPE] | np.ndarray | pd.Series[Any]) -> None: ...
    @overload
    def __setitem__(self, key: tuple[Sequence[CK], int], value: dict[CK, LOWLEVEL_TYPE] | dict[CK, SCALAR_TYPE]) -> None: ...
    @overload
    def __setitem__(self, key: tuple[int, Sequence[CK]], value: dict[CK, LOWLEVEL_TYPE] | dict[CK, SCALAR_TYPE]) -> None: ...
    @overload
    def __setitem__(self, key: tuple[Sequence[CK], slice], value: UnitedDataframe[CK] | dict[CK, LOWLEVEL_TYPE] | dict[CK, SCALAR_TYPE]) -> None: ...
    @overload
    def __setitem__(self, key: tuple[slice, Sequence[CK]], value: UnitedDataframe[CK] | dict[CK, LOWLEVEL_TYPE] | dict[CK, SCALAR_TYPE]) -> None: ...

    ################### Implementation #########################################################

    def __setitem__(self, key: Union[
        CK,
        int,
        slice,
        Sequence[CK],
        tuple[int, CK],
        tuple[CK, int],
        tuple[slice, CK],
        tuple[CK, slice],
        tuple[Sequence[CK], int],
        tuple[int, Sequence[CK]],
        tuple[Sequence[CK], slice],
        tuple[slice, Sequence[CK]],
        ], value: Any) -> None:
        """
        Set the value of a specific cell.

        Args:
            key (Any): The key to set the value for.
            value (Any): The value to set.

        Raises:
            ValueError: If the key is invalid.
            ValueError: If the value is invalid.
            ValueError: If the key is not a valid column key.
        """
        
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only.")
            
            if isinstance(key, ColumnKey|str):
                raise NotImplementedError("Setting a single cell value is not implemented.")
            elif isinstance(key, int):
                raise NotImplementedError("Setting a row value is not implemented.")
            elif isinstance(key, slice):
                raise NotImplementedError("Setting a slice value is not implemented.")
            else: # Is sequence
                if all(isinstance(k, ColumnKey|str) for k in key):
                    raise NotImplementedError("Setting a sinle cell value is not implemented.")
                else: # Is tuple
                    key_0: Any = key[0]
                    key_1: Any = key[1]

                    if isinstance(key_0, ColumnKey|str) and isinstance(key_1, int):
                        column_key: CK = key_0 # type: ignore
                        self._cell_set_value(key_1, column_key, value)
                    elif isinstance(key_0, int) and isinstance(key_1, ColumnKey|str):
                        column_key: CK = key_1 # type: ignore
                        self._cell_set_value(key_0, column_key, value)
                    elif isinstance(key_0, ColumnKey|str) and isinstance(key_1, slice):
                        raise NotImplementedError("Setting a single cell value is not implemented.")
                    elif isinstance(key_0, slice) and isinstance(key_1, ColumnKey|str):
                        raise NotImplementedError("Setting a single cell value is not implemented.")
                    elif isinstance(key_0, Sequence) and isinstance(key_1, int):
                        raise NotImplementedError("Setting a single cell value is not implemented.")
                    elif isinstance(key_0, int) and isinstance(key_1, Sequence):
                        raise NotImplementedError("Setting a single cell value is not implemented.")
                    elif isinstance(key_0, list) and isinstance(key_1, slice):
                        raise NotImplementedError("Setting a single cell value is not implemented.")
                    elif isinstance(key_0, slice) and isinstance(key_1, list):
                        raise NotImplementedError("Setting a single cell value is not implemented.")
                    else:
                        raise ValueError(f"Invalid key: {key}")
    