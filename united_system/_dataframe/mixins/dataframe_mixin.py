"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the core access patterns for the dataframe including
iteration, indexing, and item assignment through magic methods.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, Sequence, Optional, Tuple
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._units_and_dimension.unit import Unit
import pandas as pd

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe


class DataframeMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Mixin providing dataframe methods.
    
    This mixin implements:
    - contains_nan: Check if the dataframe contains any missing values
    - contains_inf: Check if the dataframe contains any infinite values
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    def _dataframe_get_as_pd_dataframe(self, column_keys: Sequence[CK|Tuple[CK, Unit]]|dict[CK, Optional[Unit]]|None = None, deepcopy: bool = True) -> pd.DataFrame:
        """
        Internal: Get the dataframe as a pandas dataframe. (no lock)

        Args:
            column_keys (Sequence[CK|Tuple[CK, Unit]]|dict[CK, Optional[Unit]]|None): The column keys to get. If None, all columns are returned. If a dict or tuple in the sequence, the keys are the column keys to be returned and values are the units of the columns to be returned.
            deepcopy (bool): Whether to copy the dataframe. Highly recommended!

        Returns:
            pd.DataFrame: The dataframe
        """

        internal_dataframe_column_names_to_return: dict[CK, str] = {}
        units_to_convert_with: dict[CK, Unit] = {}
        if column_keys is None:
            internal_dataframe_column_names_to_return = {column_key: self._internal_dataframe_column_names[column_key] for column_key in self._column_keys}
        elif isinstance(column_keys, dict):
            internal_dataframe_column_names_to_return = {column_key: self._internal_dataframe_column_names[column_key] for column_key in column_keys.keys()}
            for column_key in column_keys.keys():
                unit: Optional[Unit] = column_keys[column_key]
                if unit is not None:
                    units_to_convert_with[column_key] = unit
        else:
            for column_key in column_keys:
                if isinstance(column_key, tuple):
                    internal_dataframe_column_names_to_return[column_key[0]] = self._internal_dataframe_column_names[column_key[0]]
                    units_to_convert_with[column_key[0]] = column_key[1]
                else:
                    internal_dataframe_column_names_to_return[column_key] = self._internal_dataframe_column_names[column_key]

        dataframe_to_return: pd.DataFrame = self._internal_dataframe[list(internal_dataframe_column_names_to_return.values())].copy(deep=True) if deepcopy else self._internal_dataframe[list(internal_dataframe_column_names_to_return.values())]

        # Coverts the columns based on the units and adjust the column names
        for column_key, unit in units_to_convert_with.items():
            current_internal_column_name: str = internal_dataframe_column_names_to_return[column_key]
            new_internal_column_name: str = self._internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, unit)
            dataframe_to_return[new_internal_column_name] = unit.convert(dataframe_to_return[current_internal_column_name], self._unit_get(column_key), unit) # type: ignore
            dataframe_to_return.drop(columns=[current_internal_column_name], inplace=True)

        return dataframe_to_return
    
    def dataframe_get_as_pd_dataframe(self, column_keys: Sequence[CK|Tuple[CK, Unit]]|dict[CK, Optional[Unit]]|None = None, deepcopy: bool = True) -> pd.DataFrame:
        """
        Get the dataframe as a pandas dataframe.

        Args:
            column_keys (Sequence[CK|Tuple[CK, Unit]]|dict[CK, Optional[Unit]]|None): The column keys to get. If None, all columns are returned. If a dict or tuple in the sequence, the keys are the column keys to be returned and values are the units of the columns to be returned.
            deepcopy (bool): Whether to copy the dataframe. Highly recommended!

        Returns:
            pd.DataFrame: The dataframe
        """

        with self._rlock:
            return self._dataframe_get_as_pd_dataframe(column_keys, deepcopy)

    def dataframe_contains_nan(self) -> bool:
        """
        Check if the dataframe contains any missing values
        """
        return self._internal_dataframe.isna().any().any() # type: ignore
    
    def dataframe_contains_inf(self) -> bool:
        """
        Check if the dataframe contains any infinite values
        """
        return self._internal_dataframe.isin([np.inf, -np.inf]).any().any() # type: ignore