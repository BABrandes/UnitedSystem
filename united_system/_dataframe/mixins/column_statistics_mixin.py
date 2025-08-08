"""
Column statistics operations mixin for UnitedDataframe.

Contains all statistical operations for columns, including min, max, mean,
standard deviation, and other statistical measures.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, overload, TypeVar, Any
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._utils.scalar_type import NUMERIC_SCALAR_TYPE, SCALAR_TYPE

NST = TypeVar("NST", bound=NUMERIC_SCALAR_TYPE)
ST = TypeVar("ST", bound=SCALAR_TYPE)

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class ColumnStatisticsMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Column statistics mixin for UnitedDataframe.
    
    Provides all functionality related to column statistics operations.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Column Statistics: Min/Max ------------

    def _helper_return_numeric_scalar(self, raw_value: Any, column_key: CK, expected_type: type[NST]|None) -> NUMERIC_SCALAR_TYPE|NST:
        if expected_type is not None:
            if not self._column_types[column_key].check_type_compatibility(expected_type, "scalar"):
                raise ValueError(f"Column {column_key} is not a {expected_type} column.")
            result: NST = self._column_types[column_key].get_scalar_value_from_dataframe(raw_value, self._column_units[column_key]) # type: ignore
        else:
            result: NUMERIC_SCALAR_TYPE = self._column_types[column_key].get_scalar_value_from_dataframe(raw_value, self._column_units[column_key]) # type: ignore
        return result # type: ignore[no-any-return]
    
    def _helper_return_scalar(self, raw_value: Any, column_key: CK, expected_type: type[ST]|None) -> SCALAR_TYPE|ST:
        if expected_type is not None:
            if not self._column_types[column_key].check_type_compatibility(expected_type, "scalar"):
                raise ValueError(f"Column {column_key} is not a {expected_type} column.")
            result: ST = self._column_types[column_key].get_scalar_value_from_dataframe(raw_value, self._column_units[column_key]) # type: ignore
        else:
            result: SCALAR_TYPE = self._column_types[column_key].get_scalar_value_from_dataframe(raw_value, self._column_units[column_key]) # type: ignore
        return result # type: ignore[no-any-return]

    @overload
    def column_get_min(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_min(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_min(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the minimum value in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the minimum value
            
        Returns:
            NUMERIC_SCALAR_TYPE: The minimum value
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).min()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    @overload
    def column_get_max(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_max(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_max(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the maximum value in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the maximum value
            
        Returns:
            NUMERIC_SCALAR_TYPE: The maximum value
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).max()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)
        
    @overload
    def column_get_unique(self, column_key: CK) -> list[SCALAR_TYPE]: ...
    @overload
    def column_get_unique(self, column_key: CK, expected_type: type[ST]) -> list[ST]: ...
    def column_get_unique(self, column_key: CK, expected_type: type[ST]|None = None) -> list[ST]|list[SCALAR_TYPE]:
        """
        Get the unique values in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[ST]): The expected type of the unique values
            
        Returns:
            list[ST]: The unique values
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_unique_values: list[Any] = self._column_get_as_pd_series(column_key).unique()  # type: ignore
            
            if expected_type is not None:
                unique_values_of_known_type: list[ST] = []
                for value in raw_unique_values:
                    scalar_value: ST = self._helper_return_scalar(value, column_key, expected_type) # type: ignore
                    unique_values_of_known_type.append(scalar_value) # type: ignore
                return unique_values_of_known_type
            else:
                unique_values_of_unknown_type: list[SCALAR_TYPE] = []
                for value in raw_unique_values:
                    scalar_value: SCALAR_TYPE = self._helper_return_scalar(value, column_key, None)
                    unique_values_of_unknown_type.append(scalar_value)
                return unique_values_of_unknown_type

    # ----------- Column Statistics: Mean/Standard Deviation ------------

    @overload
    def column_get_mean(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_mean(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_mean(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the mean value in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the mean value

        Returns:
            NUMERIC_SCALAR_TYPE: The mean value
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).mean()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    @overload
    def column_get_std(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_std(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_std(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the standard deviation in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the standard deviation value

        Returns:
            NUMERIC_SCALAR_TYPE: The standard deviation
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).std()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    # ----------- Column Statistics: Sum/Product ------------

    @overload
    def column_get_sum(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_sum(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_sum(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the sum of values in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the sum value
        Returns:
            NUMERIC_SCALAR_TYPE: The sum
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).sum()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    @overload
    def column_get_product(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_product(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_product(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the product of values in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the product value
            
        Returns:
            NUMERIC_SCALAR_TYPE: The product
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).product()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    # ----------- Column Statistics: Count/Variance ------------

    def column_count_non_missing_values(self, column_key: CK) -> int:
        """
        Get the count of non-missing values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            int: The count of non-missing values
        """
        with self._rlock:
            return self._column_get_as_pd_series(column_key).count() # type: ignore
        
    def column_count_missing_values(self, column_key: CK) -> int:
        """
        Get the count of missing values in a column.
        """
        with self._rlock:
            return self._internal_dataframe[column_key].isna().sum() # type: ignore

    @overload
    def column_get_variance(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_variance(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_variance(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the variance in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the variance value

        Returns:
            NUMERIC_SCALAR_TYPE: The variance
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).var()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    # ----------- Column Statistics: Median/Quantile ------------

    @overload
    def column_get_median(self, column_key: CK) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_median(self, column_key: CK, expected_type: type[NST]) -> NST: ...
    def column_get_median(self, column_key: CK, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get the median value in a column.
        
        Args:
            column_key (CK): The column key
            expected_type (type[NST]): The expected type of the median value

        Returns:
            RealUnitedScalar: The median value
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).median()  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)

    @overload
    def column_get_quantile(self, column_key: CK, quantile: float) -> NUMERIC_SCALAR_TYPE: ...
    @overload
    def column_get_quantile(self, column_key: CK, quantile: float, expected_type: type[NST]) -> NST: ...
    def column_get_quantile(self, column_key: CK, quantile: float, expected_type: type[NST]|None = None) -> NST|NUMERIC_SCALAR_TYPE:
        """
        Get a quantile value in a column.
        
        Args:
            column_key (CK): The column key
            quantile (float): The quantile (0.0 to 1.0)
            expected_type (type[NST]): The expected type of the quantile value
            
        Returns:
            RealUnitedScalar: The quantile value
        """
        with self._rlock:
            if not self._colkey_is_numeric(column_key):
                raise ValueError(f"Column {column_key} is not a numeric column.")
            raw_value: Any = self._column_get_as_pd_series(column_key).quantile(quantile)  # type: ignore
            return self._helper_return_numeric_scalar(raw_value, column_key, expected_type)
        
    def column_count_missing_per_values(self) -> dict[CK, int]:
        """
        Count the number of missing values in each column.
        
        Returns:
            Mapping[CK, int]: Dictionary mapping column keys to number of missing values
        """
        with self._rlock:
            return {column_key: self._internal_dataframe[column_key].isna().sum() for column_key in self._column_keys} # type: ignore