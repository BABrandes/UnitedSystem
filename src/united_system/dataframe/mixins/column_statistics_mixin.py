"""
Column statistics operations mixin for UnitedDataframe.

Contains all statistical operations for columns, including min, max, mean,
standard deviation, and other statistical measures.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Union
from .dataframe_protocol import UnitedDataframeMixin, CK
from ...real_united_scalar import RealUnitedScalar
from ...complex_united_scalar import ComplexUnitedScalar
from ...int_array import IntArray
from ...float_array import FloatArray
from ...real_united_array import RealUnitedArray
from ...complex_united_array import ComplexUnitedArray

class ColumnStatisticsMixin(UnitedDataframeMixin[CK]):
    """
    Column statistics operations mixin for UnitedDataframe.
    
    Provides all statistical operations for columns, including min, max, mean,
    standard deviation, and other statistical measures.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Column Statistics: Min/Max ------------

    def column_get_min(self, column_key: CK) -> RealUnitedScalar:
        """
        Get the minimum value in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            RealUnitedScalar: The minimum value
        """
        with self._rlock:  # Full IDE support!
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'min'):
                return column_data.min()
            else:
                raise ValueError(f"Column {column_key} does not support min operation.")

    def column_get_max(self, column_key: CK) -> RealUnitedScalar:
        """
        Get the maximum value in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            RealUnitedScalar: The maximum value
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'max'):
                return column_data.max()
            else:
                raise ValueError(f"Column {column_key} does not support max operation.")

    # ----------- Column Statistics: Mean/Standard Deviation ------------

    def column_get_mean(self, column_key: CK) -> Union[RealUnitedScalar, ComplexUnitedScalar]:
        """
        Get the mean value in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            Union[RealUnitedScalar, ComplexUnitedScalar]: The mean value
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'mean'):
                return column_data.mean()
            else:
                raise ValueError(f"Column {column_key} does not support mean operation.")

    def column_get_std(self, column_key: CK) -> RealUnitedScalar:
        """
        Get the standard deviation in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            RealUnitedScalar: The standard deviation
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'std'):
                return column_data.std()
            else:
                raise ValueError(f"Column {column_key} does not support std operation.")

    # ----------- Column Statistics: Sum/Product ------------

    def column_get_sum(self, column_key: CK) -> Union[RealUnitedScalar, ComplexUnitedScalar]:
        """
        Get the sum of values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            Union[RealUnitedScalar, ComplexUnitedScalar]: The sum
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'sum'):
                return column_data.sum()
            else:
                raise ValueError(f"Column {column_key} does not support sum operation.")

    def column_get_product(self, column_key: CK) -> Union[RealUnitedScalar, ComplexUnitedScalar]:
        """
        Get the product of values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            Union[RealUnitedScalar, ComplexUnitedScalar]: The product
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'product'):
                return column_data.product()
            else:
                raise ValueError(f"Column {column_key} does not support product operation.")

    # ----------- Column Statistics: Count/Variance ------------

    def column_get_count(self, column_key: CK) -> int:
        """
        Get the count of non-missing values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            int: The count of non-missing values
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'count'):
                return column_data.count()
            else:
                return len(column_data)

    def column_get_variance(self, column_key: CK) -> RealUnitedScalar:
        """
        Get the variance in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            RealUnitedScalar: The variance
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'variance'):
                return column_data.variance()
            else:
                raise ValueError(f"Column {column_key} does not support variance operation.")

    # ----------- Column Statistics: Median/Quantile ------------

    def column_get_median(self, column_key: CK) -> RealUnitedScalar:
        """
        Get the median value in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            RealUnitedScalar: The median value
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'median'):
                return column_data.median()
            else:
                raise ValueError(f"Column {column_key} does not support median operation.")

    def column_get_quantile(self, column_key: CK, quantile: float) -> RealUnitedScalar:
        """
        Get a quantile value in a column.
        
        Args:
            column_key (CK): The column key
            quantile (float): The quantile (0.0 to 1.0)
            
        Returns:
            RealUnitedScalar: The quantile value
        """
        with self._rlock:
            column_data = self.get_column(column_key)
            if hasattr(column_data, 'quantile'):
                return column_data.quantile(quantile)
            else:
                raise ValueError(f"Column {column_key} does not support quantile operation.") 