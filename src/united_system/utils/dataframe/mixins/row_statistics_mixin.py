"""
Row statistics mixin for UnitedDataframe.

Contains all operations related to row statistics, including
counting missing values.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from .dataframe_protocol import UnitedDataframeProtocol, CK

class RowStatisticsMixin(UnitedDataframeProtocol[CK]):

    def rows_count_missing_values(self) -> dict[int, int]:
        """
        Count the number of missing values in each row.
        
        Returns:
            dict[int, int]: Dictionary mapping row indices to number of missing values per row
        """
        with self._rlock:
            return {row_index: self._internal_dataframe.iloc[row_index].isna().sum() for row_index in range(self._number_of_rows())} # type: ignore
