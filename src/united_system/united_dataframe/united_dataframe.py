"""
Modular UnitedDataframe Implementation

This is the main UnitedDataframe class that combines all functionality through mixins.
The class itself is now very lightweight and focuses on composition rather than 
implementing everything directly.
"""

from typing import Generic, TypeVar
import pandas as pd
from h5py._hl.group import Group

# Import all mixins
from .core.base import UnitedDataframeCore
from .core.validation import ValidationMixin
# TODO: Import other mixins as they are created
# from .operations.column_ops import ColumnOperationsMixin
# from .operations.row_ops import RowOperationsMixin
# from .operations.cell_ops import CellOperationsMixin
# from .operations.mask_filter import MaskFilterMixin
# from .serialization.serialization import SerializationMixin
# from .serialization.constructors import ConstructorMixin

from .legacy.utils import ColumnKey, ColumnInformation, InternalDataFrameNameFormatter, SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
from .legacy.column_type import ColumnType

CK = TypeVar("CK", bound=ColumnKey|str, default=str)


class UnitedDataframe(
    UnitedDataframeCore[CK],
    ValidationMixin[CK],
    # TODO: Add other mixins as they are created
    # ColumnOperationsMixin[CK],
    # RowOperationsMixin[CK],
    # CellOperationsMixin[CK],
    # MaskFilterMixin[CK],
    # SerializationMixin[CK],
    # ConstructorMixin[CK],
    Generic[CK]
):
    """
    A unit-aware DataFrame with modular architecture.
    
    This class combines functionality from multiple mixins to provide a complete
    dataframe implementation while maintaining clean separation of concerns.
    
    The class is composed of:
    - UnitedDataframeCore: Essential state and infrastructure
    - ValidationMixin: Validation and compatibility checking
    - ColumnOperationsMixin: Column operations and statistics
    - RowOperationsMixin: Row operations and manipulation
    - CellOperationsMixin: Individual cell operations
    - MaskFilterMixin: Filtering and masking operations
    - SerializationMixin: JSON/HDF5 serialization
    - ConstructorMixin: Factory methods and constructors
    
    Each mixin is focused on a specific area of functionality, making the code
    more maintainable and testable.
    """
    
    def __init__(self,
                 internal_canonical_dataframe: pd.DataFrame,
                 column_information: dict[CK, ColumnInformation],
                 internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter[CK] = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER):
        """
        Initialize the UnitedDataframe.
        
        Args:
            internal_canonical_dataframe: The underlying pandas DataFrame
            column_information: Metadata about each column
            internal_dataframe_column_name_formatter: Function to format internal column names
        """
        # Initialize the core - this handles all the heavy lifting
        super().__init__(
            internal_canonical_dataframe,
            column_information,
            internal_dataframe_column_name_formatter
        )
    
    def __repr__(self) -> str:
        """Return a string representation of the dataframe."""
        return f"UnitedDataframe({self.rows} rows × {self.cols} columns)"
    
    def __str__(self) -> str:
        """Return a string representation of the dataframe."""
        return self.__repr__()
    
    # TODO: As we add more mixins, we may need to add some integration methods here
    # For now, all functionality is handled by the mixins themselves
    
    # Example of how to add improved naming methods that delegate to existing methods:
    
    # New naming convention methods (these will be moved to appropriate mixins)
    def column_count(self) -> int:
        """Get the number of columns (improved naming)."""
        return self.cols
    
    def row_count(self) -> int:
        """Get the number of rows (improved naming)."""
        return self.rows
    
    # TODO: Add more improved naming methods as we extract the mixins
    
    def info_summary(self) -> str:
        """Get a quick summary of the dataframe structure."""
        return (
            f"UnitedDataframe Summary:\n"
            f"  Rows: {self.rows}\n"
            f"  Columns: {self.cols}\n"
            f"  Numeric columns: {len(self.get_numeric_column_keys())}\n"
            f"  Read-only: {self.is_read_only()}\n"
            f"  Empty: {self.empty}"
        )
