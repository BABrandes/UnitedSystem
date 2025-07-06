from dataclasses import dataclass, field
from typing import Union, Optional, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import h5py
from pandas import Timestamp
from .arrays.timestamp_array import TimestampArray as InternalTimestampArray
from .utils import JSONable, HDF5able
from .bool_array import BoolArray

@dataclass(frozen=True, slots=True, init=False)
class TimestampArray(JSONable, HDF5able):
    """User-friendly wrapper for timestamp arrays with date/time operations."""

    _wrapped_array: InternalTimestampArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray, list, InternalTimestampArray, pd.DatetimeIndex]):
        """
        Create a TimestampArray from various input formats.
        
        Examples:
            TimestampArray(['2023-01-01', '2023-01-02'])    # List of date strings
            TimestampArray(pd.date_range('2023-01-01', periods=5))  # DatetimeIndex
            TimestampArray(existing_array)                  # Wrap existing InternalTimestampArray
        """
        
        match values:
            case InternalTimestampArray():
                object.__setattr__(self, "_wrapped_array", values)
            case pd.DatetimeIndex():
                timestamp_values = values.to_numpy()
                object.__setattr__(self, "_wrapped_array", InternalTimestampArray.create(timestamp_values))
            case np.ndarray() | list():
                if isinstance(values, list):
                    # Convert to pandas timestamps first
                    values = pd.to_datetime(values).to_numpy()
                elif isinstance(values, np.ndarray):
                    if values.dtype.kind == 'M':  # datetime64 type
                        values = pd.to_datetime(values).to_numpy()
                    else:
                        values = pd.to_datetime(values).to_numpy()
                object.__setattr__(self, "_wrapped_array", InternalTimestampArray.create(values))
            case _:
                raise ValueError(f"Invalid values type: {type(values)}")

    # Properties
    @property
    def values(self) -> np.ndarray:
        """Get the raw numpy array values."""
        return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the array."""
        return self._wrapped_array.shape
    
    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self._wrapped_array.size

    @property
    def year(self) -> np.ndarray:
        """Get the year component."""
        return pd.to_datetime(self.values).year.values

    @property
    def month(self) -> np.ndarray:
        """Get the month component."""
        return pd.to_datetime(self.values).month.values

    @property
    def day(self) -> np.ndarray:
        """Get the day component."""
        return pd.to_datetime(self.values).day.values

    @property
    def hour(self) -> np.ndarray:
        """Get the hour component."""
        return pd.to_datetime(self.values).hour.values

    @property
    def minute(self) -> np.ndarray:
        """Get the minute component."""
        return pd.to_datetime(self.values).minute.values

    @property
    def second(self) -> np.ndarray:
        """Get the second component."""
        return pd.to_datetime(self.values).second.values

    @property
    def weekday(self) -> np.ndarray:
        """Get the weekday (0=Monday, 6=Sunday)."""
        return pd.to_datetime(self.values).weekday.values

    # Array access
    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[Timestamp, "TimestampArray"]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, (Timestamp, np.datetime64)):
            return Timestamp(result)
        elif isinstance(result, (int, np.integer)):
            # Handle raw integer timestamp values from get_scalar
            return Timestamp(result)
        else:
            return TimestampArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)
    
    def __contains__(self, item: Union[Timestamp, str, datetime]) -> bool:
        """Check if an item is in the array."""
        if isinstance(item, (str, datetime)):
            item = pd.Timestamp(item)
        return item in self._wrapped_array

    # Iteration
    def __iter__(self):
        """Iterate over the array elements."""
        return iter(self._wrapped_array)

    # Arithmetic operations with timedelta
    def __add__(self, other: Union[timedelta, pd.Timedelta, np.timedelta64]) -> "TimestampArray":
        """Add a time duration to all timestamps."""
        if isinstance(other, (timedelta, pd.Timedelta, np.timedelta64)):
            # Convert to pandas timedelta for proper handling
            pd_other = pd.Timedelta(other)
            result = pd.to_datetime(self.values) + pd_other
            return TimestampArray(result.to_numpy())
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other: Union[timedelta, pd.Timedelta, np.timedelta64, "TimestampArray", Timestamp]) -> "TimestampArray":
        """Subtract a time duration or timestamp."""
        if isinstance(other, (timedelta, pd.Timedelta, np.timedelta64)):
            pd_other = pd.Timedelta(other)
            result = pd.to_datetime(self.values) - pd_other
            return TimestampArray(result.to_numpy())
        elif isinstance(other, TimestampArray):
            # Returns duration array
            result = pd.to_datetime(self.values) - pd.to_datetime(other.values)
            return TimestampArray(result.to_numpy())
        elif isinstance(other, Timestamp):
            result = pd.to_datetime(self.values) - pd.Timestamp(other)
            return TimestampArray(result.to_numpy())
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    # Comparison operations
    def __eq__(self, other: Union["TimestampArray", Timestamp, str, datetime]) -> "BoolArray":
        """Element-wise equality comparison."""
        from .bool_array import BoolArray
        if isinstance(other, TimestampArray):
            result = self.values == other.values
        elif isinstance(other, (Timestamp, str, datetime)):
            if isinstance(other, (str, datetime)):
                other = pd.Timestamp(other)
            result = self.values == other.to_numpy()
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __ne__(self, other: Union["TimestampArray", Timestamp, str, datetime]) -> "BoolArray":
        """Element-wise inequality comparison."""
        from .bool_array import BoolArray
        if isinstance(other, TimestampArray):
            result = self.values != other.values
        elif isinstance(other, (Timestamp, str, datetime)):
            if isinstance(other, (str, datetime)):
                other = pd.Timestamp(other)
            result = self.values != other.to_numpy()
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __lt__(self, other: Union["TimestampArray", Timestamp, str, datetime]) -> "BoolArray":
        """Element-wise less than comparison."""
        from .bool_array import BoolArray
        if isinstance(other, TimestampArray):
            result = self.values < other.values
        elif isinstance(other, (Timestamp, str, datetime)):
            if isinstance(other, (str, datetime)):
                other = pd.Timestamp(other)
            result = self.values < other.to_numpy()
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __le__(self, other: Union["TimestampArray", Timestamp, str, datetime]) -> "BoolArray":
        """Element-wise less than or equal comparison."""
        from .bool_array import BoolArray
        if isinstance(other, TimestampArray):
            result = self.values <= other.values
        elif isinstance(other, (Timestamp, str, datetime)):
            if isinstance(other, (str, datetime)):
                other = pd.Timestamp(other)
            result = self.values <= other.to_numpy()
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __gt__(self, other: Union["TimestampArray", Timestamp, str, datetime]) -> "BoolArray":
        """Element-wise greater than comparison."""
        from .bool_array import BoolArray
        if isinstance(other, TimestampArray):
            result = self.values > other.values
        elif isinstance(other, (Timestamp, str, datetime)):
            if isinstance(other, (str, datetime)):
                other = pd.Timestamp(other)
            result = self.values > other.to_numpy()
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __ge__(self, other: Union["TimestampArray", Timestamp, str, datetime]) -> "BoolArray":
        """Element-wise greater than or equal comparison."""
        from .bool_array import BoolArray
        if isinstance(other, TimestampArray):
            result = self.values >= other.values
        elif isinstance(other, (Timestamp, str, datetime)):
            if isinstance(other, (str, datetime)):
                other = pd.Timestamp(other)
            result = self.values >= other.to_numpy()
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    # Date/time specific operations
    def floor(self, freq: str = 'D') -> "TimestampArray":
        """Floor to specified frequency (e.g., 'D' for day, 'H' for hour)."""
        result = pd.to_datetime(self.values).floor(freq)
        return TimestampArray(result.to_numpy())

    def ceil(self, freq: str = 'D') -> "TimestampArray":
        """Ceiling to specified frequency."""
        result = pd.to_datetime(self.values).ceil(freq)
        return TimestampArray(result.to_numpy())

    def round(self, freq: str = 'D') -> "TimestampArray":
        """Round to specified frequency."""
        result = pd.to_datetime(self.values).round(freq)
        return TimestampArray(result.to_numpy())

    def strftime(self, format: str) -> List[str]:
        """Format timestamps as strings."""
        return pd.to_datetime(self.values).strftime(format).tolist()

    def normalize(self) -> "TimestampArray":
        """Normalize to midnight (remove time component)."""
        result = pd.to_datetime(self.values).normalize()
        return TimestampArray(result.to_numpy())

    # Reduction methods
    def min(self) -> Timestamp:
        """Minimum timestamp."""
        return Timestamp(np.min(self.values))

    def max(self) -> Timestamp:
        """Maximum timestamp."""
        return Timestamp(np.max(self.values))

    def range(self) -> timedelta:
        """Range (max - min) of timestamps."""
        return self.max() - self.min()

    # Utility methods
    def unique(self) -> "TimestampArray":
        """Get unique timestamps in the array."""
        unique_values = np.unique(self.values)
        return TimestampArray(unique_values)

    def sort(self, ascending: bool = True) -> "TimestampArray":
        """Sort the array."""
        sorted_array = np.sort(self.values)
        if not ascending:
            sorted_array = sorted_array[::-1]
        return TimestampArray(sorted_array)

    def concatenate(self, other: "TimestampArray") -> "TimestampArray":
        """Concatenate with another TimestampArray."""
        concatenated = np.concatenate([self.values, other.values])
        return TimestampArray(concatenated)

    def is_weekend(self) -> "BoolArray":
        """Check if dates are weekends."""
        from .bool_array import BoolArray
        weekdays = pd.to_datetime(self.values).weekday
        result = (weekdays == 5) | (weekdays == 6)  # Saturday or Sunday
        return BoolArray(result)

    def is_month_start(self) -> "BoolArray":
        """Check if dates are month starts."""
        from .bool_array import BoolArray
        result = pd.to_datetime(self.values).is_month_start
        return BoolArray(result)

    def is_month_end(self) -> "BoolArray":
        """Check if dates are month ends."""
        from .bool_array import BoolArray
        result = pd.to_datetime(self.values).is_month_end
        return BoolArray(result)

    def is_year_start(self) -> "BoolArray":
        """Check if dates are year starts."""
        from .bool_array import BoolArray
        result = pd.to_datetime(self.values).is_year_start
        return BoolArray(result)

    def is_year_end(self) -> "BoolArray":
        """Check if dates are year ends."""
        from .bool_array import BoolArray
        result = pd.to_datetime(self.values).is_year_end
        return BoolArray(result)

    # Factory methods
    @classmethod
    def from_strings(cls, strings: List[str], format: Optional[str] = None) -> "TimestampArray":
        """Create from list of date strings."""
        if format:
            values = pd.to_datetime(strings, format=format)
        else:
            values = pd.to_datetime(strings)
        return cls(values.to_numpy())

    @classmethod
    def date_range(cls, start: Union[str, datetime], end: Union[str, datetime], freq: str = 'D') -> "TimestampArray":
        """Create a date range."""
        values = pd.date_range(start=start, end=end, freq=freq)
        return cls(values.to_numpy())

    @classmethod
    def date_range_periods(cls, start: Union[str, datetime], periods: int, freq: str = 'D') -> "TimestampArray":
        """Create a date range with specified number of periods."""
        values = pd.date_range(start=start, periods=periods, freq=freq)
        return cls(values.to_numpy())

    @classmethod
    def now(cls, periods: int = 1) -> "TimestampArray":
        """Create array of current timestamps."""
        values = [pd.Timestamp.now()] * periods
        return cls(values)

    @classmethod
    def today(cls, periods: int = 1) -> "TimestampArray":
        """Create array of today's dates."""
        values = [pd.Timestamp.today().normalize()] * periods
        return cls(values)

    @classmethod
    def from_values(cls, values: Union[np.ndarray, list]) -> "TimestampArray":
        """Create from values."""
        return cls(values)

    # String representation
    def __str__(self) -> str:
        """Simple string representation."""
        return str(self.values)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"TimestampArray({self.values.tolist()!r})"

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "TimestampArray":
        """Create from JSON dictionary."""
        return cls(InternalTimestampArray.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "TimestampArray":
        """Create from HDF5 group."""
        return cls(InternalTimestampArray.from_hdf5(hdf5_group)) 