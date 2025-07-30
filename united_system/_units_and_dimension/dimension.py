"""
Dimension Module for United System

This module provides the Dimension class for handling physical dimensions in scientific calculations.
Dimensions represent the fundamental units of measurement (mass, length, time, etc.) and their relationships.

Key Features:
- Support for base dimensions: M (mass), T (time), L (length), I (current), Θ (temperature), N (amount), J (luminous intensity), A (angle)
- Subscript support for distinguishing different contexts (e.g., L_elec vs L_geo)
- Logarithmic dimensions using DEC() notation
- Arithmetic operations (multiplication, division, powers, logarithms)
- Serialization to JSON and HDF5 formats
- Immutable design for thread safety

Examples:
    # Create simple dimensions
    mass = Dimension("M")           # Mass dimension
    length = Dimension("L")         # Length dimension
    time = Dimension("T")           # Time dimension
    
    # Create complex dimensions
    velocity = Dimension("L/T")     # Length per time (velocity)
    force = Dimension("M*L/T^2")    # Mass * length / time^2 (force)
    energy = Dimension("M*L^2/T^2") # Mass * length^2 / time^2 (energy)
    
    # Use subscripts to distinguish contexts
    elec_length = Dimension("L_elec")  # Electrical length
    geo_length = Dimension("L_geo")    # Geometric length
    
    # Logarithmic dimensions
    log_force = Dimension("DEC(M*L/T^2)")  # Logarithm of force
    log_ratio = Dimension("DEC(L_elec/L_geo)")  # Logarithm of length ratio
    
    # Arithmetic operations
    area = length * length          # L * L = L^2
    density = mass / (length**3)    # M / L^3
    log_area = area.log()           # DEC(L^2)
    
    # Serialization
    json_str = force.to_json()      # "M*L/T^2"
    reparsed = Dimension.from_json(json_str)
    
    # Check if dimensionless
    dimensionless = Dimension("")   # Empty string creates dimensionless dimension
    assert dimensionless.is_dimensionless
"""

from typing import TYPE_CHECKING, overload, Union, Optional, Tuple, Any
from types import MappingProxyType
from h5py import Group
from .utils import seperate_string
from .dimension_symbol import DimensionSymbol, BASE_DIMENSION_SYMBOLS

if TYPE_CHECKING:
    from .unit import Unit
    from .named_quantity import NamedQuantity
    from .united import United

EPSILON: float = 1e-12

LOG_DIMENSION_SYMBOL_STRING: str =   "DEC"

class Dimension:
    """
    A class representing physical dimensions for scientific calculations.
    
    The Dimension class handles the mathematical representation of physical dimensions,
    supporting base dimensions, subscripts, logarithmic dimensions, and arithmetic operations.
    
    Attributes:
        _proper_exponents: Dictionary mapping subscripts to tuples of dimension exponents
        _log_dimensions: Dictionary mapping log dimensions to their exponents
    
    Examples:
        # Basic usage
        mass = Dimension("M")
        length = Dimension("L")
        time = Dimension("T")
        
        # Complex dimensions
        velocity = Dimension("L/T")
        acceleration = Dimension("L/T^2")
        force = Dimension("M*L/T^2")
        
        # With subscripts
        elec_length = Dimension("L_elec")
        geo_length = Dimension("L_geo")
        
        # Logarithmic dimensions
        log_force = Dimension("DEC(M*L/T^2)")
        log_ratio = Dimension("DEC(L_elec/L_geo)")
        
        # Arithmetic operations
        area = length * length
        volume = area * length
        density = mass / volume
        
        # Powers and roots
        area_squared = area ** 2
        length_cubed = length ** 3
        
        # Logarithms
        log_area = area.log()
        log_volume = volume.log()
        
        # Serialization
        json_str = force.to_json()
        reparsed = Dimension.from_json(json_str)
        
        # Validation
        assert force.is_valid_for_addition(force, force)
        assert not force.is_dimensionless
        assert Dimension("").is_dimensionless
    """

    _CACHE: dict[str, "Dimension"] = {}
    __slots__ = (
        "_proper_exponents",
        "_log_dimensions",
        "_canonical_unit",
    )
    _proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]]
    _log_dimensions: dict["Dimension", float]
    _canonical_unit: "Unit"

################################################################################
# Constructors
################################################################################

    # Public constructors

    @overload
    def __init__(self, value: Optional[str]=None) -> None:
        """
        Initialize a Dimension from a string representation.
        
        Args:
            value: String representation of the dimension (e.g., "M*L/T^2")
        
        Examples:
            mass = Dimension("M")
            force = Dimension("M*L/T^2")
            log_force = Dimension("DEC(M*L/T^2)")
        """
        ...
    
    @overload
    def __init__(self, value: Optional["NamedQuantity"], subscript: Optional[str]=None) -> None:
        """
        Initialize a Dimension from a NamedQuantity with optional subscript.
        
        Args:
            value: NamedQuantity enum value
            subscript: Optional subscript to apply to the entire quantity
        
        Examples:
            force = Dimension(NamedQuantity.FORCE)
            elec_force = Dimension(NamedQuantity.FORCE, subscript="elec")
        """
        ...

    @overload
    def __init__(self, value: "Unit") -> None:
        """
        Initialize a Dimension from a Unit.
        """
        ...
    
    def __init__(self, value: Optional[Union[str, "NamedQuantity", "Unit"]]=None, subscript: Optional[str]=None) -> None:
        """
        Initialize a Dimension object.
        
        Args:
            value: Either a string representation or a NamedQuantity enum value
            subscript: Optional subscript (only used with NamedQuantity)
        
        Raises:
            ValueError: If the value is invalid or None
        
        Examples:
            # String-based construction
            mass = Dimension("M")
            force = Dimension("M*L/T^2")
            log_force = Dimension("DEC(M*L/T^2)")
            
            # NamedQuantity-based construction
            force = Dimension(NamedQuantity.FORCE)
            elec_force = Dimension(NamedQuantity.FORCE, subscript="elec")
            
            # Dimensionless
            dimensionless = Dimension("")
        """
        # Import here to avoid circular import
        from .named_quantity import NamedQuantity
        from .unit import Unit
        
        if isinstance(value, str):
            pass
        elif isinstance(value, NamedQuantity):
            pass  # The actual work is done in __new__
        elif isinstance(value, Unit):
            pass  # The actual work is done in __new__
        else:
            raise ValueError("Invalid value for dimension.")
        
    # Internal constructors

    def __new__(
            cls,
            value: Optional[Union[str, "NamedQuantity", "Unit"]]=None,
            subscript: Optional[str]=None,
        ) -> "Dimension":
        # Import here to avoid circular import
        from .named_quantity import NamedQuantity
        from .unit import Unit
        
        if isinstance(value, str):

            if value in cls._CACHE:
                return cls._CACHE[value]
            else:
                dimension: "Dimension" = cls._parse_string(value)
                cls._CACHE[value] = dimension
                return dimension
            
        elif isinstance(value, NamedQuantity):

            # Create dimension from NamedQuantity
            exponents = value.proper_exponents  # Get the ProperExponents from the NamedQuantity
    
            # Validate subscript if provided
            if subscript is not None:
                if subscript == "":
                    raise ValueError("Invalid subscript: empty string")
                # Check for invalid characters (only alphanumeric and underscore allowed)
                if not all(c.isalnum() or c == '_' for c in subscript):
                    raise ValueError("Invalid subscript: contains invalid characters")
    
            # Create the proper_exponents dict
            proper_exponents_dict = {}
            
            # Create separate entries for each non-zero exponent with the same subscript
            # This matches the behavior of string parsing
            if any(abs(exp) >= EPSILON for exp in [
                exponents.mass, exponents.time, exponents.length,
                exponents.current, exponents.temperature, exponents.amount,
                exponents.luminous_intensity, exponents.angle
            ]):
                subscript_key = subscript if subscript and subscript != "" else ""
                proper_exponents_dict[subscript_key] = (
                    exponents.mass,
                    exponents.time,
                    exponents.length,
                    exponents.current,
                    exponents.temperature,
                    exponents.amount,
                    exponents.luminous_intensity,
                    exponents.angle,
                )
    
            return cls._construct(proper_exponents_dict, {}) # type: ignore
        elif isinstance(value, Unit):

            # Determine the proper exponents for each subscript
            proper_exponents_dict: dict[str, Tuple[float, float, float, float, float, float, float, float]] = {}
            for subscript, unit_elements in value.unit_elements.items():
                # Use the proper_exponents_of_unit_elements function directly to avoid circular dependency
                from .proper_exponents import ProperExponents
                proper_exponents_tuple = ProperExponents.proper_exponents_of_unit_elements(unit_elements)
                proper_exponents_dict[subscript] = proper_exponents_tuple

            # Get the log dimensions
            log_dimensions: dict["Dimension", float] = {}
            for log_unit_element, inner_dimension in value.log_units:
                log_dimensions[inner_dimension] = log_unit_element.exponent
            
            return cls._construct(proper_exponents_dict, log_dimensions) # type: ignore
        
        elif value is None:
            raise ValueError("Invalid value for dimension: None")
        else:
            raise NotImplementedError("Parsing of dimensions is not implemented.")

    @classmethod
    def _construct(
        cls,
        proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]],
        log_dimensions: dict["Dimension", float] = {},
    ) -> "Dimension":
        
        self = super().__new__(cls)

        self._proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]] = MappingProxyType(proper_exponents) # type: ignore
        self._log_dimensions: dict["Dimension", float] = MappingProxyType(log_dimensions) # type: ignore

        return self

################################################################################
# Arithmetic operations
################################################################################

    def __add__(self, other: "Dimension") -> "Dimension":
        """
        Add two dimensions.
        
        For regular dimensions, addition is only valid if the dimensions are identical.
        For logarithmic dimensions, addition follows logarithmic identities.
        
        Args:
            other: The dimension to add to this one
        
        Returns:
            A new Dimension representing the sum
        
        Raises:
            ValueError: If the dimensions cannot be added
        
        Examples:
            # Regular dimensions (must be identical)
            force1 = Dimension("M*L/T^2")
            force2 = Dimension("M*L/T^2")
            total_force = force1 + force2  # Valid
            
            # Logarithmic dimensions
            log_mass = Dimension("DEC(M)")
            log_length = Dimension("DEC(L)")
            log_mass_length = log_mass + log_length  # DEC(M*L)
        """

        if not Dimension.is_valid_for_addition(self, other):
            raise ValueError("Invalid dimensions for addition.")
        
        # If both dimensions have log components, we need to apply logarithmic identities
        if self._log_dimensions and other._log_dimensions:
            # For log dimensions, addition should follow: log(a) + log(b) = log(a*b)
            # We need to combine the underlying dimensions and then take the log
            
            # Start with the proper exponents (should be the same for both dimensions)
            combined_proper_exponents = self._proper_exponents.copy()
            
            # For each log dimension, we need to get its underlying dimension
            # and combine them multiplicatively
            combined_underlying = Dimension._construct(combined_proper_exponents, {})
            
            # Process log dimensions from self
            for log_dim, exponent in self._log_dimensions.items():
                if abs(exponent) >= EPSILON:
                    # The log_dim represents the underlying dimension that was logged
                    # We need to multiply by this underlying dimension
                    combined_underlying = combined_underlying * (log_dim ** exponent)
            
            # Process log dimensions from other
            for log_dim, exponent in other._log_dimensions.items():
                if abs(exponent) >= EPSILON:
                    combined_underlying = combined_underlying * (log_dim ** exponent)
            
            # Now take the log of the combined underlying dimension
            return combined_underlying.log() # type: ignore
        
        # If only one dimension has log components, just return the dimension with log components
        elif self._log_dimensions:
            return Dimension._construct(self._proper_exponents, self._log_dimensions)
        elif other._log_dimensions:
            return Dimension._construct(other._proper_exponents, other._log_dimensions)
        else:
            # No log dimensions, just return the proper exponents (should be the same for both)
            return Dimension._construct(self._proper_exponents, {})

    def __sub__(self, other: "Dimension") -> "Dimension":
        """
        Subtract two dimensions.
        
        For regular dimensions, subtraction is only valid if the dimensions are identical.
        For logarithmic dimensions, subtraction follows logarithmic identities.
        
        Args:
            other: The dimension to subtract from this one
        
        Returns:
            A new Dimension representing the difference
        
        Raises:
            ValueError: If the dimensions cannot be subtracted
        
        Examples:
            # Regular dimensions (must be identical)
            force1 = Dimension("M*L/T^2")
            force2 = Dimension("M*L/T^2")
            net_force = force1 - force2  # Valid
            
            # Logarithmic dimensions
            log_mass = Dimension("DEC(M)")
            log_length = Dimension("DEC(L)")
            log_ratio = log_mass - log_length  # DEC(M/L)
        """

        if not Dimension.is_valid_for_addition(self, other):
            raise ValueError("Invalid dimensions for subtraction.")
        
        # If both dimensions have log components, we need to apply logarithmic identities
        if self._log_dimensions and other._log_dimensions:
            # For log dimensions, subtraction should follow: log(a) - log(b) = log(a/b)
            # We need to combine the underlying dimensions and then take the log
            
            # Start with the proper exponents (should be the same for both dimensions)
            combined_proper_exponents = self._proper_exponents.copy()
            
            # For each log dimension, we need to get its underlying dimension
            # and combine them multiplicatively (dividing by other's underlying dimensions)
            combined_underlying = Dimension._construct(combined_proper_exponents, {})
            
            # Process log dimensions from self (multiplication)
            for log_dim, exponent in self._log_dimensions.items():
                if abs(exponent) >= EPSILON:
                    # The log_dim represents the underlying dimension that was logged
                    # We need to multiply by this underlying dimension
                    combined_underlying = combined_underlying * (log_dim ** exponent)
            
            # Process log dimensions from other (division)
            for log_dim, exponent in other._log_dimensions.items():
                if abs(exponent) >= EPSILON:
                    # We need to divide by this underlying dimension
                    combined_underlying = combined_underlying / (log_dim ** exponent)
            
            # Now take the log of the combined underlying dimension
            return combined_underlying.log() # type: ignore
        
        # If only one dimension has log components, just return the dimension with log components
        elif self._log_dimensions:
            return Dimension._construct(self._proper_exponents, self._log_dimensions)
        elif other._log_dimensions:
            return Dimension._construct(other._proper_exponents, other._log_dimensions)
        else:
            # No log dimensions, just return the proper exponents (should be the same for both)
            return Dimension._construct(self._proper_exponents, {})

    @overload
    def __mul__(self, other: "Dimension") -> "Dimension":
        """
        Multiply two dimensions.
        
        Multiplication combines the exponents of corresponding base dimensions
        and merges logarithmic dimensions.
        
        Args:
            other: The dimension to multiply by
        
        Returns:
            A new Dimension representing the product
        
        Examples:
            # Basic multiplication
            length = Dimension("L")
            area = length * length  # L^2
            
            # Complex multiplication
            force = Dimension("M*L/T^2")
            area = Dimension("L^2")
            pressure = force / area  # M/L/T^2
            
            # With logarithmic dimensions
            log_mass = Dimension("DEC(M)")
            log_length = Dimension("DEC(L)")
            log_mass_length = log_mass * log_length  # DEC(M)*DEC(L)
        """
        ...

    @overload
    def __mul__(self, other: float|int) -> "Dimension":
        """
        Multiply a dimension by a scalar.

        Args:
            other: The scalar to multiply by

        Returns:
            A new Dimension representing the product
        """
        ...

    def __mul__(self, other: Union["Dimension", float, int]) -> "Dimension":

        if isinstance(other, float|int):

            new_log_dimensions: dict["Dimension", float] = {}
            if len(self._log_dimensions) > 0:
                raise NotImplementedError("Multiplication of a dimension with a scalar and log dimensions is not implemented.")
            return Dimension._construct(self._proper_exponents, new_log_dimensions)

        elif isinstance(other, Dimension): # type: ignore

            # Multiply proper exponents        
            subscripts: set[str] = set(self._proper_exponents.keys()) | set(other._proper_exponents.keys())
            proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]] = {}
            for subscript in subscripts:
                if subscript in self._proper_exponents and subscript in other._proper_exponents:
                    values_1: Tuple[float, float, float, float, float, float, float, float] = self._proper_exponents[subscript] # type: ignore
                    values_2: Tuple[float, float, float, float, float, float, float, float] = other._proper_exponents[subscript] # type: ignore                
                    proper_exponents[subscript] = (
                        values_1[0] + values_2[0] if abs(values_1[0] + values_2[0]) >= EPSILON else 0.0, # type: ignore
                        values_1[1] + values_2[1] if abs(values_1[1] + values_2[1]) >= EPSILON else 0.0, # type: ignore
                        values_1[2] + values_2[2] if abs(values_1[2] + values_2[2]) >= EPSILON else 0.0, # type: ignore
                        values_1[3] + values_2[3] if abs(values_1[3] + values_2[3]) >= EPSILON else 0.0, # type: ignore
                        values_1[4] + values_2[4] if abs(values_1[4] + values_2[4]) >= EPSILON else 0.0, # type: ignore
                        values_1[5] + values_2[5] if abs(values_1[5] + values_2[5]) >= EPSILON else 0.0, # type: ignore
                        values_1[6] + values_2[6] if abs(values_1[6] + values_2[6]) >= EPSILON else 0.0, # type: ignore
                        values_1[7] + values_2[7] if abs(values_1[7] + values_2[7]) >= EPSILON else 0.0, # type: ignore
                    )
                    if proper_exponents[subscript] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
                        del proper_exponents[subscript]
                elif subscript in self._proper_exponents:
                    values_1: Tuple[float, float, float, float, float, float, float, float] = self._proper_exponents[subscript] # type: ignore
                    proper_exponents[subscript] = (
                        values_1[0],
                        values_1[1],
                        values_1[2],
                        values_1[3],
                        values_1[4],
                        values_1[5],
                        values_1[6],
                        values_1[7],
                    )
                else:
                    values_2: Tuple[float, float, float, float, float, float, float, float] = other._proper_exponents[subscript] # type: ignore
                    proper_exponents[subscript] = (
                        values_2[0],
                        values_2[1],
                        values_2[2],
                        values_2[3],
                        values_2[4],
                        values_2[5],
                        values_2[6],
                        values_2[7],
                    )

            # Multiply log dimensions
            log_dimensions: set["Dimension"] = set(self._log_dimensions.keys()) | set(other._log_dimensions.keys()) # type: ignore
            new_log_dimensions: dict["Dimension", float] = {}
            for log_dimension in log_dimensions:
                if log_dimension in self._log_dimensions and log_dimension in other._log_dimensions:
                    new_exponent: float = self._log_dimensions[log_dimension] + other._log_dimensions[log_dimension] # type: ignore
                    if abs(new_exponent) >= EPSILON: # type: ignore
                        new_log_dimensions[log_dimension] = new_exponent
                elif log_dimension in self._log_dimensions:
                    new_log_dimensions[log_dimension] = self._log_dimensions[log_dimension]
                else:
                    new_log_dimensions[log_dimension] = other._log_dimensions[log_dimension]

            return Dimension._construct(proper_exponents, new_log_dimensions)
        
        else:
            raise ValueError("Invalid value for multiplication.")
    
    def __truediv__(self, other: "Dimension") -> "Dimension":
        """
        Divide two dimensions.
        
        Division subtracts the exponents of corresponding base dimensions
        and handles logarithmic dimensions appropriately.
        
        Args:
            other: The dimension to divide by
        
        Returns:
            A new Dimension representing the quotient
        
        Examples:
            # Basic division
            area = Dimension("L^2")
            length = Dimension("L")
            length_result = area / length  # L
            
            # Complex division
            force = Dimension("M*L/T^2")
            area = Dimension("L^2")
            pressure = force / area  # M/L/T^2
            
            # With logarithmic dimensions
            log_mass_length = Dimension("DEC(M)*DEC(L)")
            log_mass = Dimension("DEC(M)")
            log_length = log_mass_length / log_mass  # DEC(L)
        """

        # Divide proper exponents        
        subscripts: set[str] = set(self._proper_exponents.keys()) | set(other._proper_exponents.keys())
        proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]] = {}
        for subscript in subscripts:
            if subscript in self._proper_exponents and subscript in other._proper_exponents:
                values_1: Tuple[float, float, float, float, float, float, float, float] = self._proper_exponents[subscript] # type: ignore
                values_2: Tuple[float, float, float, float, float, float, float, float] = other._proper_exponents[subscript] # type: ignore                
                proper_exponents[subscript] = (
                    values_1[0] - values_2[0] if abs(values_1[0] - values_2[0]) >= EPSILON else 0.0, # type: ignore
                    values_1[1] - values_2[1] if abs(values_1[1] - values_2[1]) >= EPSILON else 0.0, # type: ignore
                    values_1[2] - values_2[2] if abs(values_1[2] - values_2[2]) >= EPSILON else 0.0, # type: ignore
                    values_1[3] - values_2[3] if abs(values_1[3] - values_2[3]) >= EPSILON else 0.0, # type: ignore
                    values_1[4] - values_2[4] if abs(values_1[4] - values_2[4]) >= EPSILON else 0.0, # type: ignore
                    values_1[5] - values_2[5] if abs(values_1[5] - values_2[5]) >= EPSILON else 0.0, # type: ignore
                    values_1[6] - values_2[6] if abs(values_1[6] - values_2[6]) >= EPSILON else 0.0, # type: ignore
                    values_1[7] - values_2[7] if abs(values_1[7] - values_2[7]) >= EPSILON else 0.0, # type: ignore
                )
                if proper_exponents[subscript] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
                    del proper_exponents[subscript]
            elif subscript in self._proper_exponents:
                values_1: Tuple[float, float, float, float, float, float, float, float] = self._proper_exponents[subscript] # type: ignore
                proper_exponents[subscript] = (
                    values_1[0],
                    values_1[1],
                    values_1[2],
                    values_1[3],
                    values_1[4],
                    values_1[5],
                    values_1[6],
                    values_1[7],
                )
            else:
                values_2: Tuple[float, float, float, float, float, float, float, float] = other._proper_exponents[subscript] # type: ignore
                proper_exponents[subscript] = (
                    -values_2[0],
                    -values_2[1],
                    -values_2[2],
                    -values_2[3],
                    -values_2[4],
                    -values_2[5],
                    -values_2[6],
                    -values_2[7],
                )

        # Divide log dimensions
        log_dimensions: set["Dimension"] = set(self._log_dimensions.keys()) | set(other._log_dimensions.keys()) # type: ignore
        new_log_dimensions: dict["Dimension", float] = {}
        for log_dimension in log_dimensions:
            if log_dimension in self._log_dimensions and log_dimension in other._log_dimensions:
                new_exponent: float = self._log_dimensions[log_dimension] - other._log_dimensions[log_dimension] # type: ignore
                if abs(new_exponent) >= EPSILON: # type: ignore
                    new_log_dimensions[log_dimension] = new_exponent
            elif log_dimension in self._log_dimensions:
                new_log_dimensions[log_dimension] = self._log_dimensions[log_dimension]
            else:
                new_log_dimensions[log_dimension] = -other._log_dimensions[log_dimension]

        return Dimension._construct(proper_exponents, new_log_dimensions)
    
    def __pow__(self, exponent: float|int) -> "Dimension":
        """
        Raise a dimension to a power.
        
        Args:
            exponent: The power to raise the dimension to
        
        Returns:
            A new Dimension representing the result
        
        Examples:
            length = Dimension("L")
            area = length ** 2  # L^2
            volume = length ** 3  # L^3
            
            force = Dimension("M*L/T^2")
            force_squared = force ** 2  # M^2*L^2/T^4
            
            # Zero power returns dimensionless
            dimensionless = length ** 0
        """
        # If exponent is 0, return dimensionless dimension
        if abs(exponent) <= EPSILON:
            return DIMENSIONLESS_DIMENSION
        
        proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]] = {}
        for subscript, values in self._proper_exponents.items():
            new_values = (
                values[0] * exponent,
                values[1] * exponent,
                values[2] * exponent,
                values[3] * exponent,
                values[4] * exponent,
                values[5] * exponent,
                values[6] * exponent,
                values[7] * exponent,
            )
            # Only add if not all zeros
            if any(abs(val) >= EPSILON for val in new_values):
                proper_exponents[subscript] = new_values
        
        new_log_dimensions: dict["Dimension", float] = {}
        for dimension, log_exponent in self._log_dimensions.items():
            new_log_exponent = log_exponent * exponent
            if abs(new_log_exponent) >= EPSILON:
                new_log_dimensions[dimension] = new_log_exponent
        
        return Dimension._construct(proper_exponents, new_log_dimensions)
    
    def __invert__(self) -> "Dimension":
        """
        Invert a dimension (take the reciprocal).
        
        Returns:
            A new Dimension representing the reciprocal
        
        Examples:
            length = Dimension("L")
            inverse_length = ~length  # L^-1
            
            force = Dimension("M*L/T^2")
            inverse_force = ~force  # M^-1*L^-1*T^2
        """
        proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]] = {}
        for subscript, values in self._proper_exponents.items():
            proper_exponents[subscript] = (
                values[0] * -1,
                values[1] * -1,
                values[2] * -1,
                values[3] * -1,
                values[4] * -1,
                values[5] * -1,
                values[6] * -1,
                values[7] * -1,
            )
        new_log_dimensions: dict["Dimension", float] = {}
        for dimension, log_exponent in self._log_dimensions.items():
            new_log_dimensions[dimension] = log_exponent * -1
        return Dimension._construct(proper_exponents, new_log_dimensions)

    def invert(self) -> "Dimension":
        """
        Invert a dimension (take the reciprocal).
        
        Returns:
            A new Dimension representing the reciprocal
        
        Examples:
            length = Dimension("L")
            inverse_length = length.invert()  # L^-1
        """
        return ~self
    
    def log(self) -> "Dimension":
        """
        Take the logarithm of a dimension.
        
        Returns:
            A new Dimension representing the logarithm
        
        Examples:
            length = Dimension("L")
            log_length = length.log()  # DEC(L)
            
            force = Dimension("M*L/T^2")
            log_force = force.log()  # DEC(M*L/T^2)
        """
        return Dimension._construct({}, {self: 1.0})
    
    def exp(self) -> "Dimension":
        """
        Take the exponential of a dimension.
        
        Only valid for dimensionless dimensions or single logarithmic dimensions.
        
        Returns:
            A new Dimension representing the exponential
        
        Raises:
            ValueError: If the dimension is not valid for exponential
        
        Examples:
            # Dimensionless
            dimensionless = Dimension("")
            exp_dimless = dimensionless.exp()  # Still dimensionless
            
            # Logarithmic dimension
            log_mass = Dimension("DEC(M)")
            mass = log_mass.exp()  # M
        """
        if self == DIMENSIONLESS_DIMENSION:
            return DIMENSIONLESS_DIMENSION
        elif len(self._log_dimensions) == 1:
            dimension, exponent = next(iter(self._log_dimensions.items()))
            if abs(exponent - 1.0) <= EPSILON:
                return dimension # type: ignore
            else:
                raise ValueError("Invalid dimension.")
        else:
            raise ValueError("Invalid dimension.")
        
    def arc(self) -> "Dimension":
        """
        Take the arc function of a dimension.
        
        Only valid for dimensionless dimensions.
        
        Returns:
            A new Dimension representing the result (angle dimension)
        
        Raises:
            ValueError: If the dimension is not valid for arc function
        
        Examples:
            dimensionless = Dimension("")
            angle = dimensionless.arc()  # A (angle dimension)
        """
        if self == DIMENSIONLESS_DIMENSION:
            return ANGLE_DIMENSION
        else:
            raise ValueError("Invalid dimension.")
        
    def trig(self) -> "Dimension":
        """
        Take the trigonometric function of a dimension.
        
        Only valid for dimensionless dimensions or angle dimensions.
        
        Returns:
            A new Dimension representing the result (dimensionless)
        
        Raises:
            ValueError: If the dimension is not valid for trigonometric function
        
        Examples:
            # Dimensionless
            dimensionless = Dimension("")
            trig_result = dimensionless.trig()  # Still dimensionless
            
            # Angle dimension
            angle = Dimension("A")
            trig_result = angle.trig()  # Dimensionless
        """
        if self == DIMENSIONLESS_DIMENSION:
            return DIMENSIONLESS_DIMENSION
        elif self == ANGLE_DIMENSION:
            return DIMENSIONLESS_DIMENSION
        else:
            raise ValueError("Invalid dimension.")
    
################################################################################
# Comparison operations
################################################################################

    def __eq__(self, other: object) -> bool:
        """
        Check if two dimensions are equal.
        
        Two dimensions are equal if they have the same subscripts,
        the same proper exponents, and the same logarithmic dimensions.
        
        Args:
            other: The dimension to compare with
        
        Returns:
            True if the dimensions are equal, False otherwise
        
        Examples:
            force1 = Dimension("M*L/T^2")
            force2 = Dimension("M*L/T^2")
            assert force1 == force2
            
            # Different dimensions
            mass = Dimension("M")
            length = Dimension("L")
            assert mass != length
        """
        if not isinstance(other, Dimension):
            return False
        if self._proper_exponents.keys() != other._proper_exponents.keys():
            return False
        for subscript in self._proper_exponents.keys():
            subscript: str = subscript # type: ignore
            if self._proper_exponents[subscript] != other._proper_exponents[subscript]:
                return False
        if self._log_dimensions.keys() != other._log_dimensions.keys():
            return False
        for dimension in self._log_dimensions.keys():
            dimension: "Dimension" = dimension # type: ignore
            if self._log_dimensions[dimension] != other._log_dimensions[dimension]:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        """
        Checks if two dimensions are not equal.
        This is the case if they have different subscripts or have different proper exponents or log dimensions.
        """
        if not isinstance(other, Dimension):
            return True
        return not self.__eq__(other)
    
    @staticmethod
    def is_valid_for_addition(dimension_1: "Dimension", dimension_2: "Dimension") -> bool:
        """
        Check if two dimensions can be added or subtracted.
        
        Two dimensions can be added/subtracted if they have the same
        subscripts and the same proper exponents.
        
        Args:
            dimension_1: First dimension
            dimension_2: Second dimension
        
        Returns:
            True if the dimensions can be added/subtracted, False otherwise
        
        Examples:
            force1 = Dimension("M*L/T^2")
            force2 = Dimension("M*L/T^2")
            assert Dimension.is_valid_for_addition(force1, force2)
            
            # Different dimensions cannot be added
            mass = Dimension("M")
            length = Dimension("L")
            assert not Dimension.is_valid_for_addition(mass, length)
        """
        if dimension_1._proper_exponents.keys() != dimension_2._proper_exponents.keys():
            return False
        for subscript in dimension_1._proper_exponents.keys():
            subscript: str = subscript # type: ignore
            if dimension_1._proper_exponents[subscript] != dimension_2._proper_exponents[subscript]:
                return False
        return True
    
    @staticmethod
    def is_valid_for_log(dimension: "Dimension") -> bool:
        """
        Check if a dimension is valid for the log function.
        
        All dimensions are valid for the log function.
        
        Args:
            dimension: The dimension to check
        
        Returns:
            Always True
        
        Examples:
            mass = Dimension("M")
            assert Dimension.is_valid_for_log(mass)
            
            force = Dimension("M*L/T^2")
            assert Dimension.is_valid_for_log(force)
        """
        return True
    
    @staticmethod
    def is_valid_for_exponentiation(dimension: "Dimension", exponent: float) -> bool:
        """
        Check if a dimension is valid for exponentiation.
        
        A dimension is valid for exponentiation if it has no proper exponents
        and exactly one logarithmic dimension.
        
        Args:
            dimension: The dimension to check
            exponent: The exponent (not used in validation)
        
        Returns:
            True if the dimension is valid for exponentiation, False otherwise
        
        Examples:
            log_mass = Dimension("DEC(M)")
            assert Dimension.is_valid_for_exponentiation(log_mass, 2.0)
            
            # Regular dimensions are not valid for exponentiation
            mass = Dimension("M")
            assert not Dimension.is_valid_for_exponentiation(mass, 2.0)
        """
        if len(dimension._proper_exponents) != 0:
            return False
        if len(dimension._log_dimensions) != 1:
            return False
        return True

    @staticmethod
    def is_valid_for_arc(dimension: "Dimension") -> bool:
        """
        Check if a dimension is valid for the arc function.
        
        Only dimensionless dimensions are valid for the arc function.
        
        Args:
            dimension: The dimension to check
        
        Returns:
            True if the dimension is valid for arc function, False otherwise
        
        Examples:
            dimensionless = Dimension("")
            assert Dimension.is_valid_for_arc(dimensionless)
            
            # Non-dimensionless dimensions are not valid
            mass = Dimension("M")
            assert not Dimension.is_valid_for_arc(mass)
        """
        return dimension.is_dimensionless
    
    @staticmethod
    def is_valid_for_trig(dimension: "Dimension") -> bool:
        """
        Check if a dimension is valid for the trig function.
        
        Only dimensionless dimensions or angle dimensions are valid for trig functions.
        
        Args:
            dimension: The dimension to check
        
        Returns:
            True if the dimension is valid for trig function, False otherwise
        
        Examples:
            dimensionless = Dimension("")
            assert Dimension.is_valid_for_trig(dimensionless)
            
            angle = Dimension("A")
            assert Dimension.is_valid_for_trig(angle)
            
            # Other dimensions are not valid
            mass = Dimension("M")
            assert not Dimension.is_valid_for_trig(mass)
        """
        return dimension.is_dimensionless or dimension == ANGLE_DIMENSION
    

################################################################################
# Hash operations
################################################################################

    def __hash__(self) -> int:
        return hash(tuple(self._proper_exponents.values()) + tuple(self._log_dimensions.values()))

################################################################################
# Properties
################################################################################

    @property
    def is_dimensionless(self) -> bool:
        """
        Check if the dimension is dimensionless.
        
        A dimension is dimensionless if it has no proper exponents
        and no logarithmic dimensions.
        
        Returns:
            True if the dimension is dimensionless, False otherwise
        
        Examples:
            dimensionless = Dimension("")
            assert dimensionless.is_dimensionless
            
            mass = Dimension("M")
            assert not mass.is_dimensionless
            
            force = Dimension("M*L/T^2")
            assert not force.is_dimensionless
        """
        if len(self._proper_exponents) != 0:
            return False
        if len(self._log_dimensions) != 0:
            return False
        return True
    
################################################################################
# Compatibility
################################################################################

    def compatible_to(self, *others: "Dimension | Unit | United") -> bool:
        """
        Check if the dimension is compatible with other dimensions.
        Two dimensions are compatible if they have the same subscripts
        and the same proper exponents.

        Args:
            *others: Other dimensions to check compatibility with
        
        """
        # Import here to avoid circular import
        from .unit import Unit
        from .united import United
        
        for other in others:
            if isinstance(other, Unit):
                other = other.dimension
            elif isinstance(other, United):
                other = other.dimension
            elif isinstance(other, "Dimension"): # type: ignore
                pass
            else:
                raise ValueError(f"Invalid dimension: {other}")
            
            if self._proper_exponents.keys() != other._proper_exponents.keys():
                return False
            for subscript in self._proper_exponents.keys():
                subscript: str = subscript # type: ignore
                if self._proper_exponents[subscript] != other._proper_exponents[subscript]:
                    return False
            if self._log_dimensions.keys() != other._log_dimensions.keys():
                return False
            for log_dimension in self._log_dimensions.keys():
                log_dimension: "Dimension" = log_dimension # type: ignore
                if self._log_dimensions[log_dimension] != other._log_dimensions[log_dimension]:
                    return False
        return True
    
################################################################################
# String representation
################################################################################

    def __str__(self) -> str:
        """
        Return the string representation of the dimension.
        
        Returns:
            String representation using fraction formatting
        
        Examples:
            force = Dimension("M*L/T^2")
            print(str(force))  # "M*L/T^2"
        """
        return self.format_string()

    def format_string(self, as_fraction: bool = True) -> str:
        """
        Format the dimension as a string.
        
        Args:
            as_fraction: If True, use fraction notation (e.g., "L/T^2");
                        If False, use negative exponents (e.g., "L*T^-2")
        
        Returns:
            Formatted string representation of the dimension
        
        Examples:
            force = Dimension("M*L/T^2")
            
            # Fraction formatting (default)
            force.format_string()  # "M*L/T^2"
            force.format_string(as_fraction=True)  # "M*L/T^2"
            
            # Negative exponent formatting
            force.format_string(as_fraction=False)  # "M*L*T^-2"
            
            # Logarithmic dimensions
            log_force = Dimension("DEC(M*L/T^2)")
            log_force.format_string()  # "DEC(M*L/T^2)"
        """

        def format_exponent(exponent: float) -> float:
            if abs(exponent) <= EPSILON: # type: ignore
                return 0.0
            elif abs(exponent - 1.0) <= EPSILON: # type: ignore
                return 1.0
            elif abs(exponent + 1.0) <= EPSILON: # type: ignore
                return -1.0
            else:
                # Check if the exponent is close to an integer
                rounded = round(exponent)
                if abs(exponent - rounded) <= EPSILON: # type: ignore
                    return float(rounded)
                else:
                    return exponent

        string: str = ""
        for subscript, values in self._proper_exponents.items():
            subscript: str = subscript # type: ignore
            if subscript == "":
                subscript_string: str = ""
            else:
                subscript_string: str = "_" + subscript # type: ignore
            for i, exponent in enumerate(values):
                exponent: float = format_exponent(exponent)
                if exponent == 0.0:
                    continue

                dimension_symbol: str = BASE_DIMENSION_SYMBOLS[i].symbol

                if exponent > 0.0 or not as_fraction:
                    # Use multiplication for positive exponents
                    if exponent == 1.0:
                        string += f"*{dimension_symbol}{subscript_string}"
                    else:
                        # Convert to int if it's an integer
                        display_exponent = int(exponent) if abs(exponent - int(exponent)) < EPSILON else exponent
                        string += f"*{dimension_symbol}^{display_exponent}{subscript_string}"
                else:
                    # Use division for negative exponents
                    if exponent == -1.0:
                        string += f"/{dimension_symbol}{subscript_string}"
                    else:
                        # For fraction formatting, use positive exponent with division
                        # Convert to int if it's an integer
                        display_exponent = int(abs(exponent)) if abs(exponent) == int(abs(exponent)) else abs(exponent)
                        string += f"/{dimension_symbol}^{display_exponent}{subscript_string}"

        log_dimension_string: str = ""
        for log_dimension, log_exponent in self._log_dimensions.items():
            log_dimension: "Dimension" = log_dimension # type: ignore
            log_exponent: float = format_exponent(log_exponent)
            if log_exponent == 0.0:
                raise AssertionError("Invalid dimension.")
            
            # Format the log dimension content
            log_content = log_dimension.format_string(as_fraction) # type: ignore
            
            # If the log content is empty (dimensionless), use just LOG_STRING without parentheses
            if log_content == "":
                if log_exponent == 1.0:
                    log_dimension_string += f"*{LOG_DIMENSION_SYMBOL_STRING}"
                elif log_exponent == -1.0 and as_fraction:
                    log_dimension_string += f"/{LOG_DIMENSION_SYMBOL_STRING}"
                elif log_exponent < 0.0 and as_fraction:
                    # For fraction formatting, use positive exponent with division
                    log_dimension_string += f"/{LOG_DIMENSION_SYMBOL_STRING}^{abs(log_exponent)}"
                else:
                    log_dimension_string += f"*{LOG_DIMENSION_SYMBOL_STRING}^{log_exponent}"
            else:
                # Normal case: use parentheses around the content
                if log_exponent == 1.0:
                    log_dimension_string += f"*{LOG_DIMENSION_SYMBOL_STRING}({log_content})"
                elif log_exponent == -1.0 and as_fraction:
                    log_dimension_string += f"/{LOG_DIMENSION_SYMBOL_STRING}({log_content})"
                elif log_exponent < 0.0 and as_fraction:
                    # For fraction formatting, use positive exponent with division
                    log_dimension_string += f"/{LOG_DIMENSION_SYMBOL_STRING}({log_content})^{abs(log_exponent)}"
                else:
                    log_dimension_string += f"*{LOG_DIMENSION_SYMBOL_STRING}({log_content})^{log_exponent}"

        if log_dimension_string.startswith("1"):
            log_dimension_string = log_dimension_string[1:]

        string += log_dimension_string

        if string.startswith("*"):
            string = string[1:]
        elif string.startswith("/"):
            string = "1" + string
        elif string == "":
            # Handle dimensionless dimensions - return empty string
            pass
        else:
            raise AssertionError("Invalid dimension.")
        
        return string
    
    def __repr__(self) -> str:
        return self.format_string()
    
################################################################################
# Parsing
################################################################################

    @classmethod
    def _parse_string(cls, string: str) -> "Dimension":
        """
        Parses a string into a named simple dimension.
        Examples:
        - "L" -> {"": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0); {}}
        - "L^2" -> {"": (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0); {}}
        - "L^2/T" -> {"": (0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0); {}}
        - "M/L^3" -> {"": (1.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0); {}}
        - "L^2_elec/L^2_geo" -> {"_elec": (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0), "_geo": (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0); {}}
        - "M*DEC(L^2_elec/L^2_geo)/DEC(M)^2" -> {"": (1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0); {DIM(L^2_elec/L^2_geo): 1, DIM(M): -2}}
        """

        seperators_and_parts: list[tuple[str, str]] = seperate_string(string)

        proper_exponents_lists: dict[str, list[float]] = {}
        log_dimensions: dict["Dimension", float] = {}

        for seperator, part in seperators_and_parts:
            if seperator == "*":
                position: float = 1.0
            elif seperator == "/":
                position: float = -1.0
            else:
                raise ValueError("Invalid seperator.")
            
            # Check if this is a log function call by checking against all LOG_UNIT_SYMBOLS and LOG_DIMENSION_SYMBOL_STRING
            from .unit_symbol import LOG_UNIT_SYMBOLS
            is_log_function = False
            
            # Check LOG_UNIT_SYMBOLS (lowercase functions)
            for log_symbol_enum in LOG_UNIT_SYMBOLS:
                for log_symbol in log_symbol_enum.value.symbols:
                    if part.startswith(log_symbol + "("):
                        is_log_function = True
                        break
                if is_log_function:
                    break
            
            # Also check LOG_DIMENSION_SYMBOL_STRING (uppercase DEC)
            if not is_log_function and part.startswith(LOG_DIMENSION_SYMBOL_STRING + "("):
                is_log_function = True
            

            
            if is_log_function:

                # Find the first ")" from the end of the string
                opening_bracket_index: int = part.find("(")
                first_closing_bracket_index: int = part.rfind(")")
                if first_closing_bracket_index == -1:
                    raise ValueError("Invalid dimension string.")

                log_dimension_string: str = part[opening_bracket_index + 1:first_closing_bracket_index]
                
                # Handle nested log functions by converting all log functions to DEC format
                from .unit_symbol import LOG_UNIT_SYMBOLS
                normalized_log_dimension_string = log_dimension_string
                for log_symbol_enum in LOG_UNIT_SYMBOLS:
                    for log_symbol in log_symbol_enum.value.symbols:
                        if log_symbol != LOG_DIMENSION_SYMBOL_STRING:  # Don't replace DEC with DEC
                            normalized_log_dimension_string = normalized_log_dimension_string.replace(f"{log_symbol}(", f"{LOG_DIMENSION_SYMBOL_STRING}(")
                
                log_dimension: "Dimension" = Dimension._parse_string(normalized_log_dimension_string)
                if part.endswith(")"):
                    log_exponent: float = position
                else:
                    pow_chae_index: int = part.rfind("^")
                    exponent: float = float(part[pow_chae_index + 1:])
                    log_exponent: float = position * exponent

                log_dimensions[log_dimension] = log_exponent
            else:
                if part.count("_") == 0:
                    dimension_element: str = part
                    subscript: str = ""
                else:
                    dimension_element, subscript = part.split("_", 1)
                    subscript: str = subscript

                # Skip "1" as it's just a placeholder
                if dimension_element == "1":
                    continue

                if not subscript in proper_exponents_lists:
                    proper_exponents_lists[subscript] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                if dimension_element.count("^") == 0:
                    symbol: str = dimension_element
                    exponent: float = 1.0 * position
                else:
                    symbol, exponent = dimension_element.split("^") # type: ignore
                    exponent: float = float(exponent) * position

                if DimensionSymbol.is_dimension_symbol(symbol):
                    proper_exponents_lists[subscript][DimensionSymbol.get_index(symbol)] = exponent
                else:
                    try:
                        from .unit_element import UnitElement
                        unit_element: UnitElement = UnitElement.parse_string(symbol, "nominator")
                        for i, exponent_of_unit_element in enumerate(unit_element.dimension._proper_exponents[""]):
                            proper_exponents_lists[subscript][i] += exponent * exponent_of_unit_element
                    except ValueError:
                        raise ValueError(f"Invalid dimension symbol: {symbol}")

        proper_exponents: dict[str, Tuple[float, float, float, float, float, float, float, float]] = {}
        for subscript, values in proper_exponents_lists.items():
            proper_exponents[subscript] = tuple(values) # type: ignore

        return Dimension._construct(proper_exponents, log_dimensions)

################################################################################
# Canonical unit
################################################################################

    @property
    def canonical_unit(self) -> "Unit":
        """
        Get the canonical unit for this dimension.
        
        Returns:
            A Unit object representing the canonical unit for this dimension
        
        Examples:
            Dimension("M").canonical_unit -> Unit("kg")
            Dimension("L").canonical_unit -> Unit("m")
            Dimension("T").canonical_unit -> Unit("s")
            Dimension("M*L/T^2").canonical_unit -> Unit("kg*m/s^2")
        """

        if not hasattr(self, "_canonical_unit"):

            # Import here to avoid circular import
            from .unit import Unit
            from .unit_element import UnitElement
            from .unit_symbol import BASE_10_LOG_UNIT_SYMBOL
            from .dimension_symbol import BASE_DIMENSION_SYMBOLS
            
            # Create unit elements from the dimension's proper exponents
            unit_elements: dict[str, tuple[UnitElement, ...]] = {}
            for subscript, exponents in self._proper_exponents.items():
                elements: list[UnitElement] = []
                for i, exponent in enumerate(exponents):
                    if abs(exponent) > EPSILON:
                        unit_element: UnitElement = UnitElement(prefix=BASE_DIMENSION_SYMBOLS[i].base_unit_prefix, unit_symbol=BASE_DIMENSION_SYMBOLS[i].base_unit_symbol, exponent=exponent)
                        elements.append(unit_element)
                unit_elements[subscript] = tuple(elements)

            # Create log unit elements from the dimension's log dimensions
            log_unit_elements: list[tuple[UnitElement, "Dimension"]] = []
            for log_dimension, log_exponent in self._log_dimensions.items():
                if abs(log_exponent) > EPSILON:
                    unit_element: UnitElement = UnitElement(prefix=None, unit_symbol=BASE_10_LOG_UNIT_SYMBOL, exponent=log_exponent)
                    log_unit_elements.append((unit_element, log_dimension))

            unreduced_canonical_unit: Unit = Unit._construct(unit_elements, log_unit_elements) # type: ignore
            self._canonical_unit: Unit = Unit.reduce_unit(unreduced_canonical_unit) # type: ignore

        return self._canonical_unit

################################################################################
# JSON serialization
################################################################################

    def to_json(self) -> str:
        """
        Convert the dimension to a JSON string representation.
        
        Returns:
            JSON string representation of the dimension
        
        Examples:
            force = Dimension("M*L/T^2")
            json_str = force.to_json()  # "M*L/T^2"
        """
        return self.format_string()

    @classmethod
    def from_json(cls, json_string: str) -> "Dimension":
        """
        Create a dimension from a JSON string representation.
        
        Args:
            json_string: JSON string representation of the dimension
        
        Returns:
            A new Dimension object
        
        Examples:
            json_str = "M*L/T^2"
            force = Dimension.from_json(json_str)
        """
        return Dimension._parse_string(json_string)
    
################################################################################
# HDF5 serialization
################################################################################

    def to_hdf5(self, hdf5_group: Group) -> None:
        """
        Save the dimension to an HDF5 group.
        
        Args:
            hdf5_group: HDF5 group to save the dimension to
        
        Examples:
            import h5py
            
            with h5py.File('data.h5', 'w') as f:
                force = Dimension("M*L/T^2")
                force.to_hdf5(f)
        """
        hdf5_group.attrs["dimension"] = self.format_string()

    @classmethod
    def from_hdf5(cls, hdf5_group: Group) -> "Dimension":
        """
        Load a dimension from an HDF5 group.
        
        Args:
            hdf5_group: HDF5 group containing the dimension
        
        Returns:
            A new Dimension object
        
        Examples:
            import h5py
            
            with h5py.File('data.h5', 'r') as f:
                force = Dimension.from_hdf5(f)
        """
        return Dimension._parse_string(hdf5_group.attrs["dimension"]) # type: ignore
    
    @classmethod
    def dimensionless_dimension(cls) -> "Dimension":
        """
        Get the dimensionless dimension.
        
        Returns:
            A dimensionless Dimension object
        
        Examples:
            dimensionless = Dimension.dimensionless_dimension()
            assert dimensionless.is_dimensionless
        """
        return DIMENSIONLESS_DIMENSION
    
    def __getstate__(self) -> dict[str, Any]:
        """Custom pickle state management for Dimension class."""
        # Convert MappingProxyType objects to regular dicts for pickling
        return {
            "_proper_exponents": dict(self._proper_exponents),
            "_log_dimensions": dict(self._log_dimensions)
        }
    
    def __getnewargs__(self) -> tuple[str]:
        """Provide arguments for __new__ during pickle restoration."""
        # Return empty string to create a basic dimensionless dimension
        # __setstate__ will then properly restore the actual state
        return ("",)
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom pickle state restoration for Dimension class."""
        # Convert dicts back to MappingProxyType and set directly on the object
        from types import MappingProxyType
        
        # Set attributes directly on the already-created object
        object.__setattr__(self, '_proper_exponents', MappingProxyType(state.get("_proper_exponents", {})))
        object.__setattr__(self, '_log_dimensions', MappingProxyType(state.get("_log_dimensions", {})))
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the dimension cache.
        
        The Dimension class caches parsed dimensions for performance.
        This method clears the cache to free memory.
        
        Examples:
            # Clear cache to free memory
            Dimension.clear_cache()
        """
        cls._CACHE.clear()

    @staticmethod
    def extract(obj: Any) -> Optional["Dimension"]:
        """
        Extract a dimension from an object.
        """
        # Import here to avoid circular import
        from .unit import Unit
        from .united import United
        from .named_quantity import NamedQuantity
        
        if isinstance(obj, Dimension):
            return obj
        elif isinstance(obj, Unit):
            return obj.dimension
        elif isinstance(obj, United):
            return obj.dimension
        elif isinstance(obj, NamedQuantity):
            return obj.dimension
        else:
            return None
        
    @classmethod
    def check_dimensions(cls, dictionary: dict[United, "Dimension"]) -> bool:
        """
        Check if the dimensions of the United objects in the dictionary are compatible.
        """
        for united in dictionary.keys():
            if not united.dimension.compatible_to(dictionary[united]):
                return False
        return True

# Predefined dimension constants
DIMENSIONLESS_DIMENSION: "Dimension"    = Dimension._construct({}, {}) # type: ignore
"""Predefined dimensionless dimension constant."""

LOG_LEVEL_DIMENSION: "Dimension"        = Dimension._construct({}, {DIMENSIONLESS_DIMENSION: 1.0}) # type: ignore
"""Predefined logarithmic dimensionless dimension constant."""

ANGLE_DIMENSION: "Dimension"            = Dimension._construct({"": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)}, {}) # type: ignore
"""Predefined angle dimension constant."""