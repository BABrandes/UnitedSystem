from typing import overload

from .real_united_scalar import RealUnitedScalar
from .real_united_array import RealUnitedArray
from .dimension import Dimension
from .unit import Unit

########################
# Addition
########################

@overload
def add(a: RealUnitedScalar, b: RealUnitedScalar) -> RealUnitedScalar:
    """
    Add two real united scalars.
    """
    ...
@overload
def add(a: RealUnitedArray, b: RealUnitedArray) -> RealUnitedArray:
    """
    Add two real united arrays.
    """
    ...
@overload
def add(a: RealUnitedScalar, b: RealUnitedArray) -> RealUnitedArray:
    """
    Add a real united scalar to a real united array.
    """
    ...
def add(a: RealUnitedArray|RealUnitedScalar, b: RealUnitedArray|RealUnitedScalar) -> RealUnitedArray|RealUnitedScalar:
    """
    Add two real united scalars or arrays.
    """
    
    raise NotImplementedError("Addition of real united scalars and arrays is not implemented")

########################
# Subtraction
########################

@overload
def subtract(a: RealUnitedScalar, b: RealUnitedScalar) -> RealUnitedScalar:
    """
    Subtract two real united scalars.
    """
    ...
@overload
def subtract(a: RealUnitedArray, b: RealUnitedArray) -> RealUnitedArray:
    """
    Subtract two real united arrays.
    """
    ...
@overload
def subtract(a: RealUnitedScalar, b: RealUnitedArray) -> RealUnitedArray:
    """
    Subtract a real united scalar from a real united array.
    """
    ...
def subtract(a: RealUnitedArray|RealUnitedScalar, b: RealUnitedArray|RealUnitedScalar) -> RealUnitedArray|RealUnitedScalar:
    """
    Subtract two real united scalars or arrays.
    """
    raise NotImplementedError("Subtraction of real united scalars and arrays is not implemented")

########################
# Multiplication
########################

@overload
def multiply(a: RealUnitedScalar, b: RealUnitedScalar) -> RealUnitedScalar:
    """
    Multiply two real united scalars.
    """
    ...
@overload
def multiply(a: RealUnitedArray, b: RealUnitedArray) -> RealUnitedArray:
    """
    Multiply two real united arrays.
    """
    ...
@overload
def multiply(a: RealUnitedScalar, b: RealUnitedArray) -> RealUnitedArray:
    """
    Multiply a real united scalar by a real united array.
    """
    ...
@overload
def multiply(a: Dimension, b: Dimension) -> Dimension:
    """
    Multiply a dimension by a dimension.
    """
    ...
@overload
def multiply(a: Unit, b: Unit) -> Unit:
    """
    Multiply a unit by a unit.
    """
    ...
@overload
def multiply(a: float, b: RealUnitedScalar) -> RealUnitedScalar:
    """
    Multiply a float by a real united scalar.
    """
    ...
@overload
def multiply(a: int, b: RealUnitedScalar) -> RealUnitedScalar:
    """
    Multiply an integer by a real united scalar.
    """
    ...
@overload
def multiply(a: float, b: RealUnitedArray) -> RealUnitedArray:
    """
    Multiply a float by a real united array.
    """
    ...
@overload
def multiply(a: int, b: RealUnitedArray) -> RealUnitedArray:
    """
    Multiply an integer by a real united array.
    """
    ...
def multiply(a: RealUnitedScalar|RealUnitedArray|Dimension|Unit, b: RealUnitedScalar|RealUnitedArray|Dimension|Unit) -> RealUnitedScalar|RealUnitedArray|Dimension|Unit:
    """
    Multiply two real united scalars or arrays.
    """
    raise NotImplementedError("Multiplication of real united scalars and arrays is not implemented")

########################
# Division
########################