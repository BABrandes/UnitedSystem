from typing import Sequence
from .unit_element import UnitElement
from .proper_exponents import ProperExponents
from ...named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES, QuantityTag
from functools import lru_cache

EPSILON: float = 1e-12

# Cache proper exponents for tuples of unit elements
@lru_cache(maxsize=2048)
def cached_proper_exponents(elements_tuple: tuple[UnitElement, ...]) -> tuple[float, float, float, float, float, float, float, float]:
    return ProperExponents.proper_exponents_of_unit_elements(elements_tuple)

def reduce_unit_elements(elements: Sequence[UnitElement]) -> Sequence[UnitElement]:
    """
    Try to reduce a group of unit elements by finding subsets that can be replaced with derived units.
    
    Args:
        elements: Sequence of unit elements to reduce
        
    Returns:
        Sequence of reduced unit elements
    """
    if len(elements) <= 1:
        return elements

    # Cache of visited exponent states to avoid redundant recursion
    visited_states: set[tuple[float, ...]] = set()
    # Precompute named quantity exponents
    NAMED_QUANTITY_EXPONENTS: list[tuple[NamedQuantity, tuple[float, float, float, float, float, float, float, float]]] = [
        (q, q.proper_exponents_as_tuple) for q in CONSIDER_FOR_REDUCTION_QUANTITIES
    ]
    
    def subtract(
            index: int,
            value_1: tuple[float, float, float, float, float, float, float, float],
            value_2: tuple[float, float, float, float, float, float, float, float]
            ) -> tuple[float, tuple[float, float, float, float, float, float, float, float]]:
        """
        Subtract the value_2 from the value_1, so that at the given index the result 0

        Args:
            index: The index to subtract at
            value_1: The first value
            value_2: The second value

        Returns:
            The result of the subtraction
        """
        exponent = value_1[index] / value_2[index]
        return exponent, tuple(value_1[i] - value_2[i] * exponent for i in range(8)) # type: ignore
    
    def is_zero(
            value: tuple[float, float, float, float, float, float, float, float]
            ) -> bool:
        """
        Check if the value is zero.
        """
        return all(abs(v) < EPSILON for v in value)
    
    absolute_max_num_named_quantities: int = 4
    max_num_named_quantities: int = min(len(elements) - 1, absolute_max_num_named_quantities)
    highest_absolute_exponent: float = max(abs(e.exponent) for e in elements)

    def update_final_named_quantities_candidates(
            current_named_quantities: list[tuple[NamedQuantity, float]],
            remaining_proper_exponents: tuple[float, float, float, float, float, float, float, float],
            final_named_quantities_candidates: list[list[tuple[NamedQuantity, float]]],
            fewest_num_named_quantities_found_so_far: int
            ) -> int:
        # Prune paths that can't improve on the best found
        if len(current_named_quantities) >= fewest_num_named_quantities_found_so_far:
            return fewest_num_named_quantities_found_so_far
        # Skip already-visited exponent states
        if remaining_proper_exponents in visited_states:
            return fewest_num_named_quantities_found_so_far
        visited_states.add(remaining_proper_exponents)

        for named_quantity, quantity_exponents in NAMED_QUANTITY_EXPONENTS:
            # Skip named quantities that don't overlap with remaining exponents
            if not any(re != 0 and qe != 0 for re, qe in zip(remaining_proper_exponents, quantity_exponents)):
                continue
            for idx, q_exp in enumerate(quantity_exponents):
                if q_exp == 0:
                    continue
                exponent, new_proper_exponents = subtract(idx, remaining_proper_exponents, quantity_exponents)
                # Skip if exponent is effectively zero
                if abs(exponent) < EPSILON:
                    continue
                # Skip if exponent is too large
                if abs(exponent) > highest_absolute_exponent:
                    continue
                tuple_to_add: tuple[NamedQuantity, float] = (named_quantity, exponent)
                extended_named_quantity_candidates = current_named_quantities + [tuple_to_add]

                if is_zero(new_proper_exponents):
                    final_named_quantities_candidates.append(extended_named_quantity_candidates)
                    if len(extended_named_quantity_candidates) < fewest_num_named_quantities_found_so_far:
                        fewest_num_named_quantities_found_so_far = len(extended_named_quantity_candidates)
                elif len(extended_named_quantity_candidates) < max_num_named_quantities and len(extended_named_quantity_candidates)-1 <= fewest_num_named_quantities_found_so_far:
                    update_final_named_quantities_candidates(
                        extended_named_quantity_candidates,
                        new_proper_exponents,
                        final_named_quantities_candidates,
                        fewest_num_named_quantities_found_so_far)
        return fewest_num_named_quantities_found_so_far

    # See if there is a direct match to a NamedQuantity
    proper_exponents: tuple[float, float, float, float, float, float, float, float] = cached_proper_exponents(tuple(elements))
    final_named_quantities_candidates: list[list[tuple[NamedQuantity, float]]] = []
    update_final_named_quantities_candidates(
        [],
        proper_exponents,
        final_named_quantities_candidates,
        len(elements)
    )

    def score_named_quantities_candidates(
            named_quantities_candidate: list[tuple[NamedQuantity, float]]
            ) -> float:
        """
        Score the named quantities candidates. Lower score is better.
        """
        score: float = len(named_quantities_candidate) * 100
        for named_quantity, exponent in named_quantities_candidate:
            score += (abs(exponent)-1) * 10
            if abs(exponent - round(exponent, 1)) > EPSILON:
                score += 20
            if QuantityTag.BASE_QUANTITY in named_quantity.tags:
                score += 0
            if QuantityTag.DERIVED_QUANTITY in named_quantity.tags:
                score += 1
        return score

    if len(final_named_quantities_candidates) == 0:
        return elements
    
    scores: list[float] = [score_named_quantities_candidates(candidate) for candidate in final_named_quantities_candidates]

    best_score: float = min(scores)
    best_candidate: list[tuple[NamedQuantity, float]] = final_named_quantities_candidates[scores.index(best_score)]
    new_unit_elements: list[UnitElement] = []
    for named_quantity, exponent in best_candidate:
        if named_quantity.unit_element is None:
            raise AssertionError("NamedQuantity has no single unit element")
        unit_element: UnitElement = UnitElement(named_quantity.unit_element.prefix, named_quantity.unit_element.unit_symbol, exponent)
        new_unit_elements.append(unit_element)
    return new_unit_elements