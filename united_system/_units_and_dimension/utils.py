from typing import Final, Literal

PREFIX_PAIRS: Final[dict[str, float]] = {
    "Y": 10**24,
    "Z": 10**21,
    "E": 10**18,
    "P": 10**15,
    "T": 10**12,
    "G": 10**9,
    "M": 10**6,
    "k": 10**3,
    "h": 10**2,
    "da": 10**1,
    "d": 10**-1,
    "c": 10**-2,
    "m": 10**-3,
    "µ": 10**-6,  # Micro Sign (U+00B5)
    "μ": 10**-6,  # Greek Small Letter Mu (U+03BC) - same value as µ
    "n": 10**-9,
    "p": 10**-12,
    "f": 10**-15,
    "a": 10**-18,
    "z": 10**-21,
    "y": 10**-24,
}

def seperate_string(unit_string: str, position: Literal["nominator", "denominator"]="nominator") -> list[tuple[str, str]]:

    def process_normal_string(string: str, position: Literal["nominator", "denominator"]="nominator") -> list[tuple[str, str]]:
        """
        Seperate a string into parts, it must start with a separator.
        """

        if len(string.strip()) == 1:
            return []
        
        if not string.startswith("*") and not string.startswith("/"):
            raise ValueError("Invalid unit string")

        parts: list[tuple[str, str]] = []
        current: list[str] = []
        separator = string[0]
        level = 0  # Parentheses depth

        for char in string:
            if char == "(":
                level += 1
                current.append(char)
            elif char == ")":
                level -= 1
                current.append(char)
            elif char in "*/" and level == 0:
                # Flush the current buffer
                if current:
                    parts.append((separator, ''.join(current)))
                    current = []
                separator = char
            else:
                current.append(char)

        if current:
            parts.append((separator, ''.join(current)))

        # Apply denominator logic
        if position == "denominator":
            parts = [("/" if sep == "*" else "*", val) for sep, val in parts]

        return parts

    seperators_and_parts: list[tuple[str, str]] = []

    if unit_string == "":
        return []
    
    if unit_string.strip() == "1":
        raise ValueError("'1' is not a valid unit string")

    if unit_string.startswith("1/"):
        unit_string = unit_string[2:]
        normal_string = "/"
    else:
        normal_string = "*"

    position_exponent: int = 1 if position == "nominator" else -1

    i = 0
    within_grouping = False
    grouping_string = ""
    paranthesis_count = 0
    while i < len(unit_string):
        char = unit_string[i]
        if char == '(':
            if within_grouping:
                paranthesis_count += 1
                grouping_string += char
            elif paranthesis_count == 0 and not within_grouping and (i == 0 or unit_string[i-1] in ['*', '/', '(']):
                within_grouping = True
                paranthesis_count = 1
                if i > 0 and unit_string[i-1] == '/':
                    current_group_position_exponent = -1 * position_exponent
                else:
                    current_group_position_exponent = 1 * position_exponent
                if normal_string.endswith("/") or normal_string.endswith("*"):
                    normal_string = normal_string[:-1]
                if normal_string:
                    seperators_and_parts.extend(process_normal_string(normal_string, position))
                normal_string = ""
            else:
                normal_string += char
        elif char == ')':
            if within_grouping:
                paranthesis_count -= 1
                if paranthesis_count == 0:
                    within_grouping = False
                    # Process the grouped content
                    grouped_parts = seperate_string(grouping_string, "nominator" if current_group_position_exponent == 1 else "denominator") # type: ignore
                    seperators_and_parts.extend(grouped_parts)
                    grouping_string = ""
                    normal_string = "*"
                else:
                    grouping_string += char
            else:
                normal_string += char
        elif within_grouping:
            grouping_string += char
        else:
            normal_string += char
        i += 1
    
    # Process any remaining normal string
    if normal_string:
        seperators_and_parts.extend(process_normal_string(normal_string, position))
 
    return seperators_and_parts
