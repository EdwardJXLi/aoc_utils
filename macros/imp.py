import re
from typing import Any, Literal

#
# Input Helpers
#


def split_many(data: str, *seps: str) -> list[str]:
    """Split a string on multiple separators by replacing all separators with the first one.

    Example:
        >>> split_many("a,b;c|d", ",", ";", "|")
        ['a', 'b', 'c', 'd']
    """
    # Replace all additional separators with the first one
    for sep in seps[1:]:
        data = data.replace(sep, seps[0])
    # Split on the first separator
    return data.split(seps[0])


def split_data(data: list[str]) -> list[list[str]]:
    """Split input data into groups based on empty line separators.

    Example:
        >>> split_data(['a', 'b', '', 'c', 'd'])
        [['a', 'b'], ['c', 'd']]
    """
    output = []
    current_group = []

    for line in data:
        if line.strip():
            current_group.append(line)
        elif current_group:
            output.append(current_group)
            current_group = []

    if current_group:
        output.append(current_group)

    return output


def ints(s: str | list[str], negatives: bool = True) -> list[int]:
    """Extract all integers from a string or list of strings.

    Example:
        >>> ints("1 2 3 4")
        [1, 2, 3, 4]
        >>> ints("1,2,3,4")
        [1, 2, 3, 4]
        >>> ints("9 -8 -7 6", negatives=False)
        [9, 8, 7, 6]
        >>> ints("There are 2 numbers: -5 and 10")
        [2, -5, 10]
        >>> ints(["1 2", "3 4"])
        [1, 2, 3, 4]
    """
    # Build regex pattern based on whether we want negatives
    pattern = r"-?\d+" if negatives else r"\d+"

    # Handle list input by joining with spaces
    text = " ".join(s) if isinstance(s, list) else s

    # Extract and convert numbers
    return [int(x) for x in re.findall(pattern, text)]


def floats(s: str | list[str], negatives: bool = True) -> list[float]:
    """Extract floating point numbers from a string or list of strings.

    Examples:
        >>> floats('1.5 -2.3 3.0')
        [1.5, -2.3, 3.0]
        >>> floats('1.5,-2.3,3.0')
        [1.5, -2.3, 3.0]
        >>> floats('1.5 -2.3 3.0', negatives=False)
        [1.5, 2.3, 3.0]
        >>> floats(['1.5', '-2.3', '3.0'])
        [1.5, -2.3, 3.0]
    """
    # Handle list input by joining with spaces
    text = " ".join(s) if isinstance(s, list) else s

    # Build regex pattern based on whether we want negatives
    pattern = r"-?\d+(?:\.\d+)?" if negatives else r"\d+(?:\.\d+)?"

    # Extract and convert numbers
    numbers = [float(x) for x in re.findall(pattern, text)]

    # Handle negatives=False by taking absolute value
    if not negatives:
        numbers = [abs(x) for x in numbers]

    return numbers


def read_single_char_grid(
    text: list[str], process: type = str
) -> tuple[list[list[any]], int, int]:
    """Convert a list of strings into a 2D grid, processing each character with the given function.

    Example:
        >>> read_single_char_grid(['1b3', 'd5f'])
        ([['1','b','3'], ['d','5','f']], 3, 2)
    """
    # Process each character in each line to build 2D grid
    grid = []
    for line in text:
        grid.append([process(x) for x in line])
    return (grid, len(grid[0]), len(grid))


def single_char_grid(
    text: list[str], process: type = str
) -> tuple[list[list[any]], int, int]:
    """Alias for read_single_char_grid."""
    return read_single_char_grid(text, process=process)


def read_char_grid(text: list[str]) -> tuple[list[list[str]], int, int]:
    """Convert a list of strings into a 2D character grid.
    Wrapper around read_single_char_grid using str processing.

    Example:
        >>> read_char_grid(['abc', 'def'])
        ([['a','b','c'], ['d','e','f']], 3, 2)
    """
    return read_single_char_grid(text, process=str)


def char_grid(text: list[str]) -> tuple[list[list[str]], int, int]:
    """Alias for read_char_grid."""
    return read_char_grid(text)


def read_int_grid(text: list[str]) -> tuple[list[list[int]], int, int]:
    """Convert a list of strings into a 2D integer grid.
    Wrapper around read_single_char_grid using int processing.

    Example:
        >>> read_int_grid(['123', '456'])
        ([[1,2,3], [4,5,6]], 3, 2)
    """
    return read_single_char_grid(text, process=int)


def int_grid(text: list[str]) -> tuple[list[list[int]], int, int]:
    """Alias for read_int_grid."""
    return read_int_grid(text)


def read_ints_grid(text: list[str]) -> tuple[list[list[int]], int, int]:
    """Convert a list of strings into a 2D integer grid by parsing numbers in each line.
    Uses ints() function to extract numbers from each line.

    Example:
        >>> read_ints_grid(['111 2222 3333', '4444 5555 6666'])
        ([[111, 2222, 3333], [4444, 5555, 6666]], 3, 2)
    """
    grid = []
    for line in text:
        grid.append(ints(line))
    return (grid, len(grid[0]), len(grid))


def ints_grid(text: list[str]) -> tuple[list[list[int]], int, int]:
    """Alias for read_ints_grid."""
    return read_ints_grid(text)


def _parse_sparse_grid(
    grid: list[list[Any]],
    true: list[str] = ["#"],
    axis: Literal["down-right", "up-right", "down-left", "up-left"] = "down-right",
    zeros: Literal["bottom-right", "top-right", "bottom-left", "top-left"] = "top-left",
    offsetX: int = 0,
    offsetY: int = 0,
    keepVal: bool = False,
) -> set[tuple[int, int] | tuple[int, int, Any]]:
    """Convert a 2D grid into a sparse set of coordinates for marked positions.

    Args:
        grid: 2D input grid
        true: List of values to consider as marked positions
        axis: Direction of axes ("down-right", "up-right", etc)
        zeros: Position of origin ("bottom-right", "top-left", etc)
        offsetX: X-axis offset to apply to coordinates
        offsetY: Y-axis offset to apply to coordinates
        keepVal: Whether to include grid values in output coordinates

    Returns:
        Set of tuples containing coordinates (and optionally values) of marked positions

    Examples:
        >>> read_sparse_grid(['#..', '##.', '###'], true=["#"])
        {(0, 1), (1, 2), (0, 0), (1, 1), (0, 2), (2, 2)}
        >>> read_sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="up-left", zeros="top-left", keepVal=True)
        {(0, 0, "A"), (-2, 0, "B"), (0, -2, "C"), (-2, -2, "D")}
    """
    points = set()


    # Handle origin position
    if "bottom" in zeros.lower():
        offsetY -= len(grid) - 1
    if "right" in zeros.lower():
        offsetX -= len(grid[0]) - 1

    # Handle axis orientation
    if "up" in axis.lower():
        grid = grid[::-1]  # Flip vertically
        offsetY *= -1
        offsetY -= len(grid) - 1
    if "left" in axis.lower():
        grid = [x[::-1] for x in grid]  # Flip horizontally
        offsetX *= -1
        offsetX -= len(grid[0]) - 1

    # Build set of coordinates
    for yi, row in enumerate(grid):
        for xi, val in enumerate(row):
            if val in true:
                coord = (
                    (xi + offsetX, yi + offsetY, val)
                    if keepVal
                    else (xi + offsetX, yi + offsetY)
                )
                points.add(coord)

    return points


def read_sparse_grid(
    text: list[str],
    process: type = str,
    true: list[str] = ["#"],
    axis: Literal["down-right", "up-right", "down-left", "up-left"] = "down-right",
    zeros: Literal["bottom-right", "top-right", "bottom-left", "top-left"] = "top-left",
    offsetX: int = 0,
    offsetY: int = 0,
    keepVal: bool = False,
) -> set[tuple[int, int] | tuple[int, int, Any]]:
    """Convert text input directly into sparse grid coordinates.
    Wrapper combining read_single_char_grid and _parse_sparse_grid.

    Args:
        text: Input text as list of strings
        process: Function to process each character
        true: List of values to consider as marked positions
        axis: Direction of axes ("down-right", "up-right", etc)
        zeros: Position of origin ("bottom-right", "top-left", etc)
        offsetX: X-axis offset to apply to coordinates
        offsetY: Y-axis offset to apply to coordinates
        keepVal: Whether to include grid values in output coordinates

    Returns:
        Set of coordinate tuples for marked positions
    """
    return _parse_sparse_grid(
        read_single_char_grid(text, process=process)[0],
        true=true,
        axis=axis,
        zeros=zeros,
        offsetX=offsetX,
        offsetY=offsetY,
        keepVal=keepVal,
    )


def sparse_grid(
    text: list[str],
    process: type = str,
    true: list[str] = ["#"],
    axis: Literal["down-right", "up-right", "down-left", "up-left"] = "down-right",
    zeros: Literal["bottom-right", "top-right", "bottom-left", "top-left"] = "top-left",
    offsetX: int = 0,
    offsetY: int = 0,
    keepVal: bool = False,
) -> set[tuple[int, int] | tuple[int, int, Any]]:
    """Alias for read_sparse_grid."""
    return read_sparse_grid(
        text,
        process=process,
        true=true,
        axis=axis,
        zeros=zeros,
        offsetX=offsetX,
        offsetY=offsetY,
        keepVal=keepVal,
    )


def regex_line(regex: str, line: str) -> list[int | str]:
    """Parse a line using a regex pattern and convert numeric matches to integers.

    Examples:
        >>> regex_line(r"(\\d+)", "123 hello 456 bye")
        [123, 456]
        >>> regex_line(r"(\\w+)", "123 hello 456 bye")
        [123, "hello", 456, "bye"]
    """
    # Store matched groups
    ret = []

    # Try to match the regex pattern
    matches = re.findall(regex, line)
    if not matches:
        raise Exception("Regex pattern did not match input line")

    # Process each captured group
    for match in matches:
        try:
            # Try to convert to integer
            ret.append(int(match))
        except ValueError:
            # Keep as string if not numeric
            ret.append(match)

    return ret


def to_grid(lines: list[str]) -> tuple[list[list[str]], int, int]:
    """Convert a list of strings into a 2D grid of characters.

    Examples:
        >>> to_grid(['abc', 'def'])
        ([['a','b','c'], ['d','e','f']], 3, 2)
        >>> to_grid(['12', '34'])
        ([['1','2'], ['3','4']], 2, 2)
    """
    grid = [[char for char in line] for line in lines]
    return (grid, len(grid[0]), len(grid))


def to_num_grid(lines: list[str]) -> tuple[list[list[int]], int, int]:
    """Convert a list of strings into a 2D grid of integers, skipping empty rows.

    Examples:
        >>> to_num_grid(['1 2 3', '4 5 6'])
        ([[1,2,3], [4,5,6]], 3, 2)
        >>> to_num_grid(['1 2', '', '3 4'])
        ([[1,2], [3,4]], 2, 2)
    """
    # Convert each non-empty row to integers
    grid = [ints(line) for line in lines if len(ints(line)) != 0]
    return (grid, len(grid[0]), len(grid))


def to_ints(input_data: str | list[str]) -> list[int]:
    """Convert a string or list of strings to a list of integers.

    Examples:
        >>> to_ints(['1', '2', '3'])
        [1, 2, 3]
    """
    return list(map(int, input_data))


def __test_imp_helpers():
    print("Testing split_many")
    assert split_many("a,b;c|d", ",", ";", "|") == ["a", "b", "c", "d"]

    print("Testing split_data")
    assert split_data(["a", "b", "", "c", "d"]) == [["a", "b"], ["c", "d"]]

    print("Testing ints")
    assert ints("1 2 3 4") == [1, 2, 3, 4]
    assert ints("1,2,3,4") == [1, 2, 3, 4]
    assert ints("9 -8 -7 6", negatives=False) == [9, 8, 7, 6]
    assert ints("There are 2 numbers: -5 and 10") == [2, -5, 10]
    assert ints(["1 2", "3 4"]) == [1, 2, 3, 4]

    print("Testing floats")
    assert floats("1.5 -2.3 3.0") == [1.5, -2.3, 3.0]
    assert floats("1.5,-2.3,3.0") == [1.5, -2.3, 3.0]
    assert floats("1.5 -2.3 3.0", negatives=False) == [1.5, 2.3, 3.0]
    assert floats(["1.5", "-2.3", "3.0"]) == [1.5, -2.3, 3.0]

    print("Testing single_char_grid")
    assert single_char_grid(["1b3", "d5f"]) == (
        [["1", "b", "3"], ["d", "5", "f"]],
        3,
        2,
    )

    print("Testing char_grid")
    assert char_grid(["abc", "def"]) == ([["a", "b", "c"], ["d", "e", "f"]], 3, 2)

    print("Testing int_grid")
    assert int_grid(["123", "456"]) == ([[1, 2, 3], [4, 5, 6]], 3, 2)

    print("Testing ints_grid")
    assert ints_grid(["111 2222 3333", "4444 5555 6666"]) == (
        [[111, 2222, 3333], [4444, 5555, 6666]],
        3,
        2,
    )

    print("Testing sparse_grid")
    assert sparse_grid(['#..', '##.', '###'], true=["#"]) == {(0, 1), (1, 2), (0, 0), (1, 1), (0, 2), (2, 2)}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="down-right", zeros="top-left", keepVal=True) == {(0, 0, "A"), (2, 0, "B"), (0, 2, "C"), (2, 2, "D")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="up-left", zeros="top-left", keepVal=True) == {(0, 0, "A"), (-2, 0, "B"), (0, -2, "C"), (-2, -2, "D")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="down-right", zeros="bottom-right", keepVal=True) == {(0, 0, "D"), (-2, 0, "C"), (0, -2, "B"), (-2, -2, "A")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="up-left", zeros="bottom-right", keepVal=True) == {(0, 0, "D"), (2, 0, "C"), (0, 2, "B"), (2, 2, "A")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="down-left", zeros="top-right", keepVal=True) == {(0, 0, "B"), (2, 0, "A"), (0, 2, "D"), (2, 2, "C")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="up-right", zeros="top-right", keepVal=True) == {(0, 0, "B"), (-2, 0, "A"), (0, -2, "D"), (-2, -2, "C")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="up-right", zeros="bottom-left", keepVal=True) == {(0, 0, "C"), (2, 0, "D"), (0, 2, "A"), (2, 2, "B")}
    assert sparse_grid(['A.B', '.X.', 'C.D'], true=["A", "B", "C", "D"], axis="down-left", zeros="bottom-left", keepVal=True) == {(0, 0, "C"), (-2, 0, "D"), (0, -2, "A"), (-2, -2, "B")}

    print("Testing regex_line")
    assert regex_line(r"(\d+)", "123 hello 456 bye") == [123, 456]
    assert regex_line(r"(\w+)", "123 hello 456 bye") == [123, "hello", 456, "bye"]

    print("Testing to_grid")
    assert to_grid(["abc", "def"]) == ([["a", "b", "c"], ["d", "e", "f"]], 3, 2)
    assert to_grid(["12", "34"]) == ([["1", "2"], ["3", "4"]], 2, 2)

    print("Testing to_num_grid")
    assert to_num_grid(["1 2 3", "4 5 6"]) == ([[1, 2, 3], [4, 5, 6]], 3, 2)
    assert to_num_grid(["1 2", "", "3 4"]) == ([[1, 2], [3, 4]], 2, 2)

    print("Testing to_ints")
    assert to_ints(["1", "2", "3"]) == [1, 2, 3]
