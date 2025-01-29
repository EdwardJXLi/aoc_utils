#
# Imports Needed Only for Macros
# Rest are in aoc_utils/libs.py
#

import operator
from functools import reduce
from itertools import chain, combinations, permutations
from typing import Any, Generator, Iterable

#
# Helpers
#


def cross_join(
    sequence1: list, sequence2: list
) -> Generator[tuple[Any, Any], None, None]:
    """Yields all possible pairs of elements from two lists.
    >>> list(cross_join([1,2], ['a','b'])) -> [(1,'a'), (1,'b'), (2,'a'), (2,'b')]
    """
    for item1 in sequence1:
        for item2 in sequence2:
            yield item1, item2


def reverse(sequence: list) -> list:
    """Reverses a list.
    >>> reverse([1,2,3]) -> [3,2,1]
    """
    return sequence[::-1]


def flipv(matrix: list) -> list:
    """Flips a 2D array vertically (reverses rows order).
    >>> flipv([[1,2], [3,4]]) -> [[3,4], [1,2]]
    """
    return matrix[::-1]


def fliph(matrix: list) -> list:
    """Flips a 2D array horizontally (reverses each row).
    >>> fliph([[1,2], [3,4]]) -> [[2,1], [4,3]]
    """
    return [row[::-1] for row in matrix]


def transpose(matrix: list, as_string: bool = False) -> list:
    """Transposes a 2D array.
    >>> transpose([[1,2], [3,4]]) -> [[1,3], [2,4]]
    >>> transpose(['ab', 'cd'], as_string=True) -> ['ac', 'bd']
    """
    if not as_string:
        return list(map(list, zip(*matrix)))
    else:
        return ["".join(row) for row in zip(*matrix)]


def rotate(matrix: list) -> list:
    """Rotates a 2D array 90 degrees clockwise.
    >>> rotate([[1,2], [3,4]]) -> [[3,1], [4,2]]
    """
    return [list(row) for row in zip(*matrix[::-1])]


def rotate_cw(matrix: list) -> list:
    """Rotates a 2D array 90 degrees clockwise (alias for rotate).
    >>> rotate_cw([[1,2], [3,4]]) -> [[3,1], [4,2]]
    """
    return rotate(matrix)


def rotate_ccw(matrix: list) -> list:
    """Rotates a 2D array 90 degrees counter-clockwise.
    >>> rotate_ccw([[1,2], [3,4]]) -> [[2,4], [1,3]]
    """
    return [list(row) for row in zip(*matrix)][::-1]


def first_column(matrix: list) -> list:
    """Returns first column of matrix as a row.
    >>> first_column([[1,2], [3,4]]) -> [1,3]
    """
    return rotate(matrix)[0][::-1]


def last_column(matrix: list) -> list:
    """Returns last column of matrix as a row.
    >>> last_column([[1,2], [3,4]]) -> [2,4]
    """
    return rotate(matrix)[-1][::-1]


def product(sequence: list) -> int:
    """Returns product of all elements in list.
    >>> product([2,3,4]) -> 24
    """
    return reduce(operator.mul, sequence, 1)


def join(sequence: list, delimiter: str) -> str:
    """Joins list elements with delimiter after converting to strings.
    >>> join([1,2,3], ',') -> '1,2,3'
    """
    return delimiter.join(list(map(str, sequence)))


def window(sequence, n=2):
    """Generate sliding windows over a sequence.
    >>> list(window([1,2,3,4]))
    [(1,2), (2,3), (3,4)]
    >>> list(window([1,2,3,4], n=3))
    [(1,2,3), (2,3,4)]
    """
    iterator = iter(sequence)
    window = []
    for _ in range(n):
        try:
            window.append(next(iterator))
        except StopIteration:
            return
    yield tuple(window)
    for item in iterator:
        window = window[1:] + [item]
        yield tuple(window)


def parts(sequence: list, num_parts: int):
    """Splits list into n equal parts. Last parts may be smaller if list length not divisible by n.
    >>> list(parts([1,2,3,4,5], 2)) -> [[1,2,3], [4,5]]
    """
    # Calculate size of each part, rounding up to ensure all elements are included
    part_size = (len(sequence) + num_parts - 1) // num_parts
    for i in range(0, num_parts):
        yield sequence[i * part_size : (i + 1) * part_size]


def unique(sequence: list) -> bool:
    """Checks if all elements in list are unique.
    >>> unique([1,2,3,1]) -> False
    >>> unique([1,2,3]) -> True
    """
    return len(sequence) == len(set(list(sequence)))


def lmap(*args, **kwargs) -> list:
    """List version of map.
    >>> lmap(str, [1,2,3]) -> ['1','2','3']
    """
    return list(map(*args, **kwargs))


def min_max(*sequence) -> tuple:
    """Returns min and max of sequence.
    >>> min_max([1,5,3,2,4]) -> (1,5)
    """
    if len(sequence) == 1:
        sequence = sequence[0]
    return min(sequence), max(sequence)


def max_min(*sequence, **kwargs) -> tuple:
    """Returns max and min of sequence.
    >>> max_min([1,5,3,2,4]) -> (5,1)
    """
    if len(sequence) == 1:
        sequence = sequence[0]
    return max(sequence, **kwargs), min(sequence, **kwargs)


def max_minus_min(sequence: list, **kwargs) -> int:
    """Returns difference between max and min of sequence.
    >>> max_minus_min([1,5,3,2,4]) -> 4
    """
    return max(sequence, **kwargs) - min(sequence, **kwargs)


def remove_value(sequence: list, value: any) -> list:
    """Returns list with all occurrences of value removed.
    >>> remove_value([1,2,3,2,4], 2) -> [1,3,4]
    """
    return [item for item in sequence if item != value]


def flatten(sequence: list, recursive: bool = True, output_type: type = list) -> list:
    """Flattens nested lists into single list.
    >>> flatten([[1,[2,3]], [4,5]]) -> [1,2,3,4,5]
    """
    if recursive:
        if sequence == []:
            return sequence
        if isinstance(sequence[0], output_type):
            return flatten(sequence[0]) + flatten(sequence[1:])
        return output_type(sequence[:1] + flatten(sequence[1:]))
    else:
        return output_type([item for subseq in sequence for item in subseq])


def powerset(sequence: list) -> list:
    """Returns all possible subsets of iterable.
    >>> list(powerset([1,2])) -> [(), (1,), (2,), (1,2)]
    """
    items = list(sequence)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


def hex_distance(x: int, y: int, z: int) -> int:
    """Calculate the distance from a hex grid point to the origin.

    Uses cube coordinates where x + y + z = 0.

    Examples:
        >>> hex_distance(1, -1, 0)
        1
        >>> hex_distance(2, -1, -1)
        2
    """
    return (abs(x) + abs(y) + abs(z)) // 2


def manhattan_distance(x1: int, y1: int, x2: int = 0, y2: int = 0) -> int:
    """Calculate Manhattan distance between two points.
    >>> manhattan_distance(3, 4) -> 7
    >>> manhattan_distance(1, 2, 4, 6) -> 7
    """
    return abs(x1 - x2) + abs(y1 - y2)


#
# Tests
#


def __test_helpers():
    print("Testing cross_join")
    assert list(cross_join([1, 2], ["a", "b"])) == [
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
    ]

    print("Testing reverse")
    assert reverse([1, 2, 3]) == [3, 2, 1]

    print("Testing flipv")
    assert flipv([[1, 2], [3, 4]]) == [[3, 4], [1, 2]]

    print("Testing fliph")
    assert fliph([[1, 2], [3, 4]]) == [[2, 1], [4, 3]]

    print("Testing transpose")
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert transpose(["ab", "cd"], as_string=True) == ["ac", "bd"]

    print("Testing rotate/rotate_cw")
    assert rotate([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
    assert rotate_cw([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]

    print("Testing rotate_ccw")
    assert rotate_ccw([[1, 2], [3, 4]]) == [[2, 4], [1, 3]]

    print("Testing first_column")
    assert first_column([[1, 2], [3, 4]]) == [1, 3]

    print("Testing last_column")
    assert last_column([[1, 2], [3, 4]]) == [2, 4]

    print("Testing product")
    assert product([2, 3, 4]) == 24

    print("Testing join")
    assert join([1, 2, 3], ",") == "1,2,3"

    print("Testing window")
    assert list(window([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(window([1, 2, 3, 4], n=3)) == [(1, 2, 3), (2, 3, 4)]

    print("Testing parts")
    assert list(parts([1, 2, 3, 4, 5], 2)) == [[1, 2, 3], [4, 5]]

    print("Testing unique")
    assert not unique([1, 2, 3, 1])
    assert unique([1, 2, 3])

    print("Testing lmap")
    assert lmap(str, [1, 2, 3]) == ["1", "2", "3"]

    print("Testing min_max")
    assert min_max([1, 5, 3, 2, 4]) == (1, 5)

    print("Testing max_min")
    assert max_min([1, 5, 3, 2, 4]) == (5, 1)

    print("Testing max_minus_min")
    assert max_minus_min([1, 5, 3, 2, 4]) == 4

    print("Testing remove_value")
    assert remove_value([1, 2, 3, 2, 4], 2) == [1, 3, 4]

    print("Testing flatten")
    assert flatten([[1, [2, 3]], [4, 5]]) == [1, 2, 3, 4, 5]

    print("Testing powerset")
    assert list(powerset([1, 2])) == [(), (1,), (2,), (1, 2)]

    print("Testing hex_distance")
    assert hex_distance(1, -1, 0) == 1
    assert hex_distance(2, -1, -1) == 2

    print("Testing manhattan_distance")
    assert manhattan_distance(3, 4) == 7
    assert manhattan_distance(1, 2, 4, 6) == 7
