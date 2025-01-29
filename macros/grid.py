from collections import Counter
from typing import Callable


def make_grid(val: any, x: int, y: int) -> list[list[any]]:
    """Create a 2D grid initialized with a given value.

    Examples:
        >>> make_grid(0, 2, 3)
        [[0, 0], [0, 0], [0, 0]]
        >>> make_grid(".", 3, 2)
        [[".", ".", "."], [".", ".", "."]]
    """
    # Create y rows of x columns filled with val
    return [[val for _ in range(x)] for _ in range(y)]


def grid_map(grid: list[list[any]], op: Callable[[any], any]) -> list[list[any]]:
    """Apply a function to each element in a 2D grid.

    Examples:
        >>> grid_map([[1, 2], [3, 4]], lambda x: x * 2)
        [[2, 4], [6, 8]]
        >>> grid_map([['a', 'b'], ['c', 'd']], str.upper)
        [['A', 'B'], ['C', 'D']]
    """
    # Create new empty grid
    newgrid = []

    # Transform each element using the operation
    for y in range(len(grid)):
        newgrid.append(list())
        for x in range(len(grid[0])):
            newgrid[y].append(op(grid[y][x]))

    return newgrid


def gm_context(
    grid: list[list[any]], op: Callable[[int, int, any], any]
) -> list[list[any]]:
    """Apply a function to each element in a 2D grid with position context.

    The operation function receives x position, y position, and grid value as arguments.

    Examples:
        >>> gm_context([[1, 2], [3, 4]], lambda x, y, v: v + x + y)
        [[1, 3], [4, 6]]
    """
    # Create new empty grid
    newgrid = []

    # Transform each element using operation with position context
    for y in range(len(grid)):
        newgrid.append(list())
        for x in range(len(grid[0])):
            newgrid[y].append(op(x, y, grid[y][x]))

    return newgrid


def neighbors4(
    g: list[list[any]],
    x: int,
    y: int,
    max_x: int | None = None,
    max_y: int | None = None,
) -> list[tuple[any, int, int]]:
    """Get the 4 orthogonally adjacent neighbors (up, down, left, right) of a grid position.

    Examples:
        >>> neighbors4([[1,2,3], [4,5,6], [7,8,9]], 1, 1)
        [(4, 0, 1), (6, 2, 1), (2, 1, 0), (3, 1, 2)]
    """
    # Use grid dimensions if bounds not specified
    max_x = max_x if max_x else len(g[0])
    max_y = max_y if max_y else len(g)

    # Store valid neighbor positions
    res = []

    # Check each orthogonal offset
    for off_x, off_y in (
        (-1, 0),  # Left
        (1, 0),  # Right
        (0, -1),  # Up
        (0, 1),  # Down
    ):
        new_x, new_y = x + off_x, y + off_y
        # Add neighbor if within bounds
        if 0 <= new_x < max_x and 0 <= new_y < max_y:
            res.append((g[new_y][new_x], new_x, new_y))
    return res


def neighbors8(
    g: list[list[any]],
    x: int,
    y: int,
    max_x: int | None = None,
    max_y: int | None = None,
) -> list[tuple[any, int, int]]:
    """Get all 8 adjacent neighbors (orthogonal + diagonal) of a grid position.

    Examples:
        >>> neighbors8([[1,2,3], [4,5,6], [7,8,9]], 1, 1)
        [(4, 0, 1), (6, 2, 1), (2, 1, 0), (8, 1, 2), (1, 0, 0), (9, 2, 2), (3, 2, 0), (7, 0, 2)]
    """
    # Use grid dimensions if bounds not specified
    max_x = max_x if max_x else len(g[0])
    max_y = max_y if max_y else len(g)

    # Store valid neighbor positions
    res = []

    # Check each orthogonal and diagonal offset
    for off_x, off_y in (
        (-1, 0),  # Left
        (1, 0),  # Right
        (0, -1),  # Up
        (0, 1),  # Down
        (-1, -1),  # Up-left
        (1, 1),  # Down-right
        (1, -1),  # Up-right
        (-1, 1),  # Down-left
    ):
        new_x, new_y = x + off_x, y + off_y
        # Add neighbor if within bounds
        if 0 <= new_x < max_x and 0 <= new_y < max_y:
            res.append((g[new_y][new_x], new_x, new_y))
    return res


def min_max_xy(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Get the minimum and maximum x,y coordinates from a list of points.

    Examples:
        >>> min_max_xy([(1,2), (3,4), (0,1)])
        (0, 3, 1, 4)
        >>> min_max_xy([Point(1,2), Point(3,4)])
        (1, 3, 2, 4)
    """
    # Handle tuple points
    if isinstance(points[0], tuple):
        # Extract x,y coordinates using list comprehension
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), max(xs), min(ys), max(ys)

    # Handle Point objects
    else:
        raise NotImplementedError("Point objects not supported yet")


def print_grid(grid: list[list[any]]) -> None:
    """Print a 2D grid with elements joined together.

    Examples:
        >>> print_grid([['a','b'], ['c','d']])
        ab
        cd
    """
    for line in grid:
        print(*line, sep="")


def format_grid(
    grid: list[list[any]]
    | dict[tuple[int, int], any]
    | set[tuple[int, int] | tuple[int, int, any]],
    f: callable = None,
    quiet: bool = False,
) -> tuple[list[str], Counter]:
    """Format and print a 2D grid, with optional value transformation and statistics.

    Works with both 2D arrays and sparse grids (dictionaries with (x,y) coordinate keys).
    Prints the formatted grid to stdout and returns both the serialized grid and value counts.

    Args:
        grid: Either a 2D list or dict with (x,y) coordinate keys
        f: Optional function to transform grid values before printing
        quiet: If True, suppress printing to stdout

    Returns:
        tuple containing:
            - list[str]: Serialized rows of the formatted grid
            - Counter: Count of each value in the grid

    Examples:
        >>> g = [['#', '.'], ['.', '#']]
        >>> format_grid(g)
        #.
        .#
        height=2 (0 -> 1)
        width=2 (0 -> 1)
        Statistics:
        #: 2
        .: 2
        (['#.', '.#'], Counter({'#': 2, '.': 2}))
    """
    # Default transform function just converts to string
    if f is None:
        f = str

    # Track value counts and serialized rows
    counts = Counter()
    serialized = []
    min_x, max_x, min_y, max_y = 0, 0, 0, 0

    # Handle sparse grid (dictionary)
    if isinstance(grid, dict):
        if not quiet:
            print("=== [ SPARSE GRID (DICT) ] ===")
        positions = list(grid.keys())
        min_x, max_x, min_y, max_y = min_max_xy(positions)

        # Only handle tuple coordinates for now
        if isinstance(positions[0], tuple):
            # Build each row by getting values at coordinates
            for y in range(min_y, max_y + 1):
                row = "".join(f(grid.get((x, y), " ")) for x in range(min_x, max_x + 1))
                if not quiet:
                    print(row)
                serialized.append(row)
                for c in row:
                    counts[c] += 1
            if not quiet:
                print("=" * (max_x - min_x + 3))
            print("")
        else:
            raise TypeError("Grid dictionary keys must be (x,y) tuples")

    # Handle 2D array grid
    elif isinstance(grid, list):
        if not quiet:
            print("=== [ 2D ARRAY GRID ] ===")
        min_x, max_x, min_y, max_y = 0, len(grid[0]) - 1, 0, len(grid) - 1

        # Process each row
        for y in range(min_y, max_y + 1):
            row = "".join(f(grid[y][x]) for x in range(min_x, max_x + 1))
            if not quiet:
                print(row)
            serialized.append(row)
            for c in row:
                counts[c] += 1
        if not quiet:
            print("=" * (max_x - min_x + 3))
            print("")

    # Handle sparse grid (set)
    elif isinstance(grid, set):
        if not quiet:
            print("=== [ SPARSE GRID (SET) ] ===")

        grid_map = {}
        for value in grid:
            if len(value) == 2:
                x, y = value
                c = "#"
            else:
                x, y, c = value
            grid_map[(x, y)] = c

        positions = list(grid_map.keys())
        min_x, max_x, min_y, max_y = min_max_xy(positions)
        # Build each row by getting values at coordinates
        for y in range(min_y, max_y + 1):
            row = "".join(grid_map.get((x, y), " ") for x in range(min_x, max_x + 1))
            if not quiet:
                print(row)
            serialized.append(row)
            for c in row:
                counts[c] += 1
        if not quiet:
            print("=" * (max_x - min_x + 3))
            print("")

    else:
        raise TypeError("Unsupported grid type")

    # Print grid statistics
    if not quiet:
        print(f"height={max_y - min_y + 1} ({min_y} -> {max_y})")
        print(f"width={max_x - min_x + 1} ({min_x} -> {max_x})")
        print("Statistics:")
        for item, num in counts.most_common():
            print(f"{item}: {num}")

    return serialized, counts


def __test_grid():
    print("Testing make_grid")
    assert make_grid(0, 2, 3) == [[0, 0], [0, 0], [0, 0]]
    assert make_grid(".", 3, 2) == [[".", ".", "."], [".", ".", "."]]

    print("Testing grid_map")
    assert grid_map([[1, 2], [3, 4]], lambda x: x * 2) == [[2, 4], [6, 8]]
    assert grid_map([["a", "b"], ["c", "d"]], str.upper) == [["A", "B"], ["C", "D"]]

    print("Testing gm_context")
    assert gm_context([[1, 2], [3, 4]], lambda x, y, v: v + x + y) == [[1, 3], [4, 6]]

    print("Testing neighbors4")
    assert neighbors4([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1, 1) == [
        (4, 0, 1),
        (6, 2, 1),
        (2, 1, 0),
        (8, 1, 2),
    ]

    print("Testing neighbors8")
    assert neighbors8([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1, 1) == [
        (4, 0, 1),
        (6, 2, 1),
        (2, 1, 0),
        (8, 1, 2),
        (1, 0, 0),
        (9, 2, 2),
        (3, 2, 0),
        (7, 0, 2),
    ]

    print("Testing min_max_xy")
    assert min_max_xy([(1, 2), (3, 4), (0, 1)]) == (0, 3, 1, 4)
