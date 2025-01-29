import sys
from typing import Any, Optional

#
# Legacy Input Parsing
#


def get_input(
    file_path: str | None = None,
    split: Optional[str] = None,
    auto_zero_index: bool = True,
):
    """Parse input from either command line argument file or stdin.

    Args:
        file_path (str | None): If provided, read from this file path
        split (Optional[str]): If provided, splits input on this delimiter before line splitting
        auto_zero_index (bool): If True, single-line groups are automatically indexed to [0]

    Returns:
        If split is None:
            List of strings, one per line from input file with newlines stripped
        If split is provided:
            Generator of either strings (for single-line groups) or lists of strings (multi-line groups)
            Each group is split on newlines after initial split on delimiter

    Examples:
        # Input file "input.txt":
        # a,b
        # c,d

        get_input() -> ['a,b', 'c,d']

    Examples:
        # Input file "input.txt":
        # 1 2 3 4
        # 2 3 4 5
        # ===
        # a b c d
        # b c d e
        # c d e f

        get_input(split="===") -> [['1 2 3 4', '2 3 4 5'], ['a b c d', 'b c d e', 'c d e f']]

    Examples:
        # Input file "input.txt":
        # 12345

        get_input() -> ['12345']
        get_input(auto_zero_index=True) -> '12345'

    Examples:
        # Input file "input.txt":
        # 1234
        # ~
        # hello
        # this
        # is
        # a
        # test

        get_input(split="~", auto_zero_index=True) -> ['1234', ['hello', 'this', 'is', 'a', 'test']]
    """
    # Use command line arg if provided, otherwise use passed file_path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if "demo" in file_path:
            print("WARNING: DEMO INPUT FILE!")

    if file_path is not None:
        # Read from file
        with open(file_path, "r") as f:
            if split is None:
                # Simple case - just read lines and strip newlines
                data = [line.rstrip() for line in f]
            else:
                # Split on delimiter first, then newlines within groups
                content = f.read().strip()
                data = []
                for group in content.split(split):
                    lines = group.strip().split("\n")
                    # For single-line groups with auto_zero_index, return just the line
                    data.append(
                        lines[0] if auto_zero_index and len(lines) == 1 else lines
                    )
    else:
        # Read from stdin until empty line
        data = []
        while (line := input()) != "":
            data.append(line)

    # For overall single-line input with auto_zero_index, return just the line
    if auto_zero_index and len(data) == 1:
        data = data[0]

    return data
