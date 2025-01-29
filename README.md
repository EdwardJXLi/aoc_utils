# AOC (AdventOfCode) Utils

Personal collection of functions & utils for [AdventOfCode](https://adventofcode.com/) that I've written, borrowed, and ~~stolen~~ over the years (since [AOC 2020](https://adventofcode.com/2020)). 

> **⚠️ NOTE:** Although this repo is public and open-source, it was very much designed with my personal workflow in mind. As such, it may have a bunch of caveats and "unpythonic" features that I generally wouldn't recommend in any other project. **Use at your own risk!**

## Usage

You can probably pip install this, but I personally just symlink this project into each day's solution of AdventOfCode (not great, but it works). It is also recommended to install `more_itertools`, though it is not strictly required.

#### The general base template with `aoc_utils` is as follows:

```py
from aoc_utils import *

# Main Solution Code
def run(data: str | list[str] | list[list[str]]):
    ...  # Your code here
    return data  # Your final output here (optional)

# Init Code
if __name__ == "__main__":
    if (res := run(get_input())) is not None:
        print(res)
```

## Features

AOC Utils comes with 100+ additional helper functions, 30+ constants, custom input parsers, graph/grid helpers, and (soon) automated cloning and submission of solutions.

#### :: Input Parsing
> These are just a handful of examples. Check out all the features in [parsing.py](parsing.py)

AOC Utils supports auto-zero-indexing and splitting of multiple input blocks.

**Example Input:**
```
1 2
3 4

alpha beta gamma

a b c
d e f
g h i
```

**Example Output:**
```
>>> get_input()  # Reads from sys.argv[1], or optional `file_path=...` argument
['1 2', '3 4', '', 'alpha beta gamma', '', 'a b c', 'd e f', 'g h i']

>>> get_input(split="\n\n", auto_zero_index=True)  # get_input with auto-zero-indexing and data splitting
[
    ['1 2', '3 4'], 
    'alpha beta gamma', 
    ['a b c', 'd e f', 'g h i']
]
```


#### :: Grid/Map Parsers
> These are just a handful of examples. Check out all the features in [macros/imp.py](macros/imp.py)

**[Single-Char Grids] Example Input:**
```
1b3
d5f
```

**Example Output:**
```
>>> single_char_grid(get_input())
(
    [['1','b','3'], 
     ['d','5','f']], 
    3, 2
)  # (grid, size_x, size_y)
```

**[Integer Grids] Example Input:**
```
111 2222 3333
4444 5555 6666
```

**Example Output:**
```
>>> int_grid(get_input())
(
    [[111, 2222, 3333], 
     [4444, 5555, 6666]], 
    3, 2
)  # (grid, size_x, size_y)
```

**[Sparse Grids] Example Input:**
```
#..
##.
###
```

**Example Output:**
```
>>> sparse_grid(get_input(), true=["#"])
{(0, 1), (1, 2), (0, 0), (1, 1), (0, 2), (2, 2)}
```

**Example Input:**
```
A.B
.X.
C.D
```

**Example Output:**
```
>>> sparse_grid(get_input(), true=["A", "B", "C", "D"], axis="down-right", zeros="top-left", keepVal=True)
{(0, 0, "A"), (2, 0, "B"), (0, 2, "C"), (2, 2, "D")}
```


#### :: Grid/Map Helpers
> These are just a handful of examples. Check out all the features in [macros/grid.py](macros/grid.py)

```
>>> make_grid(".", 3, 2)
[[".", ".", "."], [".", ".", "."]]

>>> grid_map([['a', 'b'], ['c', 'd']], str.upper)
[['A', 'B'], ['C', 'D']]

>>> gm_context([[1, 2], [3, 4]], lambda x, y, v: v + x + y)  # grid_map with (x, y) context
[[1, 3], [4, 6]]

>>> neighbors4([[1,2,3], [4,5,6], [7,8,9]], 1, 1)
[(4, 0, 1), (6, 2, 1), (2, 1, 0), (3, 1, 2)]  # (value, x, y)

>>> min_max_xy([(1,2), (3,4), (0,1)])
(0, 3, 1, 4)  # (min_x, max_x, min_y, max_y)
```

#### :: Helper Functions
> These are just a handful of examples. Check out all of them in [macros/helpers.py](macros/helpers.py)

```
>>> split_many("a,b;c|d", ",", ";", "|")
['a', 'b', 'c', 'd']

>>> ints("1 2 3 4")
[1, 2, 3, 4]

>>> floats("There are 2 numbers: -5.1 and 11.22")
[2.0, -5.1, 11.22]

>>> window([1,2,3,4], n=3)
[(1,2,3), (2,3,4)]

>>> parts([1,2,3,4,5], 2)
[[1,2,3], [4,5]]

>>> cross_join([1,2], ['a','b'])
[(1,'a'), (1,'b'), (2,'a'), (2,'b')]

>>> flatten([[1,[2,3]], [4,5]])
[1,2,3,4,5]
```

#### :: Constants
> These are just a handful of examples. Check out all of them in [macros/constants.py](macros/constants.py)

```
>>> UPPERVOWELS
{'E', 'I', 'U', 'A', 'O'}

>>> CONSONANTS
{'d', 'q', 'm', 'g', 'h', 'w', 'p', 'k', 'z', 'b', 't', 'r', 'n', 's', 'c', 'l', 'v', 'x', 'f', 'y', 'j'}

>>> N4
((-1, 0), (1, 0), (0, -1), (0, 1))

>>> N4SELF
((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

>>> N8
((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1))

>>> THREE_N6
((0, 1, -1), (0, -1, 1), (1, 0, -1), (1, -1, 0), (-1, 0, 1), (-1, 1, 0))
```

#### :: Algorithms
> This is just a subset of the algorithms. Check out all of them in [macros/algorithms.py](macros/algorithms.py)

```
>>> def is_greater_than_5(x): return x > 5
>>> bisect_x_int(is_greater_than_5, 0, 10)
6

>>> hamming_distance("karolin", "kathrin")
3

>>> edit_distance("kitten", "sitting")
3

>>> resolve_mapping({'a': [1,2], 'b': [1], 'c': [2,3]})
{'b': 1, 'a': 2, 'c': 3}

>>> matmat([[1, 2], [3, 4]], [[5, 6], [7, 8]])
[[19, 22], [43, 50]]

>>> gcd(48, 18)
6

>>> lcm(12, 15)
60

>>> factors(24)
[1, 2, 3, 4, 6, 8, 12, 24]

>>> crt([2, 3, 2], [3, 5, 7])
(23, 105)  # 23 ≡ 2 (mod 3), 23 ≡ 3 (mod 5), 23 ≡ 2 (mod 7)
```

## Contact & Questions

Feel free to contact me (links in bio) for any questions or help!