#
# Constants
#

inf = INF = float("inf")
ALPHA = LETTERS = [x for x in "abcdefghijklmnopqrstuvwxyz"]
UPPER = UPPERLETTERS = [x.upper() for x in ALPHA]
VOWELS = {"a", "e", "i", "o", "u"}
UPPERVOWELS = {x.upper() for x in VOWELS}
CONSONANTS = set(x for x in LETTERS if x not in VOWELS)
UPPERCONSONANTS = set(x for x in UPPERLETTERS if x not in UPPERVOWELS)
DIGITS = [str(x) for x in range(10)]
HEXDIGITS = DIGITS + ["a", "b", "c", "d", "e", "f"]
UPPERHEXDIGITS = DIGITS + ["A", "B", "C", "D", "E", "F"]
PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
WHITESPACE = set(" \t\n\r\v\f")
PRINTABLE = set(ALPHA + UPPER + DIGITS + list(PUNCTUATION) + list(WHITESPACE))

#
# Grid Traversal Constants
#

N4 = NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N4SELF = NEIGHBORS4SELF = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
N8 = NEIGHBORS8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1))
N8SELF = NEIGHBORS8SELF = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)
N24 = NEIGHBORS24 = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (2, 2),
    (-2, -2),
    (2, -2),
    (-2, 2),
    (1, 2),
    (-1, -2),
    (1, -2),
    (-1, 2),
    (2, 1),
    (-2, -1),
    (2, -1),
    (-2, 1),
)
N24SELF = NEIGHBORS24SELF = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (2, 2),
    (-2, -2),
    (2, -2),
    (-2, 2),
    (1, 2),
    (-1, -2),
    (1, -2),
    (-1, 2),
    (2, 1),
    (-2, -1),
    (2, -1),
    (-2, 1),
)

#
# 3D Grid Traversal Constants
#

THREE_N6 = THREE_NEIGHBORS6 = (
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
)
THREE_N6SELF = THREE_NEIGHBORS6SELF = (
    (0, 0, 0),
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
)
THREE_N18 = THREE_NEIGHBORS18 = (
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 2, -2),
    (0, -2, 2),
    (2, 0, -2),
    (2, -2, 0),
    (-2, 0, 2),
    (-2, 2, 0),
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
)
THREE_N18SELF = THREE_NEIGHBORS18SELF = (
    (0, 0, 0),
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 2, -2),
    (0, -2, 2),
    (2, 0, -2),
    (2, -2, 0),
    (-2, 0, 2),
    (-2, 2, 0),
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
)
THREE_N26 = THREE_NEIGHBORS26 = (
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 2, -2),
    (0, -2, 2),
    (2, 0, -2),
    (2, -2, 0),
    (-2, 0, 2),
    (-2, 2, 0),
    (0, 3, -3),
    (0, -3, 3),
    (3, 0, -3),
    (3, -3, 0),
    (-3, 0, 3),
    (-3, 3, 0),
    (1, 2, -3),
    (1, -2, 3),
    (2, 1, -3),
    (2, -3, 1),
    (-1, 2, 3),
    (-1, -3, 2),
    (-2, 1, 3),
    (-2, 3, -1),
    (1, -3, 2),
    (1, 3, -2),
    (3, -1, 2),
    (3, 2, -1),
    (-1, -2, 3),
    (-1, 3, -2),
    (-2, -1, 3),
    (-2, 3, -1),
    (1, -3, 2),
    (1, 3, -2),
    (3, -1, 2),
    (3, 2, -1),
)
THREE_N26SELF = THREE_NEIGHBORS26SELF = (
    (0, 0, 0),
    (0, 1, -1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (-1, 0, 1),
    (-1, 1, 0),
    (0, 2, -2),
    (0, -2, 2),
    (2, 0, -2),
    (2, -2, 0),
    (-2, 0, 2),
    (-2, 2, 0),
    (0, 3, -3),
    (0, -3, 3),
    (3, 0, -3),
    (3, -3, 0),
    (-3, 0, 3),
    (-3, 3, 0),
    (1, 2, -3),
    (1, -2, 3),
    (2, 1, -3),
    (2, -3, 1),
    (-1, 2, 3),
    (-1, -3, 2),
    (-2, 1, 3),
    (-2, 3, -1),
    (1, -3, 2),
    (1, 3, -2),
    (3, -1, 2),
    (3, 2, -1),
    (-1, -2, 3),
    (-1, 3, -2),
    (-2, -1, 3),
    (-2, 3, -1),
    (1, -3, 2),
    (1, 3, -2),
    (3, -1, 2),
    (3, 2, -1),
)
