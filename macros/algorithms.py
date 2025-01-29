import hashlib
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar

T = TypeVar("T")


def bisect_x_int(
    f: Callable[[int], bool], lo: int = 0, hi: Optional[int] = None
) -> int:
    """
    Binary search to find value x where f(x) changes from False to True. Returns the FIRST integer that satisfies f(x).

    Args:
        f: Function that returns bool, assumed monotonic
        lo: Lower bound for search
        hi: Upper bound for search (if None, will be found automatically)
        eps: Precision threshold

    Returns:
        Value x where f changes from False to True

    Example:
        >>> def is_greater_than_5(x): return x > 5
        >>> bisect_x_int(is_greater_than_5, 0, 10)
        6
    """
    lo_bool = f(lo)
    if hi is None:
        # Find upper bound by doubling offset
        offset = 1
        while f(lo + offset) == lo_bool:
            offset *= 2
        hi = lo + offset
    else:
        assert f(hi) != lo_bool, "f(lo) and f(hi) must have different values"

    # Binary search
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if f(mid) == lo_bool:
            lo = mid
        else:
            hi = mid
    return lo if lo_bool else hi


def bisect_x_float(
    f: Callable[[float], bool],
    lo: float = 0,
    hi: Optional[float] = None,
    eps: float = 1e-9,
) -> float:
    """
    Binary search to find value x where f(x) changes from False to True. Returns the FIRST float that satisfies f(x).

    Args:
        f: Function that returns bool, assumed monotonic
        lo: Lower bound for search
        hi: Upper bound for search (if None, will be found automatically)
        eps: Precision threshold

    Returns:
        Value x where f changes from False to True

    Example:
        >>> def is_greater_than_5(x): return x > 5
        >>> bisect_x(is_greater_than_5, 0, 10)
        5.0
    """
    lo_bool = f(lo)
    if hi is None:
        # Find upper bound by doubling offset
        offset = 1
        while f(lo + offset) == lo_bool:
            offset *= 2
        hi = lo + offset
    else:
        assert f(hi) != lo_bool, "f(lo) and f(hi) must have different values"

    # Binary search
    while hi - lo > eps:
        mid = (hi + lo) / 2
        if f(mid) == lo_bool:
            lo = mid
        else:
            hi = mid

    return lo if lo_bool else hi


BLANK = object()


def hamming_distance(a: str, b: str) -> int:
    """
    Calculate Hamming distance between two sequences.

    Args:
        a: First sequence
        b: Second sequence

    Returns:
        Number of positions where sequences differ

    Example:
        >>> hamming_distance("karolin", "kathrin")
        3
    """
    return sum(
        i is BLANK or j is BLANK or i != j
        for i, j in zip_longest(a, b, fillvalue=BLANK)
    )


def edit_distance(a: str, b: str) -> int:
    """
    Calculate Levenshtein distance between two strings using dynamic programming.

    Args:
        a: First string
        b: Second string

    Returns:
        Minimum number of single-character edits needed to change a into b

    Example:
        >>> edit_distance("kitten", "sitting")
        3
    """
    n, m = len(a), len(b)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[n][m] = 0

    def aux(i: int, j: int) -> int:
        if dp[i][j] is not None:
            return dp[i][j]

        if i == n:
            dp[i][j] = 1 + aux(i, j + 1)  # deletion
        elif j == m:
            dp[i][j] = 1 + aux(i + 1, j)  # insertion
        else:
            dp[i][j] = min(
                (a[i] != b[j]) + aux(i + 1, j + 1),  # substitution
                1 + aux(i + 1, j),  # deletion
                1 + aux(i, j + 1),  # insertion
            )
        return dp[i][j]

    return aux(0, 0)


def matmat(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """
    Matrix multiplication.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Product matrix

    Example:
        >>> matmat([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
    """
    n, k1 = len(a), len(a[0])
    k2, m = len(b), len(b[0])
    assert k1 == k2, "Matrix dimensions must match"

    out = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(k1))
    return out


def matvec(a: List[List[float]], v: List[float]) -> List[float]:
    """
    Matrix-vector multiplication.

    Args:
        a: Matrix
        v: Vector

    Returns:
        Product vector

    Example:
        >>> matvec([[1, 2], [3, 4]], [5, 6])
        [17, 39]
    """
    return [j for i in matmat(a, [[x] for x in v]) for j in i]


def matexp(a: List[List[int]], k: int) -> List[List[int]]:
    """
    Matrix exponentiation by squaring.

    Args:
        a: Square matrix
        k: Power to raise matrix to

    Returns:
        Matrix raised to power k

    Example:
        >>> matexp([[1, 1], [1, 0]], 3)  # Fibonacci matrix
        [[3, 2], [2, 1]]
    """
    n = len(a)
    out = [[int(i == j) for j in range(n)] for i in range(n)]
    while k > 0:
        if k % 2 == 1:
            out = matmat(a, out)
        a = matmat(a, a)
        k //= 2
    return out


def gcd(a: int, b: int) -> int:
    """
    Compute greatest common divisor using Euclidean algorithm.

    Args:
        a: First number
        b: Second number

    Returns:
        Greatest common divisor

    Example:
        >>> gcd(48, 18)
        6
    """
    while b > 0:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """
    Compute least common multiple.

    Args:
        a: First number
        b: Second number

    Returns:
        Least common multiple

    Example:
        >>> lcm(12, 15)
        60
    """
    return a * b // gcd(a, b)


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.

    Args:
        a: First number
        b: Second number

    Returns:
        Tuple (gcd, x, y) where gcd is the greatest common divisor
        and x, y satisfy ax + by = gcd

    Example:
        >>> egcd(17, 13)
        (1, -3, 4)  # -3*17 + 4*13 = 1
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def modinv(a: int, n: int) -> int:
    """
    Compute modular multiplicative inverse.

    Args:
        a: Number to find inverse for
        n: Modulus

    Returns:
        Modular multiplicative inverse of a modulo n

    Example:
        >>> modinv(3, 11)
        4  # Because (3 * 4) % 11 = 1
    """
    g, x, _ = egcd(a, n)
    if g == 1:
        return x % n
    raise ValueError(f"{a} is not invertible mod {n}")


def crt(rems: List[int], mods: List[int]) -> Tuple[int, int]:
    """
    Chinese Remainder Theorem solver.

    Args:
        rems: List of remainders
        mods: List of moduli (not necessarily coprime)

    Returns:
        Tuple (solution, lcm_of_moduli)

    Example:
        >>> crt([2, 3, 2], [3, 5, 7])
        (23, 105)  # 23 ≡ 2 (mod 3), 23 ≡ 3 (mod 5), 23 ≡ 2 (mod 7)
    """
    rems, mods = list(rems), list(mods)
    newrems, newmods = [], []

    # Handle non-coprime moduli
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            g = gcd(mods[i], mods[j])
            if g == 1:
                continue
            if rems[i] % g != rems[j] % g:
                raise ValueError(
                    f"Inconsistent remainders at positions {i} and {j} (mod {g})"
                )
            mods[j] //= g

            while True:
                g = gcd(mods[i], mods[j])
                if g == 1:
                    break
                mods[i] //= g
                mods[j] *= g

        if mods[i] == 1:
            continue

        newrems.append(rems[i] % mods[i])
        newmods.append(mods[i])

    rems, mods = newrems, newmods

    # Standard CRT
    n = 1
    for k in mods:
        n *= k

    s = 0
    for rem, mod in zip(rems, mods):
        ni = n // mod
        s += rem * modinv(ni, mod) * ni
    return s % n, n


def resolve_mapping(candidates: Dict[Any, List[Any]]) -> Dict[Any, Any]:
    """
    Resolve a mapping where each key maps to multiple candidate values.

    Args:
        candidates: Dict mapping keys to lists of possible values

    Returns:
        Dict with unique value assignments

    Example:
        >>> resolve_mapping({'a': [1,2], 'b': [1], 'c': [2,3]})
        {'b': 1, 'a': 2, 'c': 3}
    """
    resolved = {}
    candidates_map = {k: set(v) for k, v in candidates.items()}

    while len(resolved) < len(candidates_map):
        for candidate in candidates_map:
            if len(candidates_map[candidate]) == 1 and candidate not in resolved:
                value = candidates_map[candidate].pop()
                for c in candidates_map:
                    candidates_map[c].discard(value)
                resolved[candidate] = value
                break

    return resolved


def _eratosthenes(n: int) -> Iterator[int]:
    """
    Sieve of Eratosthenes generator.

    Args:
        n: Upper bound

    Yields:
        Prime numbers up to n
    """
    _primes = [True] * n
    _primes[0] = _primes[1] = False

    for i, is_prime in enumerate(_primes):
        if is_prime:
            yield i
            for j in range(i * i, n, i):
                _primes[j] = False


def primes(n: int) -> List[int]:
    """
    Generate list of primes up to n.

    Args:
        n: Upper bound

    Returns:
        List of primes less than n

    Example:
        >>> primes(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    return list(_eratosthenes(n))


def factors(n: int) -> List[int]:
    """
    Find all factors of n.

    Args:
        n: Number to factorize

    Returns:
        Sorted list of factors

    Example:
        >>> factors(24)
        [1, 2, 3, 4, 6, 8, 12, 24]
    """
    return sorted(
        x
        for tup in ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)
        for x in tup
    )


def md5(msg: bytes) -> str:
    """
    Calculate MD5 hash.

    Args:
        msg: Bytes to hash

    Returns:
        Hex string of hash

    Example:
        >>> md5(b"hello")
        '5d41402abc4b2a76b9719d911017c592'
    """
    m = hashlib.md5()
    m.update(msg)
    return m.hexdigest()


def sha256(msg: bytes) -> str:
    """
    Calculate SHA256 hash.

    Args:
        msg: Bytes to hash

    Returns:
        Hex string of hash

    Example:
        >>> sha256(b"hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    s = hashlib.sha256()
    s.update(msg)
    return s.hexdigest()


def knot_hash(msg: str) -> str:
    """
    Calculate Knot Hash (from Advent of Code 2017).

    Args:
        msg: Input string

    Returns:
        32-character hex string

    Example:
        >>> knot_hash("1,2,3")
        '3efbe78a8d82f29979031a4aa0b16a9d'
    """
    lengths = [ord(x) for x in msg] + [17, 31, 73, 47, 23]
    sparse = list(range(256))
    pos = skip = 0

    # 64 rounds of hashing
    for _ in range(64):
        for l in lengths:
            # Reverse sublist of length l
            for i in range(l // 2):
                x = (pos + i) % 256
                y = (pos + l - i - 1) % 256
                sparse[x], sparse[y] = sparse[y], sparse[x]

            pos = (pos + l + skip) % 256
            skip += 1

    # Dense hash
    hash_val = 0
    for i in range(16):
        res = 0
        for j in range(16):
            res ^= sparse[(i * 16) + j]
        hash_val += res << ((15 - i) * 8)

    return "%032x" % hash_val


def __test_algorithms():
    print("Testing bisect_x_int")
    assert bisect_x_int(lambda x: x > 5, 0, 10) == 6

    print("Testing bisect_x_float")
    assert abs(bisect_x_float(lambda x: x > 1.1, 0, 10) - 1.1) < 1e-9

    print("Testing hamming_distance")
    assert hamming_distance("karolin", "kathrin") == 3

    print("Testing edit_distance")
    assert edit_distance("kitten", "sitting") == 3

    print("Testing matmat")
    assert matmat([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]

    print("Testing matvec")
    assert matvec([[1, 2], [3, 4]], [5, 6]) == [17, 39]

    print("Testing matexp")
    assert matexp([[1, 1], [1, 0]], 3) == [[3, 2], [2, 1]]

    print("Testing gcd")
    assert gcd(48, 18) == 6

    print("Testing lcm")
    assert lcm(12, 15) == 60

    print("Testing egcd")
    assert egcd(17, 13) == (1, -3, 4)

    print("Testing modinv")
    assert modinv(3, 11) == 4

    print("Testing crt")
    assert crt([2, 3, 2], [3, 5, 7]) == (23, 105)

    print("Testing resolve_mapping")
    assert resolve_mapping({"a": [1, 2], "b": [1], "c": [2, 3]}) == {
        "b": 1,
        "a": 2,
        "c": 3,
    }

    print("Testing primes")
    assert primes(20) == [2, 3, 5, 7, 11, 13, 17, 19]

    print("Testing factors")
    assert factors(24) == [1, 2, 3, 4, 6, 8, 12, 24]

    print("Testing md5")
    assert md5(b"hello") == "5d41402abc4b2a76b9719d911017c592"

    print("Testing sha256")
    assert (
        sha256(b"hello")
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )

    print("Testing knot_hash")
    assert knot_hash("1,2,3") == "3efbe78a8d82f29979031a4aa0b16a9d"
    assert knot_hash("AoC 2017") == "33efeb34ea91902bb2f59c9920caa6cd"

    print("All tests passed")
