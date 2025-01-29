# Really hacky way to run tests for all modules

if __name__ == "__main__":
    from macros.helpers import __test_helpers

    print("===========")
    print("TESTING HELPERS.PY")
    __test_helpers()
    print("===========")

    from macros.imp import __test_imp_helpers

    print("===========")
    print("TESTING IMP_HELPERS.PY")
    __test_imp_helpers()
    print("===========")

    from macros.grid import __test_grid

    print("===========")
    print("TESTING GRID.PY")
    __test_grid()
    print("===========")

    from macros.algorithms import __test_algorithms

    print("===========")
    print("TESTING ALGORITHMS.PY")
    __test_algorithms()
    print("===========")
