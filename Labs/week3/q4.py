from q3 import generate_elimination_matrices, back_substitution, forward_elimination
import numpy as np
import functools


def right_to_left_dotprod(n, n_plus_1): return np.dot(n_plus_1, n)


def l_u_factorise(a, b):
    e_mats = generate_elimination_matrices(a, b)
    u = functools.reduce(right_to_left_dotprod, [a]+e_mats)
    l = functools.reduce(right_to_left_dotprod, map(
        np.linalg.inv, list(reversed(e_mats))))
    return l, u


def solve_by_lu_fac(a, b):
    l, u = l_u_factorise(a, b)
    ys = np.dot(np.linalg.inv(l), b)
    return back_substitution(u, ys)


if __name__ == "__main__":
    a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]])
    b = np.array([3, 6, 10])

    l, u = l_u_factorise(a, b)
    print("l\n", l)
    print("u\n", u)

    print("original a\n", a)
    print("derived a from l and u\n", np.dot(l, u))

    print("lu fac answer:\n", solve_by_lu_fac(a, b))
