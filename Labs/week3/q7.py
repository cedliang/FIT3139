from q6 import forward_elimination_with_swapping
from q4 import right_to_left_dotprod
import numpy as np
import itertools
import functools


def pa_lu_factorise(a: np.ndarray, b: np.ndarray):
    u, z, e_mats, ps = forward_elimination_with_swapping(a, b)

    bad_order_l = np.linalg.inv(np.array(functools.reduce(right_to_left_dotprod, itertools.chain.from_iterable(map(list, (zip(ps, e_mats)))))))
    combined_p = np.array(functools.reduce(right_to_left_dotprod, ps))
    l = np.dot(combined_p, bad_order_l)

    return combined_p, l, u

if __name__ == "__main__":
    a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]])
    b = np.array([3, 6, 10])

    p, l, u = pa_lu_factorise(a, b)

    print("p\n", p)
    print("a\n", a)
    print("l\n", l)
    print("u\n", u)