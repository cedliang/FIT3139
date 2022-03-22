import itertools
import numpy as np
import random
from q2 import gs_method, jacobi_method
import timeit

#code from last week
def back_substitution(u: np.ndarray, z: np.ndarray):
    def summation_terms(n):
        return lambda i, xs: [
            xs[j]*u[i, j] for j in range(i+1, n)]

    def _back_substitution(n):
        def f(xs: list, i: int):
            xs[i] = (1/u[i, i]) * (z[i] - sum(summation_terms(n)(i, xs)))
            return xs if i == 0 else f(xs, i-1)
        return f
    return _back_substitution(len(z))([None]*len(z), len(z)-1)

def gaussian_elimination_swaps(a, b):
    u, z, e_mats, ps = forward_elimination_with_swapping(a, b)

    return back_substitution(u, z)


def swaps_col_i(a, b, i, moved_pivot_indices: list):
    column_i = list(a[:, i])
    swap_index = max(filter(lambda tup: tup[1] not in moved_pivot_indices, zip(
        column_i, range(len(column_i)))), key=lambda tup: tup[0])[1]

    if swap_index == i:
        return np.identity(len(column_i)), a, b, moved_pivot_indices
    p = np.identity(len(column_i))

    row_i = [0]*len(column_i)
    row_i[swap_index] = 1
    p[i] = np.array(row_i)
    row_swap = [0]*len(column_i)
    row_swap[i] = 1
    p[swap_index] = np.array(row_swap)

    return p, np.dot(p, a), np.dot(p, b), moved_pivot_indices + [i]


def forward_elimination_with_swapping(a: np.ndarray, b: np.ndarray):
    def generate_elim_matrix(a: np.ndarray, n: int):
        def column_value(pivot_value):
            def g(current_index, current_value):
                if current_index < n:
                    return 0
                elif current_index == n:
                    return 1
                else:
                    return -1*(current_value/pivot_value)
            return g

        elim_matrix_nth_column = np.array(list(map(column_value(
            (source_column := a[:, n])[n]), range(len(source_column)), source_column)))
        ident = np.identity(len(a[0]))
        ident[:, n] = elim_matrix_nth_column
        return ident

    def _forward_elimination(a: np.ndarray, b: np.ndarray, n: int, e_mats: list, swapped_rows: list, ps: list):
        p, pa, pb, new_swapped_rows = swaps_col_i(a, b, n, swapped_rows)
        e_mat = generate_elim_matrix(pa, n)
        return (np.dot(e_mat, pa), np.dot(e_mat, pb), e_mats + [e_mat], new_swapped_rows, ps + [p]) if n == len(b) - 1 else _forward_elimination(np.dot(e_mat, pa), np.dot(e_mat, pb), n+1, e_mats + [e_mat], new_swapped_rows, ps+[p])

    u, z, e_mats, swapped_rows, ps = _forward_elimination(a, b, 0, [], [], [])

    return u, z, e_mats, ps



#use the diagonal dominant convergence property to generate problems that are guaranteed to converge
def gen_single_sparse(size: int, prop_zeroes: float):
    x = np.array([random.uniform(-100, 100) for _ in range(size)])

    a_sing = True

    while a_sing:
        a_raw = np.random.randint(-100, 100, (size, size))

        for (idx, row), i in itertools.product(enumerate(a_raw), range(size)):
            if i != idx and random.random() < prop_zeroes:
                row[i] = 0

        a = a_raw.copy()
        for i in range(size):
            column = a_raw[:,i]

            column[i] = sum(abs(col_entry) for col_entry in column) - abs(column[i])
            a[:,i] = column

        if 0 not in np.diagonal(a):
            a_sing = False


    b = np.dot(a, x)
    return a, x, b

def generate_sparse(num_probs: int, size: int, prop_zeroes: float):
    return [gen_single_sparse(size, prop_zeroes) for _ in range(num_probs)]




#use the diagonal dominant convergence property to generate problems that are guaranteed to converge
def gen_single_dense(size):
    x = np.array([random.uniform(-100, 100) for _ in range(size)])

    a_sing = True

    while a_sing:
        a_raw = np.random.randint(-100, 100, (size, size))

        a = a_raw.copy()
        for i in range(size):
            column = a_raw[:,i]
            
            column[i] = sum(abs(col_entry) for col_entry in column) - abs(column[i])
            a[:,i] = column

        if 0 not in np.diagonal(a):
            a_sing = False


    b = np.dot(a, x)
    return a, x, b

def generate_dense(num_probs, size):
    return [gen_single_dense(size) for _ in range(num_probs)]

def solve_probs_gs():
    [gs_method(problem[0], problem[2]) for problem in probs]

def solve_probs_jac():
    [jacobi_method(problem[0], problem[2]) for problem in probs]

def solve_probs_direct():
    [gaussian_elimination_swaps(problem[0], problem[2]) for problem in probs]

if __name__ == "__main__":
    # probs = generate_dense(5, 800)

    # print(timeit.timeit(solve_probs_gs, number=1))
    # print(timeit.timeit(solve_probs_jac, number=1))
    # print(timeit.timeit(solve_probs_direct, number=1))

    probs = generate_sparse(100, 100, 0.8)
    print(timeit.timeit(solve_probs_gs, number=1))
    print(timeit.timeit(solve_probs_jac, number=1))
    print(timeit.timeit(solve_probs_direct, number=1))

    


    #some runs:
    # in general, iterative methods lose efficiency when the sparcity increases
    # however, they rapidly become faster for very large matrices (rank 1000+)

    #sparse: ratio 0.2
        # num problems:  1000
        # size:          3x3
        # time gs:       2.20 s
        # time jacobi:   3.81 s
        # time ge:       0.03 s 

    #sparse: ratio 0.4
        # num problems:  1000
        # size:          3x3
        # time gs:       2.97 s
        # time jacobi:   3.96 s
        # time ge:       0.03 s 

    #sparse: ratio 0.8
        # num problems:  1000
        # size:          3x3
        # time gs:       4.19 s
        # time jacobi:   3.42 s
        # time ge:       0.03 s 
        
        # num problems:  100
        # size:          100x100
        # time gs:       17.6 s
        # time jacobi:   26.9 s
        # time ge:        2.5 s 


    #non-sparse
        # num problems:  1000
        # size:          3x3
        # time gs:       1.25 s
        # time jacobi:   3.00 s
        # time ge:       0.02 s 

        # num problems:  1000
        # size:          4x4
        # time gs:       1.19 s
        # time jacobi:   2.49 s
        # time ge:       0.03 s 

        # num problems:  1000
        # size:          10x10
        # time gs:       2.38 s
        # time jacobi:   3.94 s
        # time ge:       0.10 s 

        # num problems:  100
        # size:          100x100
        # time gs:       12.5 s
        # time jacobi:   16.7 s
        # time ge:        1.4 s 

        # num problems:  10
        # size:          500x500
        # time gs:       25.4 s
        # time jacobi:   30.8 s
        # time ge:       22.7 s 

        # num problems:  5
        # size:          800x800
        # time gs:       30.9 s
        # time jacobi:   37.0 s
        # time ge:       55.5 s 