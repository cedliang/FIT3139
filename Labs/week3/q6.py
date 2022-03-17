import numpy as np


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


def print_elems_newline(iterable):
    list(map(print, iterable))


if __name__ == "__main__":
    a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]])
    b = np.array([3, 6, 10])

    u, z, e_mats, ps = forward_elimination_with_swapping(a, b)

    print("u\n", u, "\nz\n", z)

    print("elimination matrices")
    print_elems_newline(e_mats)

    print("permutation matrices")
    print_elems_newline(ps)
