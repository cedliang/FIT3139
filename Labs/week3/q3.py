import numpy as np
from scipy.linalg import lu


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


def forward_elimination(a: np.ndarray, b: np.ndarray):
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

    def _forward_elimination(a: np.ndarray, b: np.ndarray, n: int):
        e_mat = generate_elim_matrix(a, n)
        return (np.dot(e_mat, a), np.dot(e_mat, b)) if n != len(b) - 1 else _forward_elimination(np.dot(e_mat, a), np.dot(e_mat, b), n+1)

    return _forward_elimination(a, b, 0)


if __name__ == "__main__":
    upper_triangular = np.array([[1, 2, 2], [0, -4, -6], [0, 0, -1]])
    z = np.array([3, -6, -2])

    print("Np answer:\n", np.linalg.solve(upper_triangular, z))
    print("My answer:\n", back_substitution(upper_triangular, z))

    print("Forward elimination:")
    a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]])
    b = np.array([3, 6, 10])
    u, z = forward_elimination(a, b)
    print("u\n", u)
    print("z\n", z)
