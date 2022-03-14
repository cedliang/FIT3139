import numpy as np


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


if __name__ == "__main__":
    upper_triangular = np.array([[1, 2, 2], [0, -4, -6], [0, 0, -1]])
    z = np.array([3, -6, -2])

    print("Np answer:\n", np.linalg.solve(upper_triangular, z))
    print("My answer:\n", back_substitution(upper_triangular, z))
