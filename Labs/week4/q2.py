import numpy as np


def ldr_decompose(a: np.ndarray):
    l, u = np.tril(a, k=-1), np.triu(a, k=1)
    return l, a-l-u, u


def jacobi_method(a: np.ndarray, b: np.ndarray, error_targ: float = 10**-12):
    def _jacobi_method(x_vec: np.ndarray, n: int, abs_errors: float):
        def x_value(j: int):
            return (1/(a[j, j]))*(b[j]-sum(map(lambda k: 0 if k == j else a[j, k]*x_vec[k], range(len(x_vec)))))

        new_x_vec = np.array(list(map(x_value, range(len(x_vec)))))
        abs_error = abs(np.linalg.norm(x_vec) - np.linalg.norm(new_x_vec))

        return "Does not converge" if (len(abs_errors) > 100 and abs_error > abs_errors[-50]) or n >= 500 else x_vec if abs_error < error_targ else _jacobi_method(new_x_vec, n+1, abs_errors + [abs_error])

    return _jacobi_method(np.array([0]*len(b)), 0, [])


def gs_method(a: np.ndarray, b: np.ndarray, error_targ: float = 10**-12):
    def _gs_method(x_vec: np.ndarray, n: int, abs_errors: float):
        def calc_x_vec(x_vec_this_iter, j):
            x_val = (1/(a[j, j]))*(b[j] - sum(
                map(lambda k: a[j, k]*x_vec[k], range(j+1, len(x_vec)))) - sum(
                map(lambda k: a[j, k]*x_vec_this_iter[k], range(j))))

            return x_vec_this_iter + [x_val] if j == len(x_vec) - 1 else calc_x_vec(x_vec_this_iter + [x_val], j+1)

        new_x_vec = calc_x_vec([], 0)
        abs_error = abs(np.linalg.norm(x_vec) - np.linalg.norm(new_x_vec))

        return "Does not converge" if (len(abs_errors) > 100 and abs_error > abs_errors[-50]) or n >= 500 else x_vec if abs_error < error_targ else _gs_method(new_x_vec, n+1, abs_errors + [abs_error])

    return _gs_method(np.array([0]*len(b)), 0, [])


if __name__ == "__main__":
    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                 [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])

    print(jacobi_method(a, b))
    print(gs_method(a, b))
    print(np.linalg.solve(a, b))