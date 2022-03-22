import numpy as np


def scale_vector(x):
    return x if 0 in x else (1/min(x, key=abs))*x


def power_method(a, error_targ=10**-10):
    def find_eigenvector(a, error_targ):
        def _find_eigenvector(x, n):
            new_x = np.dot(a, x)

            x_norm = np.linalg.norm(scale_vector(x))
            new_x_norm = np.linalg.norm(scale_vector(new_x))
            return "No dominant eigenvalue, does not converge" if n > 900 else new_x if abs(x_norm - new_x_norm) < error_targ else _find_eigenvector((1/np.linalg.norm(new_x))*new_x, n+1)

        r = _find_eigenvector(np.array([1, 0]), 0)
        if isinstance(r, str):
            return r
        return scale_vector(r)

    def find_eigenvalue(a, v):
        return np.dot(np.dot(a, v), v)/np.dot(v, v)

    v = find_eigenvector(a, error_targ)
    if isinstance(v, str):
        return v
    return (v, find_eigenvalue(a, v))


if __name__ == "__main__":
    a = np.array([[3, 1], [1, 3]])

    v = power_method(a)

    print(v)
