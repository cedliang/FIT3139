import numpy as np
import itertools



# selects the relevant lambda from the lambda function array based on an index and applies it, returning a single value in f(x) or j(x)
# nested lambdas to ensure lazy evaluation - select the function from the array and apply it, instead of selecting computed values from a full map
fmat_apply = lambda fmat: lambda *xs: lambda idx: fmat[idx](*xs)
jfmat_apply = lambda jfmat: lambda *xs: lambda idx0, idx1: jfmat[idx0][idx1](*xs)


# maps the application over all the elems in x, computing f(x) and j(x)
fx = lambda fmat, x: list(map(fmat_apply(fmat)(*x), range(len(x))))
jfx = lambda jfmat, x: [list(map(lambda idxtup: jfmat_apply(jfmat)(
    *x)(*idxtup), itertools.product(range(len(x)), range(len(x)))))[i:i+len(x)] for i in range(0, len(x)**2, len(x))]


def multi_newtons(fmat, jfmat, init, tol=10**-9, max_iter=800):
    def __multi_newtons(x, n):
        h = np.linalg.solve(np.array(jfx(jfmat, list(x))), 
                         -1*np.array(fx(fmat, list(x))))
        new_x = x + h
        return "Does not converge" if n >= max_iter else (new_x, n) if np.sqrt(sum(map(lambda h_i: h_i**2, list(h)))) < tol else __multi_newtons(new_x, n+1)

    return __multi_newtons(init, 0)


if __name__ == "__main__":
    fmat = [lambda x_0, x_1: x_0**2 + x_1**2 - 4,
            lambda x_0, x_1: np.e**x_0 + np.e**x_1 - 2]

    jfmat = [[lambda x_0, _: 2*x_0, lambda _, x_1: 2*x_1],
             [lambda x_0, _: np.e**x_0, lambda _, x_1: np.e**x_1]]

    print(multi_newtons(fmat, jfmat, [0, 1]))


    