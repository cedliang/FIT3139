from q1 import p_f1, p_f2, t_f, e_f
import numpy as np


def find_root_newton(func, dfunc, init, tol = 10**-9, max_iter=500):
    def __recurse_newton(x, n):
        new_x = x - func(x)/dfunc(x)
        return "Does not converge" if n >= max_iter else (new_x, n) if abs(new_x - x) < tol else __recurse_newton(new_x, n+1)

    return __recurse_newton(init, 0)


def find_root_secant(func, inits, tol = 10**-9, max_iter=500):
    def __recurse_secant(x_minus, x, n):
        new_x = x-(func(x)*(x-x_minus))/(func(x)-func(x_minus))
        return "Does not converge" if n >= max_iter else (new_x, n) if abs(new_x - x) < tol else __recurse_secant(x, new_x, n+1)

    return __recurse_secant(*inits, 0)


dp_f1 = lambda x: 3*(x**2) - 2
dp_f2 = lambda x: 3*(x**2) - 6*x + 3
de_f = lambda x: -(np.e**(-x)) - x
dt_f = lambda x: np.sin(x) + x*np.cos(x)


if __name__ == "__main__":
    print("Newtons")
    print(find_root_newton(p_f1, dp_f1, 2))
    print(find_root_newton(e_f, de_f, 2))
    print(find_root_newton(t_f, dt_f, 2))
    print(find_root_newton(p_f2, dp_f2, 2))

    print("\nSecants")
    print(find_root_secant(p_f1, (2, 3)))
    print(find_root_secant(e_f, (2, 3)))
    print(find_root_secant(t_f, (2, 3)))
    print(find_root_secant(p_f2, (2, 3)))
