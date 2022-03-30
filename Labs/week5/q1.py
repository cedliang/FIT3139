import numpy as np


def bisection(func, a, b, tol=10**-5, max_iter=500):
    def __bisection(a, b, n):
        c = (a + b)/2
        a, b = (c, b) if np.sign(func(c)) == np.sign(func(a)) else (a, c)
        return "Cannot find root" if n >= max_iter else (c, n) if abs(a-b) < tol else __bisection(a, b, n+1)

    return __bisection(a, b, 0)


p_f1 = lambda x: (x**3)-(2*x)-5
p_f2 = lambda x: (x**3)-(3*(x**2))+(3*x)-1
e_f = lambda x: (np.e**(-1*x))-x
t_f = lambda x: (np.sin(x)*x)-1


if __name__ == "__main__":
    print(bisection(p_f1, -100, 100))
    print(bisection(e_f, -100, 100))
    print(bisection(t_f, -100, 100))
    print(bisection(p_f2, -100, 100))
