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






p_f1 = lambda x: (x**20)-(2*x)-5




coefficients = [1,
   -210,
   20615,
   -1256850,
   53327946,
   -1672280820,
   40171771630,
   -756111184500,
   11310276995381,
   -135585182899530,
   1307535010540395,
   -10142299865511450,
   63030812099294896,
   -311333643161390640,
   1206647803780373360,
   -3599979517947607200,
   8037811822645051776,
   -12870931245150988800,
   13803759753640704000,
   -8752948036761600000,
   2432902008176640000]

exponents = list(range(20, -1, -1))


zipped = list(zip(coefficients, exponents))

def p_fcrazy(x):
    print(sum(tup[0]*x**tup[1] for tup in zipped))
    return sum(tup[0]*x**tup[1] for tup in zipped)





if __name__ == "__main__":
    # print("Newtons")
    # print(find_root_newton(p_f1, dp_f1, 2))
    # print(find_root_newton(e_f, de_f, 2))
    # print(find_root_newton(t_f, dt_f, 2))
    # print(find_root_newton(p_f2, dp_f2, 2))

    # print("\nSecants")
    # print(find_root_secant(p_f1, (2, 3)))
    # print(find_root_secant(e_f, (2, 3)))
    # print(find_root_secant(t_f, (2, 3)))
    # print(find_root_secant(p_f2, (2, 3)))

    # f_fuelrod = lambda x: (1/np.tan(x))-((x**2 - 1)/(2*x))
    # print("\nFuelrod")
    # solutions = list(map(lambda n: find_root_secant(f_fuelrod, (2, n))[0], range(3, 100)))
    # print(min(filter(lambda root: root>0, solutions)))

    print(find_root_secant(p_fcrazy, (23452345, 1515)))