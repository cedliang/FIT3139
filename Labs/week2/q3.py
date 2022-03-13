import math
import itertools

#curried approach
def e_x_nthterm(x: int):
    def f(n: int):
        return (x**n)/math.factorial(n)
    return f

#generator approach
def e_x_nthterm_gen(x: int):
    n = 0
    while True:
        yield (x**n)/math.factorial(n)
        n += 1


def fixed_num_terms(x: int, num_terms: int):
    current_approx = 0

    for nth_term in itertools.islice(e_x_nthterm_gen(x), num_terms):
        last_approx = current_approx
        current_approx += nth_term
        approx_error = (current_approx - last_approx)/current_approx

        print(approx_error)
    print("Final approximation", current_approx)


# iterates until the error for that iteration is lower than 10^tolerance_exponent
# tolerance_exponent should be a negative integer for best results, ie -12
def error_tolerance(x: int, tolerance_exponent: int):
    current_approx = 0
    series_term_generator = e_x_nthterm_gen(x)
    n = 0
    firstLoop = True

    while firstLoop or approx_error > 10**tolerance_exponent:
        last_approx = current_approx
        current_approx += next(series_term_generator)
        approx_error = (current_approx - last_approx)/current_approx

        if firstLoop:
            firstLoop = False

        n += 1

    print("Numterms used:", n)
    print("Final approximation", current_approx)


if __name__ == "__main__":
    #fixed_num_terms(10,40)

    error_tolerance(10, -16)
