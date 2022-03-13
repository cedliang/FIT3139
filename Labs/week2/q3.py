import math
import itertools
import operator


def e_x_nthterm(x: int):
    return lambda n: (x**n)/math.factorial(n)

def e_x_nthterm_gen(x: int):
    n = 0
    while True:
        yield (x**n)/math.factorial(n)
        n += 1


def fixed_num_map(x: int, num_terms: int):
    def cumulative_sum_list_curried(acc):
        return lambda list_to_sum: ([acc+list_to_sum[0]] + cumulative_sum_list_curried(acc+list_to_sum[0])(list_to_sum[1:]) if list_to_sum != [] else [])

    cumulative_sum = cumulative_sum_list_curried(0)(
        series_terms := list(map(e_x_nthterm(x), list(range(num_terms)))))
    errors = list(map(lambda tup: operator.truediv(
        tup[0], tup[1]), zip(series_terms, cumulative_sum)))

    return series_terms, cumulative_sum, errors


def fixed_num_terms(x: int, num_terms: int):
    current_approx = 0

    for nth_term in itertools.islice(e_x_nthterm_gen(x), num_terms):
        last_approx = current_approx
        current_approx += nth_term
        approx_error = (current_approx - last_approx)/current_approx

        print(approx_error)
    print("Final approximation", current_approx)


def error_tolerance(x: int, tolerance_exponent: int):
    def _error_tolerance_tail_rec(series_terms, cumulative_sum, errors, n):
        if series_terms and errors[-1] < 10**tolerance_exponent:
            return series_terms, cumulative_sum, errors
        return _error_tolerance_tail_rec((new_series_terms := series_terms+[(current_term := e_x_nthterm(x)(n))]),
                                         (new_cum_sum := cumulative_sum+[
                                          cumulative_sum[-1]+current_term] if cumulative_sum else [current_term]),
                                         errors +
                                         [new_series_terms[-1]/new_cum_sum[-1]],
                                         n+1)
    return _error_tolerance_tail_rec([], [], [], 0)


if __name__ == "__main__":
    # fixed_num_terms(10,40)

    #list(map(lambda tuple: print(tuple[-1]), fixed_num_map(10, 40)))

    list(map(lambda tuple: print(tuple[-1]),
         error_tolerance(10, -16)))
