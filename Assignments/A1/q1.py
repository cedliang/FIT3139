from progress.bar import Bar
import sys
import copy
import itertools
import functools
import numpy as np
import operator
from multiprocess import Pool


class ArrayFloat:
    mult_table = [[([], 0), ([], 0), ([], 0), ([], 0), ([], 0), ([], 0), ([], 0), ([], 0), ([], 0), ([], 0)],
                  [([], 0), ([1], 0), ([2], 0), ([3], 0), ([4], 0),
                   ([5], 0), ([6], 0), ([7], 0), ([8], 0), ([9], 0)],
                  [([], 0), ([2], 0), ([4], 0), ([6], 0), ([8], 0), ([1], 1),
                   ([1, 2], 0), ([1, 4], 0), ([1, 6], 0), ([1, 8], 0)],
                  [([], 0), ([3], 0), ([6], 0), ([9], 0), ([1, 2], 0), ([
                      1, 5], 0), ([1, 8], 0), ([2, 1], 0), ([2, 4], 0), ([2, 7], 0)],
                  [([], 0), ([4], 0), ([8], 0), ([1, 2], 0), ([1, 6], 0),
                   ([2], 1), ([2, 4], 0), ([2, 8], 0), ([3, 2], 0), ([3, 6], 0)],
                  [([], 0), ([5], 0), ([1], 1), ([1, 5], 0), ([2], 1),
                   ([2, 5], 0), ([3], 1), ([3, 5], 0), ([4], 1), ([4, 5], 0)],
                  [([], 0), ([6], 0), ([1, 2], 0), ([1, 8], 0), ([2, 4], 0),
                   ([3], 1), ([3, 6], 0), ([4, 2], 0), ([4, 8], 0), ([5, 4], 0)],
                  [([], 0), ([7], 0), ([1, 4], 0), ([2, 1], 0), ([2, 8], 0), ([
                      3, 5], 0), ([4, 2], 0), ([4, 9], 0), ([5, 6], 0), ([6, 3], 0)],
                  [([], 0), ([8], 0), ([1, 6], 0), ([2, 4], 0), ([3, 2], 0),
                   ([4], 1), ([4, 8], 0), ([5, 6], 0), ([6, 4], 0), ([7, 2], 0)],
                  [([], 0), ([9], 0), ([1, 8], 0), ([2, 7], 0), ([3, 6], 0), ([4, 5], 0), ([5, 4], 0), ([6, 3], 0), ([7, 2], 0), ([8, 1], 0)]]

    def __init__(self, dec_str, direct_constructor=None, precision=20):
        self.precision = precision

        self.array, self.exponent, self.sign = direct_constructor or self.from_str(
            dec_str)

    # rounding error exists only for the shorthand string representation
    def __str__(self):
        return self.sign + str(int("".join(map(str, self.array))) * 10 ** self.exponent) if self.array else "0."

    def __repr__(self):
        return self.sign + str(int("".join(map(str, self.array))) * 10 ** self.exponent) if self.array else "0."

    def from_str(self, dec_str: str) -> list:
        if dec_str[0] not in ["-", "+"]:
            sign = "+"
        else:
            sign = dec_str[0]
            dec_str = dec_str[1:]

        if "." not in dec_str:
            dec_str += "."

        # remove leading and trailing zeros
        nonzero_elems_filt = list(filter(lambda char: char != "0", dec_str))

        first_nonzero_index = dec_str.index(nonzero_elems_filt[0])
        last_nonzero_index = len(
            dec_str) - 1 - dec_str[::-1].index("".join(nonzero_elems_filt)[-1])

        dec_str = dec_str[first_nonzero_index:last_nonzero_index+1]

        # these operations actually change the number's meaning - so need to track changes with exponent
        exponent = 0

        # first move decimal point to the rightmost
        num_dig_after_dec = len(dec_str) - dec_str.index(".") - 1

        dec_str = dec_str.replace(".", "") + "."
        exponent = -num_dig_after_dec

        # remove leading zeros
        first_nonzero_index = dec_str.index(
            next(iter(filter(lambda char: char != "0", dec_str))))
        dec_str = dec_str[first_nonzero_index:]

        # then move it left until you get a nonzero
        if dec_str != ".":
            while dec_str[dec_str.index(".") - 1] == "0":
                dec_str = dec_str.replace("0.", ".0")
                exponent += 1

        # remove trailing zeros
        last_nonzero_index = len(dec_str) - 1 - dec_str[::-1].index(
            "".join(list(filter(lambda char: char != "0", dec_str)))[-1])
        dec_str = dec_str[:last_nonzero_index+1]

        # make array
        digs_array = list(map(int, dec_str[:-1]))

        if not digs_array:
            sign = "+"

        # truncate to precision limit
        if len(digs_array) > self.precision:
            truncated_digs = len(digs_array) - self.precision
            digs_array = digs_array[:self.precision]
            exponent += truncated_digs

            # remove trailing zeros from list again to reduce computation time needed for operations
            last_nonzero_index = np.max(np.nonzero(digs_array))
            before_len = len(digs_array)
            digs_array = digs_array[:last_nonzero_index + 1]
            exponent += before_len - len(digs_array)

        return digs_array, exponent, sign

    def __eq__(self, other):
        if self.array or other.array:
            return self.array == other.array and self.exponent == other.exponent and self.sign == other.sign
        else:
            return True

    def __lt__(self, other):
        if self == other:
            return False
        if self.sign != other.sign:
            if self.sign == "-":
                return True
            else:
                return False

        if self.array == []:
            return other.sign == "+" and bool(other.array)
        if other.array == []:
            return self.sign == "-" and bool(self.array)

        def recursive_array_compare(a1, a2, e1, e2):
            a1_dig_exp = len(a1) - 1 + e1
            a2_dig_exp = len(a2) - 1 + e2

            if not a1:
                return True
            if not a2:
                return False
            if a1_dig_exp < a2_dig_exp:
                return True
            if a2_dig_exp < a1_dig_exp:
                return False
            if a1[0] < a2[0]:
                return True
            if a1[0] > a2[0]:
                return False

            # same dig - recurse
            a1.pop(0)
            a2.pop(0)

            return recursive_array_compare(a1, a2, e1, e2)

        rac = recursive_array_compare(
            copy.deepcopy(self.array), copy.deepcopy(other.array), self.exponent, other.exponent)
        return rac if self.sign != "-" else not rac

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return not self < other

    def __gt__(self, other):
        return not self < other and self != other

    def abs(self):
        return ArrayFloat("", direct_constructor=(self.array, self.exponent, "+"), precision=self.precision)

    def neg(self):
        return ArrayFloat("", direct_constructor=(self.array, self.exponent, "+" if self.sign == "-" else "-"), precision=self.precision)

    def add_vals(self, other):
        def add_procedure(a1, a2):
            total_digits = len(a1)
            output_array = [None]*total_digits
            acc = 0
            for i in range(total_digits-1, -1, -1):
                sum_digs = a1[i] + a2[i] + acc
                acc = 0
                if sum_digs >= 10:
                    sum_digs -= 10
                    acc = 1
                output_array[i] = sum_digs

            return [acc] + output_array if acc else output_array

        def sub_procedure(a1, a2):
            total_digits = len(a1)
            output_array = [None]*total_digits
            acc = 0
            for i in range(total_digits - 1, -1, -1):
                diff_digs = a1[i] - a2[i] - acc
                acc = 0
                if diff_digs < 0:
                    diff_digs += 10
                    acc = 1
                output_array[i] = diff_digs

            return output_array

        def gen_adjust_mag_array(array, min_exponent, common_exponent):
            # fully contained
            if common_exponent <= min_exponent:
                return array + [0]*(min_exponent - common_exponent)

            number_discard = common_exponent - min_exponent
            if number_discard >= len(array):
                return []

            return array[:-number_discard]

        mag_range_self = (len(self.array) - 1 + self.exponent, self.exponent)
        mag_range_other = (len(other.array) - 1 +
                           other.exponent, other.exponent)

        # range of magnitudes for which we care about calculating
        mag_range_overlap = (max(mag_range_self[0], mag_range_other[0]), max(min(
            mag_range_self[1], mag_range_other[1]), max(mag_range_self[0], mag_range_other[0]) - self.precision))

        # standardise these arrays for the range of magnitudes we care about
        common_exponent = mag_range_overlap[1]

        # if a value's mag range is completely enclosed by the range (ie if the common exponent is <= min_mag_range of value)
        adjusted_self = gen_adjust_mag_array(
            self.array, self.exponent, common_exponent)
        adjusted_other = gen_adjust_mag_array(
            other.array, other.exponent, common_exponent)

        # fill with zeros
        adjusted_self = [0]*(max(len(adjusted_self), len(adjusted_other)
                                 ) - len(adjusted_self)) + adjusted_self
        adjusted_other = [0]*(max(len(adjusted_self), len(adjusted_other)
                                  ) - len(adjusted_other)) + adjusted_other

        if self.sign == other.sign:
            result_sign = self.sign
            procedure_result_array = add_procedure(
                adjusted_self, adjusted_other)

        else:
            # subtract smaller abs value from bigger abs value - and flip the sign if they don't match up
            self_abs_greater = self.abs() > other.abs()
            result_sign = "+"
            if (self_abs_greater and self.sign == "-") or (not self_abs_greater and self.sign == "+"):
                result_sign = "-"

            procedure_result_array = sub_procedure(
                adjusted_self, adjusted_other) if self_abs_greater else sub_procedure(adjusted_other, adjusted_self)

        if common_exponent < 0:
            procedure_result_array.insert(common_exponent, ".")
        elif common_exponent > 0:
            procedure_result_array += [0]*common_exponent

        result_str = result_sign + "".join(map(str, procedure_result_array))
        return ArrayFloat(result_str, precision=self.precision)

    def __add__(self, other):
        return self.add_vals(other)

    def __sub__(self, other):
        return self.add_vals(other.neg())

    def __mul__(self, other):
        def get_digit_prod(a, b, add_exponent):
            return ArrayFloat("", direct_constructor=((tup := self.mult_table[a][b])[0], tup[1]+add_exponent, "+"), precision=self.precision)

        # mult by zero case
        if not self.array or not other.array:
            return ArrayFloat("0", precision=self.precision)

        final_sign = '-' if self.sign != other.sign else '+'

        # tuple in self_digit_vals: (digit, exponent)
        self_digit_vals = []
        self_array_mask = list(range(len(self.array)-1, -1, -1))
        for i in range(len(self.array)):
            self_digit_vals.append(
                (self.array[i], self_array_mask[i]+self.exponent))

        other_digit_vals = []
        other_array_mask = list(range(len(other.array)-1, -1, -1))
        for i in range(len(other.array)):
            other_digit_vals.append(
                (other.array[i], other_array_mask[i]+other.exponent))

        # cartesian product of tuples
        product_sums_tuples = list(itertools.product(
            self_digit_vals, other_digit_vals))

        highest_magnitude = product_sums_tuples[0][0][1] + \
            product_sums_tuples[0][1][1]
        removed_imprecise_elems = filter(
            lambda tup: tup[0][1]+tup[1][1] + self.precision >= highest_magnitude, product_sums_tuples)
        removed_zeros = filter(
            lambda tup: tup[0][0] != 0 and tup[1][0] != 0, removed_imprecise_elems)

        product_sums = list(map(lambda tup: get_digit_prod(
            tup[0][0], tup[1][0], tup[0][1] + tup[1][1]), removed_zeros))

        summed = functools.reduce(operator.add, product_sums)

        return ArrayFloat("", direct_constructor=(summed.array, summed.exponent, final_sign), precision=self.precision)


def find_root_newton(new_x_func, init, tol, max_iter=500):
    def __recurse_newton(x, n):
        new_x = x - new_x_func(x)

        print(new_x)
        return "Does not converge" if n >= max_iter else (new_x, n) if new_x_func(x).abs() < tol else __recurse_newton(new_x, n+1)

    return __recurse_newton(init, 0)


def mod_coeffs_third_term(k, mod=True):
    mod_amt = 10 ** k if mod else 0
    return list(map(lambda i: ArrayFloat(i, precision=50), map(str, [
        1,
        -210,
        20615 + mod_amt,
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
        2432902008176640000])))


def crazy_func(k, mod=False):
    def f(x):
        elems_list = []
        for idx, power in enumerate(range(20, -1, -1)):
            x_arr = [x]*power
            x_power = (functools.reduce(operator.mul, x_arr,
                                        ArrayFloat("1", precision=50)))
            elems_list.append(mod_coeffs_third_term(k, mod)[idx]*x_power)

        return functools.reduce(operator.add, elems_list, ArrayFloat("0", precision=50))
    return f


if __name__ == "__main__":
    # a = ArrayFloat("3.1415926565897932384626")
    # b = ArrayFloat("2.7182818284590452353602")

    # print(a + b)
    # print(a * b)

    # 1d
    # derived from x - f(x)/f'(x)
    # def new_x_func(x): return (x*x*x) - ArrayFloat("0.5", precision=54)*x
    # ten_power_fifty = ArrayFloat(
    #     "0.000000000000000000000000000000000000000000000000001", precision=54)

    # print(find_root_newton(new_x_func, ArrayFloat(
    #     "1", precision=54), ten_power_fifty)[0])

    # for a convergent case, we expect the difference between one iteration to the next to decrease over iterations.
    # if the magnitude of the difference between iterations has reached the level of 10^50, then we would not expect
    # following iterations to have enough of an impact on the estimation to deviate it from that value.

    # q1e - let this run, it takes about 5 seconds on my machine, but depends on your multithreading capabilities
    roots = range(1, 21)
    k_range = range(-20, 10)
    
    all_probs = itertools.product(k_range, roots)

    def v_k_tup_map(v_k_tup):
        return crazy_func(v_k_tup[0], mod=True)(
            ArrayFloat(str(v_k_tup[1]), precision=50))

    with Pool() as pool:
        results_eval = list(pool.map(v_k_tup_map, all_probs))

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    chunk_results = list(zip(k_range, chunks(results_eval, 20)))
    chunk_result_iter = iter(chunk_results)

    while str((next_elem := next(chunk_result_iter))[1]) == "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]":
        print(next_elem[0], next_elem[1])

    # this should print the first element that isn't empty
    print(next_elem[0], next_elem[1])

    print((next_elem := next(chunk_result_iter))[0], next_elem[1])
    print((next_elem := next(chunk_result_iter))[0], next_elem[1])
    print((next_elem := next(chunk_result_iter))[0], next_elem[1])
    print((next_elem := next(chunk_result_iter))[0], next_elem[1])
