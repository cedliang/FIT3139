import copy
import itertools
import functools
from unicodedata import numeric
from unittest import result
import numpy as np

class ArrayFloat:
    def __init__(self, initStr):
        self.repr_array = self.from_str(str(initStr))

    def __str__(self):
        return self.to_str(self.repr_array)

    def __mul__(self, other):
        return ArrayFloat(self.to_str(self.mult_floats(self.repr_array, other.repr_array)))

    def __add__(self, other):
        return ArrayFloat(self.to_str(self.add_floats(self.repr_array, other.repr_array)))

    def __sub__(self, other):
        return ArrayFloat(self.to_str(self.add_floats(self.repr_array, (ArrayFloat("-1")*other).repr_array)))

    def __eq__(self, other):
        return self.repr_array == other.repr_array

    def __lt__(self, other):
        return self.repr_array != other.repr_array and (self - other).repr_array[0] == '-'

    def __gt__(self, other):
        return self.repr_array != other.repr_array and (self - other).repr_array[0] == '+'

    def __ne__(self, other):
        return not self == other

    def __ge__(self, other):
        return self == other or self > other

    def __le__(self, other):
        return self == other or self < other

    def abs(self):
        return ArrayFloat((ArrayFloat(-1)*self).__str__()) if self < ArrayFloat(0) else self

    def from_str(self, dec_str):
        if dec_str[0] not in ["-", "+"]:
            dec_str = f"+{dec_str}"
        if "." not in dec_str:
            dec_str += "."

        return_array = []
        acc = []
        for char in dec_str:
            if char in ["-", "+"]:
                return_array.append(char)
            elif char == ".":
                return_array.append(acc)
                acc = []
            else:
                acc.append(int(char))
        return_array.append(acc)

        # remove leading zeros in elems before dec, trailing zeros in elems after
        filtered_int_elems = list(
            filter(lambda dig: dig != 0, return_array[1]))
        stripped_int_elems = return_array[1][return_array[1].index(
            filtered_int_elems[0]):] if filtered_int_elems else []

        reversed_fracs = list(reversed(return_array[2]))
        filtered_frac_elems_reversed = list(
            filter(lambda dig: dig != 0, reversed_fracs))
        stripped_frac_elems = list(reversed(reversed_fracs[reversed_fracs.index(
            filtered_frac_elems_reversed[0]):])) if filtered_frac_elems_reversed else []

        return [return_array[0], stripped_int_elems, stripped_frac_elems]

    def to_str(self, n: list):
        n_copy = copy.deepcopy(n)
        if not n_copy[1]:
            n_copy[1].append("0")
        dec_point = "." if n_copy[2] else ""
        sign = n_copy[0]
        stringified = list(
            map(str, [sign] + n_copy[1] + [dec_point] + n_copy[2]))
        return "".join(stringified)

    def add_floats(self, a: list, b: list):
        if a[1] == b[1] and a[2] == b[2] and a[0] != b[0]:
            return ["+", [], []]

        longest_int = max(len(a[1]), len(b[1]))
        longest_dec = max(len(a[2]), len(b[2]))

        # pad out the arrays such that they have leading zeroes for integer components or trailing zeros for decimal components
        padded_a = (a[0], (longest_int - len(a[1]))*[0] +
                    a[1], a[2] + (longest_dec - len(a[2]))*[0])
        padded_b = (b[0], (longest_int - len(b[1]))*[0] +
                    b[1], b[2] + (longest_dec - len(b[2]))*[0])

        integer_digits = len(padded_a[1])
        decimal_digits = len(padded_a[2])
        total_digits = integer_digits + decimal_digits

        a_digs = padded_a[1] + padded_a[2]
        b_digs = padded_b[1] + padded_b[2]

        # addition - but also takes care of the case of both numbers being negative
        if a[0] == b[0]:
            output_array = [None]*total_digits
            acc = 0
            for i in range(total_digits-1, -1, -1):
                sum_digs = a_digs[i] + b_digs[i] + acc
                acc = 0
                if sum_digs >= 10:
                    sum_digs -= 10
                    acc = 1
                output_array[i] = sum_digs

            return_int = [acc] + output_array[:integer_digits]
            return_dec = output_array[integer_digits:]

            return self.from_str("".join(map(str, [a[0]] + return_int + ['.'] + return_dec)))

        # subtraction case - with precondition: a != b

        # we always want to subtract from the object with the biggest absolute value to the smallest absolute value.
        # if these do not correspond to the signage, we can perform the subtraction, then add an '-' to the front at the end
        first_diff_digit_pair = next(
            iter(filter(lambda tup: tup[0] != tup[1], zip(a_digs, b_digs))))
        a_greater = first_diff_digit_pair[0] > first_diff_digit_pair[1]

        greater_digits = a_digs if a_greater else b_digs
        lesser_digits = b_digs if a_greater else a_digs
        final_sign = a[0] if a_greater else b[0]

        output_array = [None]*total_digits
        acc = 0
        for i in range(total_digits - 1, -1, -1):
            diff_digs = greater_digits[i] - lesser_digits[i] - acc
            acc = 0
            if diff_digs < 0:
                diff_digs += 10
                acc = 1
            output_array[i] = diff_digs

        return_int = output_array[:integer_digits]
        return_dec = output_array[integer_digits:]

        return self.from_str("".join(map(str, [final_sign] + return_int + ['.'] + return_dec)))

    # we could do this by repeated addition all the way up to the numbers with many digits - but we have access to memory, which means there's a better way to do this...

    def mult_floats(self, a: list, b: list):
        # works for positive integers
        def raw_repeated_addition(a, b):
            if (not a[1] and not a[2]) or (not b[1] and not b[2]):
                return ["+", [], []]
            minus_one = ["-", [1], []]
            acc = ["+", [], []]
            counter = copy.deepcopy(b)
            while counter != ["+", [], []]:
                acc = self.add_floats(acc, a)
                counter = self.add_floats(counter, minus_one)

            return acc

        def generate_multiplication_table():
            table_results = [[None]*11 for _ in range(11)]

            for a in range(11):
                a_array_repr = self.from_str(str(a))
                for b in range(11):
                    b_array_repr = self.from_str(str(b))
                    table_results[a][b] = raw_repeated_addition(
                        a_array_repr, b_array_repr)

            return table_results

        # multiplies an INTEGER by a power of ten
        def mult_by_power_of_ten(n, exponent):
            return [n[0], n[1] + [0]*exponent, n[2]]

        # mult by zero
        if (not a[1] and not a[2]) or (not b[1] and not b[2]):
            return ["+", [], []]

        final_sign = '-' if a[0] != b[0] else '+'
        num_decs = len(a[2]) + len(b[2])

        # convert decimal removeds to their array representation manually
        a_digs_repr = a[1] + a[2]
        b_digs_repr = b[1] + b[2]

        longest_int = max(len(a_digs_repr), len(b_digs_repr))

        padded_a_digs = [0]*(longest_int - len(a_digs_repr)) + a_digs_repr
        padded_b_digs = [0]*(longest_int - len(b_digs_repr)) + b_digs_repr

        mult_table = generate_multiplication_table()
        place_value_table = list(reversed(range(longest_int)))

        sums_list = []

        for i, j in itertools.product(range(longest_int), range(longest_int)):
            sums_list.append(mult_by_power_of_ten(
                mult_table[padded_a_digs[i]][padded_b_digs[j]], place_value_table[i] + place_value_table[j]))

        reduced_list = functools.reduce(self.add_floats, sums_list)[1]

        # add decimals back in
        if num_decs >= len(reduced_list):
            zeros_pad = max(0, num_decs - len(reduced_list))
            r_string = self.to_str(
                [final_sign, [], [0]*zeros_pad + reduced_list])

        elif num_decs == 0:
            r_string = self.to_str([final_sign, reduced_list, []])

        else:
            r_string = self.to_str(
                [final_sign, reduced_list[:-num_decs], reduced_list[-num_decs:]])

        if len(r_string) > 100:
            r_string = r_string[:100]

        return self.from_str(r_string)


def find_root_newton(new_x_func, init, tol=10**-50, max_iter=500):
    def __recurse_newton(x, n):
        new_x = x - new_x_func(x)

        print(new_x)
        return "Does not converge" if n >= max_iter else (new_x, n) if (new_x - x).abs() < tol else __recurse_newton(new_x, n+1)

    return __recurse_newton(init, 0)








class OptimisedArrayFloat:
    def __init__(self, dec_str, direct_constructor = None, precision = 20):
        self.precision = precision

        self.array, self.exponent, self.sign = direct_constructor or self.from_str(dec_str)

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

        #remove leading and trailing zeros
        nonzero_elems_filt = list(filter(lambda char: char != "0", dec_str))

        first_nonzero_index = dec_str.index(nonzero_elems_filt[0])
        last_nonzero_index = len(dec_str) - 1 - dec_str[::-1].index("".join(nonzero_elems_filt)[-1])

        dec_str = dec_str[first_nonzero_index:last_nonzero_index+1]

        # these operations actually change the number's meaning - so need to track changes with exponent
        exponent = 0

        # first move decimal point to the rightmost
        num_dig_after_dec = len(dec_str) - dec_str.index(".") - 1

        dec_str = dec_str.replace(".", "") + "."
        exponent = -num_dig_after_dec

        # remove leading zeros
        first_nonzero_index = dec_str.index(next(iter(filter(lambda char: char != "0", dec_str))))
        dec_str = dec_str[first_nonzero_index:]

        # then move it left until you get a nonzero
        if dec_str != ".":
            while dec_str[dec_str.index(".") - 1] == "0":
                dec_str = dec_str.replace("0.", ".0")
                exponent += 1

        #remove trailing zeros
        last_nonzero_index = len(dec_str) - 1 - dec_str[::-1].index("".join(list(filter(lambda char: char != "0", dec_str)))[-1])
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

        rac = recursive_array_compare(self.array, other.array, self.exponent, other.exponent)
        return rac if self.sign != "-" else not rac

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return not self < other
    
    def __gt__(self, other):
        return not self < other and self != other

    def abs(self):
        return OptimisedArrayFloat("", direct_constructor = (self.array, self.exponent, "+"), precision = self.precision)

    def neg(self):
        return OptimisedArrayFloat("", direct_constructor = (self.array, self.exponent, "+" if self.sign == "-" else "-"), precision=self.precision)

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
        mag_range_other = (len(other.array) - 1 + other.exponent, other.exponent)

        # range of magnitudes for which we care about calculating
        mag_range_overlap = (max(mag_range_self[0], mag_range_other[0]), max(min(mag_range_self[1], mag_range_other[1]), max(mag_range_self[0], mag_range_other[0]) - self.precision))

        # standardise these arrays for the range of magnitudes we care about
        common_exponent = mag_range_overlap[1]

        # if a value's mag range is completely enclosed by the range (ie if the common exponent is <= min_mag_range of value)
        adjusted_self = gen_adjust_mag_array(self.array, self.exponent, common_exponent)
        adjusted_other = gen_adjust_mag_array(other.array, other.exponent, common_exponent)

        #fill with zeros
        adjusted_self = [0]*(max(len(adjusted_self), len(adjusted_other)) - len(adjusted_self)) + adjusted_self
        adjusted_other = [0]*(max(len(adjusted_self), len(adjusted_other)) - len(adjusted_other)) + adjusted_other

        if self.sign == other.sign:
            result_sign = self.sign
            procedure_result_array = add_procedure(adjusted_self, adjusted_other)

        else:
            # subtract smaller abs value from bigger abs value - and flip the sign if they don't match up 
            self_abs_greater = self.abs() > other.abs()
            result_sign = "+"
            if (self_abs_greater and self.sign == "-") or (not self_abs_greater and self.sign == "+"):
                result_sign = "-"

            procedure_result_array = sub_procedure(adjusted_self, adjusted_other) if self_abs_greater else sub_procedure(adjusted_other, adjusted_self)

        if common_exponent < 0:
            procedure_result_array.insert(common_exponent, ".")
        elif common_exponent > 0:
            procedure_result_array += [0]*common_exponent

        result_str = result_sign + "".join(map(str, procedure_result_array))
        return OptimisedArrayFloat(result_str, precision=self.precision)

    def __add__(self, other):
        return self.add_vals(other)

    def __sub__(self, other):
        return self.add_vals(other.neg())


if __name__ == "__main__":
    # a = ArrayFloat("3.1415926565897932384626")
    # b = ArrayFloat("2.7182818284590452353602")

    # print(a + b)
    # print(a * b)

    # 1d
    # derived from x - f(x)/f'(x)
    # def new_x_func(x): return (x*x*x) - ArrayFloat(0.5)*x
    # ten_power_fifty = ArrayFloat(
    #     "0.000000000000000000000000000000000000000000000000001")

    # print(find_root_newton(new_x_func, ArrayFloat(0.5), ten_power_fifty)[0])

    # for a convergent case, we expect the difference between one iteration to the next to decrease over iterations.
    # if the magnitude of the difference between iterations has reached the level of 10^50, then we would not expect
    # following iterations to have enough of an impact on the estimation to deviate it from that value.



    #for the last question, reimplement in terms of ordered powers of 10 - but ensuring that null elements are discarded.

    # test_values = ["1.23016700", "5", "100.000", "-0.0001", "0", "-.", "1.00000100000000000000000000001"]

    # print(list(map(OptimisedArrayFloat, test_values)))


    a = OptimisedArrayFloat("3.1415926565897932384626")
    b = OptimisedArrayFloat("2.7182818284590452353602")

    print(a - b)