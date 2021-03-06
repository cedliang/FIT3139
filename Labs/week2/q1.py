import math
import functools


def composeBin(powers_list: list[list[int, int]]):
    if (returnstr := functools.reduce(lambda prevString, digit_pair: prevString+str(digit_pair[1])+"." if digit_pair[0] == 0
                                      else prevString+str(digit_pair[1]), powers_list, ""))[-1] == ".":
        return returnstr[:-1]
    if (appendZeros := powers_list[-1][0]) > 0:
        return returnstr+appendZeros*"0"
    if (powers_list[0][0]) < 0:
        return f"0.{returnstr}"
    return returnstr


def toBinAux(currentexp: int, currentn: int, current_powerslist: list[list[int, int]]):
    if currentexp == -16 or currentn == 0:
        return currentexp, currentn, current_powerslist
    if currentn >= 2**currentexp:
        return toBinAux(currentexp-1, currentn - 2**currentexp, current_powerslist+[[currentexp, 1]])
    return toBinAux(currentexp-1, currentn, current_powerslist+[[currentexp, 0]])


def toBin(real_n: float):
    return (composeBin(toBinAux(math.floor(math.log(real_n, 2)), real_n, [])[2]) if real_n > 0 else "-"+composeBin(toBinAux(math.floor(math.log((-1)*real_n, 2)), (-1)*real_n, [])[2])) if real_n != 0 else 0


if __name__ == "__main__":
    dec_num: float = 0

    print(toBin(dec_num))
