

def convert_str(dec_str: str) -> list:
    if dec_str[0] not in ["-", "+"]:
        dec_str = f"+{dec_str}"
    if "." not in dec_str:
        dec_str += "."

    return [(int(char) if char.isdigit() else char) for char in dec_str]


def convert_str_alt(dec_str: str) -> list:
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

    #remove leading zeros in elems before dec, trailing zeros in elems after
    filtered_int_elems = list(filter(lambda dig: dig != 0, return_array[1]))
    stripped_int_elems = return_array[1][return_array[1].index(filtered_int_elems[0]):] if filtered_int_elems else []

    reversed_fracs = list(reversed(return_array[2]))
    filtered_frac_elems_reversed = list(filter(lambda dig: dig != 0, reversed_fracs))
    stripped_frac_elems = list(reversed(reversed_fracs[reversed_fracs.index(filtered_frac_elems_reversed[0]):])) if filtered_frac_elems_reversed else []

    return [return_array[0], stripped_int_elems, stripped_frac_elems]


if __name__ == "__main__":
    dec_string = "-0015.15000"

    print(convert_str_alt(dec_string))
