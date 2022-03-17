import sys


if __name__ == "__main__":
    e = 1.0
    while e + 1 > 1:
        if (e/2) + 1 > 1:
            e = e/2
        else:
            break

    print(e, e == sys.float_info.epsilon)
