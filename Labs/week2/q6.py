import sys


if __name__ == "__main__":
    e = 1.0
    while e > 0:
        if (e/2) > 0:
            e = e/2
        else:
            break

    print(e, e == sys.float_info.min*sys.float_info.epsilon)
