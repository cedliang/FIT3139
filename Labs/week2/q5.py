
import sys


if __name__ == "__main__":
    e = 1.0
    while e > 0:
        e /= 2
    print(e, e == sys.float_info.epsilon)
