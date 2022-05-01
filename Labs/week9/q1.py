import random
from matplotlib import pyplot as plt


def urv_interval(a, b):
    return a + (b-a)*random.random()


if __name__ == "__main__":
    num_rands = 1000
    a = 1
    b = 4

    rands_list = [urv_interval(a, b) for _ in range(num_rands)]

    print(rands_list)
    plt.hist(rands_list)
    plt.show()
