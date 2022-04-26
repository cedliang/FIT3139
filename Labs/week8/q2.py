import matplotlib.pyplot as plt
import operator
import random
from multiprocessing import Pool
import numpy as np
from functools import partial


def singleWorker(n, r):
    crosses_line = 0
    for _ in range(n):
        u = random.random()
        theta = random.random() * (np.pi/2)

        if u + r*np.cos(theta) >= 1:
            crosses_line += 1

    print(f"{r} done")
    return crosses_line


if __name__ == '__main__':
    n = 2000000

    rs = list(np.linspace(0, 3, 100))

    n_rs = list(zip([n]*len(rs), rs))
    with Pool() as pool:
        result = pool.starmap(singleWorker, n_rs)

    probs = list(map(partial(operator.mul, (1/n)), result))


    plt.plot(rs, probs)
    plt.grid()
    plt.xlabel("r")
    plt.ylabel("Probability of intersection")
    plt.title("Monte carlo simulation of Buffon's needle")
    plt.show()