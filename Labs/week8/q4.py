import numpy as np
import operator
import random
from multiprocessing import Pool
from functools import partial


def singleWorker(func, xa, xb, ya, yb, trials):
    underCurve = 0
    for _ in range(trials):
        x = random.random()*(xb-xa) + xa
        y = random.random()*(yb-ya) + ya

        fx = func(x)
        if fx < 0:
            raise Exception("Function is negative in domain.")

        if y < fx:
            underCurve += 1
    return underCurve

def f1(x):
    return np.sin(np.pi*x)

# def f1(x):
#     return 1/(x+1)

if __name__ == '__main__':
    n = 10000000

    xa = 0
    xb = 1
    ya = 0
    yb = 1

    per_core = [(f1, xa, xb, ya, yb, n)]*8

    with Pool() as pool:
        result = pool.starmap(singleWorker, per_core)

    prob_undercurve = sum(result)/(8*n)
    area_square = (xb-xa)*(yb-ya)

    area_undercurve = area_square*prob_undercurve

    # part 1
    pi = 2/area_undercurve
    print(pi)

    # # part 2
    # log2 = area_undercurve
    # print(log2)