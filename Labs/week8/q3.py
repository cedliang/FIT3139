import operator
import random
from multiprocessing import Pool
from functools import partial


def singleWorker(trials, num_players):
    wins = [0]*num_players

    for _ in range(trials):
        num_rolls = 0
        while random.random() > (1/6):
            num_rolls += 1
        wins[num_rolls % num_players] += 1

    return wins


if __name__ == '__main__':
    n = 10000000
    num_players = 3

    per_core = [(n, num_players)] * 8

    with Pool() as pool:
        result = pool.starmap(singleWorker, per_core)

    results_aggregated = list(map(sum, list(map(list, zip(*result)))))
    probs_aggregated = list(
        map(partial(operator.mul, (1/sum(results_aggregated))), results_aggregated))

    print(probs_aggregated)
