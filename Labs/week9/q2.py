import random
from matplotlib import pyplot as plt


def sample_discrete(values, probabilities):
    r = random.random()
    cum_prob = 0

    for i in range(len(values)):
        cum_prob += probabilities[i]
        if r < cum_prob:
            return values[i]


if __name__ == "__main__":
    values = ["a", 13, "bcd", 0.5]
    probabilities = [0.25, 0.5, 0.10, 0.15]

    r = sample_discrete(values, probabilities)
    print(r)
