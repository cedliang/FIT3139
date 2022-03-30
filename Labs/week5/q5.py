import matplotlib.pyplot as plt
import numpy as np
import itertools
from q4 import gen_cobweb_plot


def model(lamb, b, N0, funct):
    k = 0
    Nk = N0
    while True:
        if k == 0:
            yield N0
        else:
            Nk = funct(lamb, b)(Nk)
            yield Nk
        k += 1


if __name__ == "__main__":
    funct = lambda lamb, b: lambda Nk: Nk*lamb/((1+Nk)**b)

    num_gens = 200
    lamb_b = {"moth 10": (1.3, 0.1, 10), "mosquito 10": (10.6, 1.9, 10), "potato beetle 10": (75.0, 3.4, 10)}

    pops = {animal: list(itertools.islice(model(*lamb_b_tup, funct), num_gens)) for animal, lamb_b_tup in lamb_b.items()}

    # for animal, populations in pops.items():
    #     plt.plot(list(range(num_gens)), populations, label=animal)

    # plt.legend()
    # plt.show()

    insect_params = lamb_b["mosquito 10"]
    insect_pop = pops["mosquito 10"]
    gen_cobweb_plot(funct(*insect_params[:-1]), insect_pop)

    # the system will always tend towards the steady state solution at the intersection between the function and x=y.
    # since the population is always positive, the term that Nk is multiplied by is always positive, and as such there is no chance of the population becoming negative or zero

    # I can't prove that is only one ever one intersection between y=x and the function in the positive domain, but if this holds true, then it makes sense that all cases converge
    # towards this steady state solution