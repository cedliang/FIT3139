import matplotlib.pyplot as plt
import numpy as np
import itertools


def model(a, K, N0, funct):
    k = 0
    Nk = N0
    while True:
        if k == 0:
            yield N0
        else:
            Nk = funct(a, K)(Nk)
            yield Nk
        k += 1


def gen_cobweb_plot(function, pop, show=True):
    pop_p = []
    pop_pplus = []
    for i in range(1, len(pop)):
        pop_p += [pop[i-1], pop[i-1]]
        pop_pplus += [pop[i-1], pop[i]]  

    domain_linspace = np.linspace(min(pop), max(pop), 1000)

    plt.plot(domain_linspace, list(map(function, domain_linspace)))
    plt.plot(pop[0], pop[0], 'ro')
    plt.plot(pop_p, pop_pplus)
    plt.plot(domain_linspace, domain_linspace)
    if show:
        plt.show()


if __name__ == "__main__":
    funct = lambda a, K: lambda Nk: Nk*np.e**(a*(1- Nk/K))

    num_gens = 200
    K = 1000
    N0 = 100
    a_vals = [1, 1.98, 2, 2.55]
    pops = {a:list(itertools.islice(model(a, K, N0, funct), num_gens)) for a in a_vals}

    for a, populations in pops.items():
        plt.plot(list(range(num_gens)), populations, label=f"a = {a}")

    plt.legend()
    plt.show()

    #no oscillation, convergent oscillation, oscillation between two values, oscillation between four values


    # for a, populations in pops.items():
    #     gen_cobweb_plot(funct(a, K), populations, show=False)
    # plt.show()
