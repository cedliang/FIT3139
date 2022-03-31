import matplotlib.pyplot as plt
import numpy as np
import itertools


def model(a, K, N0, funct):
    k = 0
    Nk = N0
    while True:
        yield N0 if k == 0 else (Nk := funct(a, K)(Nk))
        k += 1


def gen_cobweb_plot(function, pop, show=True):
    def __gen_doubled_list(pop_inner):
        return [pop_inner[0]]*2 if len(pop_inner) == 1 else [pop_inner[0]]*2 + __gen_doubled_list(pop_inner[1:])

    pop_p, pop_pplus = (doubled_list := __gen_doubled_list(pop))[:-2], doubled_list[1:-1]
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
    a_vals = [1, 1.98, 2, 2.55, 2.8]
    pops = {a: list(itertools.islice(model(a, K, N0, funct), num_gens))
            for a in a_vals}


    if plot_timewise := [1, 1.98, 2, 2.55, 2.8]:
        for a in plot_timewise:
            plt.plot(list(range(num_gens)), pops[a], label=f"a = {a}")
        plt.legend()
        plt.show()


    if plot_cobwebs := [2.55, 1.98]:
        for a in plot_cobwebs:
            gen_cobweb_plot(funct(a, K), pops[a], show=False)
        plt.show()
