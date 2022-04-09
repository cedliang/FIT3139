import itertools
import numpy as np
import matplotlib.pyplot as plt


def model(m, c, x0, funct):
    yield x0
    x = x0
    while True:
        yield (x := funct(m, c)(x))


def gen_cobweb_plot(function, pop, show=True):
    def __gen_doubled_list(pop_inner):
        return [pop_inner[0]]*2 if len(pop_inner) == 1 else [pop_inner[0]]*2 + __gen_doubled_list(pop_inner[1:])

    pop_p, pop_pplus = (doubled_list := __gen_doubled_list(pop))[
        :-2], doubled_list[1:-1]
    domain_linspace = np.linspace(min(pop), max(pop), 1000)

    plt.plot(domain_linspace, list(map(function, domain_linspace)))
    plt.plot(pop[0], pop[0], 'ro')
    plt.plot(pop_p, pop_pplus)
    plt.plot(domain_linspace, domain_linspace)
    if show:
        plt.show()


if __name__ == "__main__":
    def funct(m, c): return lambda x: x*m + c

    num_gens = 50
    m = 2
    c = 3

    x0 = 1

    xs = list(itertools.islice(model(m, c, x0, funct), num_gens))

    gen_cobweb_plot(funct(m, c), xs)
