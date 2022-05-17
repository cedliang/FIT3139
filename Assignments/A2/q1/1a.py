import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import operator


def rk2(b, t0, t_max, t_step, z0, f):
    # dummy function for systems that aren't directly time dependent
    def t_dependent_f(t, zvec):
        return f(zvec)

    a = 1-b
    alphabeta = (1/2)/b

    t_i = t0
    z_i = z0
    t = []
    z = []
    while t_i < t_max:
        t.append(t_i)
        z.append(z_i)
        k1 = t_dependent_f(t, z_i)
        k2 = t_dependent_f(t_i+alphabeta*t_step, z_i+alphabeta*k1*t_step)
        t_i = t_i+t_step
        z_i = z_i+t_step*(a*k1+b*k2)
    return t, z


def system_equations(rbirth, rdeath, fbirth, fdeath, alpha, beta, fhunt, z):
    x, y = z[0], z[1]
    return np.array([
        x*(rbirth-alpha*y-fhunt-rdeath),
        y*(fbirth+beta*x-fhunt/2-fdeath)
    ])


if __name__ == "__main__":
    rbirth = 0.8
    rdeath = 0.1
    fbirth = 0.1
    fdeath = 0.6
    alpha = 0.04
    beta = 0.01

    rk2_func = partial(rk2, 0.5, 0, 100, 0.1)
    init_pops = np.array([20, 10])

    plotCounter = itertools.count()

    fhunts = [0.20, 0.4, 0.55, 0.65, 0.7]
    for fhunt in fhunts:
        system = partial(system_equations, rbirth, rdeath, fbirth, fdeath, alpha, beta, fhunt)
        ts, zs = rk2_func(init_pops, system)
        xs = list(map(operator.itemgetter(0), zs))
        ys = list(map(operator.itemgetter(1), zs))

        # timeplot
        plt.figure(next(plotCounter))
        plt.plot(ts, xs, label="Rabbit population")
        plt.plot(ts, ys, label="Fox population")
        plt.legend(loc="best")
        plt.title(f"Rabbit-Fox population time-plot: hunt rate f={fhunt}")
        plt.ylabel("Animal count")
        plt.xlabel("Time")

        # phaseplot
        plt.figure(next(plotCounter))
        plt.plot(xs, ys)
        plt.title(f"Rabbit-fox phase diagram: hunt rate f={fhunt}")
        plt.xlabel("Rabbit count")
        plt.ylabel("Fox count")

    plt.show()
