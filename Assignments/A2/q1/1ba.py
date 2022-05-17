import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import operator


# events by indices
# 0 Rabbit is born
# 1 Rabbit dies naturally
# 2 Rabbit is hunted
# 3 Rabbit is eaten
# 4 Fox is born
# 5 Fox dies naturally
# 6 Fox survives by eating rabbit
# 7 Fox is hunted


def run_gillespie(rbirth, rdeath, fbirth, fdeath, alpha, beta, t_max, z0, fhunt):
    events = [
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([-1, 0]),
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([0, 1]),
        np.array([0, -1]),
    ]
    t_i = 0
    z_i = z0
    calc_times_par = partial(calc_times, rbirth, rdeath, fbirth,
                             fdeath, alpha, beta, fhunt)

    t = []
    z = []
    while t_i < t_max:
        t.append(t_i)
        z.append(z_i)
        dzidx, dt = calc_times_par(z_i)
        t_i += dt
        z_i = np.copy(z_i) + events[dzidx]

    return t, z


def calc_times(rbirth, rdeath, fbirth, fdeath, alpha, beta, fhunt, z):
    x = z[0]
    y = z[1]

    probs = [
        rbirth*x,
        rdeath*x,
        fhunt*x,
        alpha*x*y,
        fbirth*y,
        fdeath*y,
        beta*x*y,
        y*fhunt/2
    ]

    time_untils = [-1 * np.log(random.random()) / prob for prob in probs]
    return min(enumerate(time_untils), key=operator.itemgetter(1))


if __name__ == "__main__":
    rbirth = 0.8
    rdeath = 0.1
    fbirth = 0.1
    fdeath = 0.6
    alpha = 0.04
    beta = 0.01
    init_pops = np.array([20, 10])

    plotCounter = itertools.count()

    fhunts = [0.1, 0.5]

    for fhunt in fhunts:
        rgillespie = partial(run_gillespie, rbirth, rdeath,
                             fbirth, fdeath, alpha, beta, 10, init_pops)

        ts, zs = rgillespie(fhunt)
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
