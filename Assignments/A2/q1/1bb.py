import random
import itertools
import numpy as np
from functools import partial
import operator
import multiprocessing as mp


def run_gillespie_survived(rbirth, rdeath, fbirth, fdeath, alpha, beta, t_max, z0, fhunt):
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

    while t_i < t_max:
        dzidx, dt = calc_times_par(z_i)
        t_i += dt
        z_i = np.copy(z_i) + events[dzidx]

        if 0 in z_i:
            return False

    return True


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


def gillespie_worker(rbirth, rdeath, fbirth, fdeath, alpha, beta, t_max, z0, fhunt, n):
    return sum(bool(run_gillespie_survived(rbirth, rdeath, fbirth, fdeath, alpha, beta, t_max, z0, fhunt)) for _ in range(n))


if __name__ == "__main__":
    rbirth = 0.8
    rdeath = 0.1
    fbirth = 0.1
    fdeath = 0.6
    alpha = 0.04
    beta = 0.01
    init_pops = np.array([20, 10])

    par_gillespie_worker = partial(gillespie_worker, rbirth, rdeath,
                                   fbirth, fdeath, alpha, beta, 10, init_pops)
    cores = 8
    numSamples = 10000
    samples_per_core = numSamples//cores
    ns = cores*[samples_per_core]
    probs = []

    fhunts = [0.1, 0.2, 0.3, 0.4, 0.5]

    for fhunt in fhunts:
        w = partial(par_gillespie_worker, fhunt)
        with mp.Pool() as pool:
            rs = pool.map(w, ns)

        probs.append(sum(rs))

    extinction_probs = [1-(prob/numSamples) for prob in probs]

    print(fhunts)
    print(extinction_probs)
