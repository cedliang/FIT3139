
import random
import multiprocessing as mp
import itertools
import numpy as np
import operator
import gc


ROUTING_TABLE = {
    0: (1, 2),
    1: (3, 4),
    2: (4, 5),
    3: (6, 7),
    4: (7, 8),
    5: (8, 9),
    6: (15, 10),
    7: (10, 13),
    8: (13, 11),
    9: (11, 16),
    10: (15, 12),
    11: (14, 16),
    12: (15, 13),
    13: (12, 14),
    14: (13, 16),
    15: (15, 15),
    16: (16, 16),
}

INITIAL_STATE = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def build_tmat_tennis(p):
    """Built transition matrix with p
    """
    tmat = []
    for state in range(17):
        row = [0]*17
        if state in range(15):
            row[ROUTING_TABLE[state][0]] += p
            row[ROUTING_TABLE[state][1]] += 1-p
        else:
            row[state] = 1
        tmat.append(row)
    return np.array(tmat)


def repeat_dot_prod(tmat, n):
    """
    Repeats dot product - exponentiate
    """
    pcopy = np.copy(tmat)
    for _ in range(n):
        tmat = np.dot(pcopy, tmat)
    return tmat


def who_wins_computational(p):
    """
    Computes who wins after 1000 points.
    """
    return tuple(np.dot(INITIAL_STATE, repeat_dot_prod(build_tmat_tennis(p), 1000)))[-2:]


def get_r(tmat):
    return tmat[:15, 15:]


def get_q(tmat):
    return tmat[:15, :15]


def get_n(tmat):
    return np.linalg.inv(np.identity(len(tmat)-2)-get_q(tmat))


def get_absorp_t(tmat):
    return np.dot(get_n(tmat), np.array((len(tmat)-2)*[1]))


def get_b(tmat):
    return np.dot(get_n(tmat), get_r(tmat))


def simulate_game(p):
    state, num_points = 0, 0
    while state not in {15, 16}:
        state = ROUTING_TABLE[state][0] if random.random(
        ) < p else ROUTING_TABLE[state][1]
        num_points += 1
    return state == 15, num_points


if __name__ == "__main__":
    ps = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    print("\nmatrix exponentiation A win prob")
    print(list(map(operator.itemgetter(0), map(who_wins_computational, ps))))

    bstore = []
    tstore = []
    for p in ps:
        tmat = build_tmat_tennis(p)
        q = get_q(tmat)
        r = get_r(tmat)
        n = get_n(tmat)
        t = get_absorp_t(tmat)
        b = get_b(tmat)

        bstore.append(b[0][0])
        tstore.append(t[0])

    print("\nfundamental matrix A win prob")
    print(bstore)

    print("\nfundamental matrix mean game length")
    print(tstore)

    # monte carlo approach
    num_sims = 1000000
    ps_rep = [itertools.islice(itertools.repeat(p), num_sims) for p in ps]

    a_win_probs = []
    mean_game_lengths = []

    for p_rep in ps_rep:
        with mp.Pool() as pool:
            rs = pool.map(simulate_game, p_rep)

        cum_sums = [0, 0]
        for elem in rs:
            if elem[0]:
                cum_sums[0] += 1
            cum_sums[1] += elem[1]

        a_win_probs.append(cum_sums[0]/num_sims)
        mean_game_lengths.append(cum_sums[1]/num_sims)

        del rs
        gc.collect()

    print("\nmonte carlo A win prob")
    print(a_win_probs)

    print("\nmonte carlo mean game length")
    print(mean_game_lengths)
