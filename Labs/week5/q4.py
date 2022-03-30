import matplotlib.pyplot as plt
import numpy as np
import itertools

def model(a, K, N0):
    k = 0
    Nk = N0
    while True:
        if k == 0:
            yield N0
        else:
            Nk = Nk*np.e**(a*(1- Nk/K))
            yield Nk
        k += 1

if __name__ == "__main__":
    num_gens = 200
    K = 1000
    N0 = 100
    a_vals = [1, 1.98, 2, 2.55]
    pops = {a:list(itertools.islice(model(a, K, N0), num_gens)) for a in a_vals}

    for a, populations in pops.items():
        plt.plot(list(range(num_gens)), populations, label=f"a = {a}")

    plt.legend()
    plt.show()

