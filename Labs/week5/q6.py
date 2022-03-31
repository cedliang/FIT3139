import itertools


def model(r, K, N0, funct):
    k = 0
    Nk = N0
    while True:
        yield N0 if k == 0 else (Nk := funct(r, K)(Nk))
        k += 1


if __name__ == "__main__":
    funct = lambda r, K: lambda Nk: Nk+r*Nk*(1-Nk/K)

    num_gens = 100
    K = 1000
    rs = [0.5, 2.3, 3]


    Ns = [99, 100, 101]
    pops = {r:{n:list(itertools.islice(model(r, K, n, funct), num_gens))[-1] for n in Ns} for r in rs}

    print(pops)



    # case 0.5 and case 2.3 do not exhibit chaotic behaviour - case 0.5 converges to K, whilst case 2.3 oscillates without chaos
    # case 3 however exhibits chaotic behaviour, showing vastly behaviours for small changes in initial conditions