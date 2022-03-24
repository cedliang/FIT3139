import matplotlib.pyplot as plt

# s_k+1 = s_k-1 + 0.5 s_k


def gen_terms(num_terms: int):
    def f(k_list: list[float]):
        return k_list if len(k_list) == num_terms + 1 else f(k_list + [k_list[-2] + 0.5*k_list[-1]])
    return [0, 1] if num_terms < 2 else f([0, 1])


if __name__ == "__main__":
    max_k = 100

    x = list(range(max_k + 1)) if max_k > 1 else [0, 1]
    y = gen_terms(max_k)

    print(y)

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(x, y)
    plt.show()

    # plotting this with log y axes shows that this system experiences exponential growth
