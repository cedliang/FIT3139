import numpy as np
from itertools import repeat
import itertools
import matplotlib.pyplot as plt


def euler(x0, y0, x_max, h, f):
    x_i = x0
    y_i = y0
    x = []
    y = []
    while x_i < x_max:
        x.append(x_i)
        y.append(y_i)
        y_i = y_i + h*f(x_i, y_i)
        x_i = x_i + h
    return x, y


def rk2(b):
    def rk2_method(x0, y0, x_max, h, f):
        a = 1-b
        alphabeta = (1/2)/b

        x_i = x0
        y_i = y0
        x = []
        y = []
        while x_i < x_max:
            x.append(x_i)
            y.append(y_i)
            k1 = f(x_i, y_i)
            k2 = f(x_i+alphabeta*h, y_i+alphabeta*k1*h)
            x_i = x_i+h
            y_i = y_i+h*(a*k1+b*k2)
        return x, y
    return rk2_method




if __name__ == "__main__":
    hs = [0.1, 0.05, 0.025, 0.0125]
    bs = [0.1, 0.5, 0.66, 1]
    
    # 2a
    x0_2a = 0
    y0_2a = 4
    x_max = 1.7

    func_2a = lambda x, y: x*np.sqrt(1-((y-4)**2))
    exact_2a = lambda x: np.sin(x**2/2) + 4

    args_2a = list(zip(repeat(x0_2a), repeat(y0_2a), repeat(x_max), hs, repeat(func_2a)))

    ys = list(map(lambda b: list(itertools.starmap(rk2(b), args_2a)), bs))


    plot_h = 0.025

    for bsidx, results in enumerate(ys):
        b = bs[bsidx]
        for rsidx, result in enumerate(results):
            h = hs[rsidx]

            # vary plot condition here
            if h == plot_h:
                plt.plot(result[0], result[1], label=f"b={b}, h={h}")

    rs_euler_2a = euler(x0_2a, y0_2a, x_max, plot_h, func_2a)
    plt.plot(rs_euler_2a[0], rs_euler_2a[1], label=f'euler, h={h}')

    xs = np.linspace(x0_2a, x_max, 1000)
    ys = list(map(exact_2a, list(xs)))
    plt.plot(xs, ys, label="Exact solution", color='black')

    plt.legend(loc='best')
    plt.grid()
    plt.show()



    # # 2b
    # x0_2b = 0
    # y0_2b = 1
    # x_max_2b = 0.5

    # func_2b = lambda x, y: y**3
    # exact_2b = lambda x: 1/np.sqrt(1-2*x)

    # args_2b = list(zip(repeat(x0_2b), repeat(y0_2b), repeat(x_max_2b), hs, repeat(func_2b)))

    # ys = list(map(lambda b: list(itertools.starmap(rk2(b), args_2b)), bs))


    # plot_h = 0.0125

    # for bsidx, results in enumerate(ys):
    #     b = bs[bsidx]
    #     for rsidx, result in enumerate(results):
    #         h = hs[rsidx]

    #         # vary plot condition here
    #         if h == plot_h:
    #             plt.plot(result[0], result[1], label=f"b={b}, h={h}")

    # rs_euler_2b = euler(x0_2b, y0_2b, x_max_2b, plot_h, func_2b)
    # plt.plot(rs_euler_2b[0], rs_euler_2b[1], label=f'euler, h={h}')

    # xs = np.linspace(x0_2b, x_max_2b, 1000)
    # ys = list(map(exact_2b, list(xs)))
    # plt.plot(xs, ys, label="Exact solution", color='black')

    # plt.legend(loc='best')
    # plt.xlim(0, 0.49)
    # plt.ylim(0, 10)
    # plt.grid()
    # plt.show()
