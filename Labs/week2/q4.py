import math
import matplotlib.pyplot as plt
import numpy as np

def f(x: int):
    return math.tan(x)

def fprime(x: int ):
    return 1 + (math.tan(x))**2

def condition(x: int):
    return abs((x*fprime(x))/f(x))



if __name__ == "__main__":
    xs = np.linspace(-10, 10, 500)
    ys = np.array(list(map(condition, xs)))

    fig = plt.figure()

    plt.axes()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-10,10])
    plt.ylim([0,1600])
    plt.title("Conditioning of y = tan(x)")
    plt.plot(xs, ys)
    plt.show()