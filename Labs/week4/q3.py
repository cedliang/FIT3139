import numpy as np
from q2 import gs_method, jacobi_method
import random
import multiprocessing
import itertools


def generate_problem(size):
    x = np.array([random.uniform(-50, 50) for _ in range(size)])

    a = np.random.randint(-50, 50, (size, size))
    while 0 in np.diagonal(a):
        a = np.random.randint(-50, 50, (size, size))

    b = np.dot(a, x)
    return a, x, b


def check_convergence(prob_tuple):
    method = jacobi_method if prob_tuple[1] == "j" else gs_method
    return not isinstance(method(prob_tuple[2], prob_tuple[4]), str)


def generate_problems(sizes: range, num_samples: int):
    problems = []

    for size, _ in itertools.product(sizes, range(num_samples)):
        a, x, b = generate_problem(size)
        problems.extend(((size, "j", a, x, b), (size, "g", a, x, b)))
    return problems


if __name__ == "__main__":

    size_range = range(2, 8)
    num_samples = 10000

    problems = generate_problems(size_range, num_samples)

    with multiprocessing.Pool() as pool:
        results = pool.map(check_convergence, problems)

    zip_res = list(zip(problems, results))

    counts = {size: [0, 0] for size in size_range}

    for elem in zip_res:
        counts[elem[0][0]][0 if elem[0][1] == "j" else 1] += int(elem[1])

    for count in counts.values():
        count[0], count[1] = count[0]/num_samples, count[1]/num_samples

    print(counts)

    # running with num samples 1000000 yielded (took about 30 minutes)
    # {2: [0.478348, 0.493903], 3: [0.110999, 0.18196], 4: [0.015594, 0.044308], 5: [0.001086, 0.006538], 6: [4.3e-05, 0.00062], 7: [1e-06, 2.6e-05]}
