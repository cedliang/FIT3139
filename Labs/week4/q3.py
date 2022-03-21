
import numpy as np
from q2 import gs_method, jacobi_method
import random

def generate_problem(size):
    x = np.array([random.uniform(-50, 50) for _ in range(size)])
    
    a = np.random.randint(-50, 50, (size, size))
    while 0 in np.diagonal(a):
        a = np.random.randint(-50, 50, (size, size))

    b = np.dot(a, x)
    return a, x, b

def sample_problem(size, num_samples=100):
    def sample_single():
        a, x, b = generate_problem(size)
        return not isinstance(jacobi_method(a, b), str), not isinstance(gs_method(a, b), str)

    samples = [sample_single() for _ in range(num_samples)]
    j_conv = sum([sample[0] for sample in samples])/len(samples)
    gs_conv = sum([sample[1] for sample in samples])/len(samples)
    return j_conv, gs_conv

if __name__ == "__main__":
    size_range = range(2, 6)
    num_samples = 1000

    results = []
    for i in size_range:
        j_conv, gs_conv = sample_problem(i, num_samples)
        results.append((i, j_conv, gs_conv))

    for i, j_conv, gs_conv in results:
        print(f"\nSize {i} matrices:\nj_conv: {j_conv}\ngs_conv: {gs_conv}\n")
