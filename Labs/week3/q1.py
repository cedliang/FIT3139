import numpy as np

# I assume that the first question is exercising our ability to write a linear system in matrix representation, and as such we would be permitted to use inbuilt
# functions to solve this problem.

if __name__ == "__main__":
    a = np.array([[0, -6, 5], [0, 2, 7], [-4, 3, -7]])
    b = np.array([50, -30, 50])

    print("Solutions:\n", np.linalg.solve(a, b))
    print("Coefficient matrix transposed:\n", a.T)
    print("Coefficient matrix inverted:\n", np.linalg.inv(a))
