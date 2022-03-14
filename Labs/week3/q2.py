import numpy as np

def det_two_by_two(matrix: np.ndarray):
    return matrix[0, 0]*matrix[1, 1] - matrix[1, 0]*matrix[0, 1]

def det_three_by_three(matrix: np.ndarray):
    return matrix[0, 0]*det_two_by_two(matrix[1:, 1:]) - matrix[0, 1]*det_two_by_two(np.array([[matrix[1, 0], matrix[1, 2]],[matrix[2, 0], matrix[2, 2]]])) + matrix[0, 2]*det_two_by_two(matrix[1:, :2])

if __name__ == "__main__":
    a = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    b = np.array([0.1, 0.3, 0.5])

    print("Solutions:\n", np.linalg.solve(a, b))
    print("Det(a):\n", np.linalg.det(a))

    #for the second part, instead of doing a computation by hand on paper, I'll do one better - I'll use the property that for an nxn matrix A and a scalar b, 
    #the property 
    # det(bA) = (b^n)*det(A)
    #holds.

    #Suppose I then scale this matrix up by a scalar factor of 10 - then we expect
    # det(10A) = 10^3 det(A)
    # det(A) = det(10A) * (1/1000)

    #Then (making sure to use integers instead of floats)

    a_prime = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Hand calculated determinant:\n", det_three_by_three(a_prime))

    #We note the determinant of the matrix is 0, which means the determinant of the original matrix is also zero! It's likely the discrepancy is caused by
    #rounding error, given that the process of calculating the determinant involves several steps of successive multiplication between floats.
    #While small, when the determinant is expected to be exactly zero, this rounding error is enough to result in a non-zero determinant.