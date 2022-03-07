import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[1,1,1],[4,5,6],[7,8,9]])

#add
print("sum\n", a+b, "\n")
#show dimensions
print("shape\n", a.shape, "\n")
#transpose
print("transposition\n",(a+b).T, "\n")

#note the difference between list multiplication and np.ndarray multiplication
pya = [[1,2,3],[4,5,6],[7,8,9]]
print("python list *2\n", pya*2, "\n")
print("np.ndarray *2 \n", a*2, "\n")

#dot product of two vectors
print("dot product\n", np.dot(np.array([1,2,3]), np.array([4,5,6])), "\n")

#note that np.dot also calculates matrix multiplication!
print("matrix multiplication\n", np.dot(a, b), "\n")