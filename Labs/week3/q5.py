from q3 import gaussian_elimination
import numpy as np

# 50 + c_3 = 6 c_1
# 3 c_1 = c_2
# 160 = 9 c_3
# 8 c_3 + c_2 + 2 c_5 = 11 c_4
# 3 c_1 = 4 c_5

# [6 0 -1 0 0] = [50]
# [3 -1 0 0 0] = [0]
# [0 0 9 0 0] = [160]
# [0 1 8 -11 2] = [0]
# [3 0 0 0 -4] = [0]

if __name__ == "__main__":
    print(gaussian_elimination(np.array([[6, 0, -1, 0, 0], [3, -1, 0, 0, 0], [0, 0, 9, 0, 0], [0, 1, 8, -11, 2], [3, 0, 0, 0, -4]]), np.array([50, 0, 160, 0, 0])))

