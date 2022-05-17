import numpy as np


def snakesLaddersTMat(numStates, snakes_ladders):
    """Generate a transition matrix for the given snakes and ladders game.

    Args:
        numStates (_type_): Number of states in the game of snakes and ladders
        snakes_ladders (_type_): List of tuples of snakes and ladders, with first elem in the tuple being the origin and the second elem being the destination
    """

    tMat = []
    for i in range(numStates):
        row = [0]*numStates
        if i < numStates - 6:
            for destSquare in range(i+1, i+7):
                row[destSquare] = 1/6
        else:
            numAllocated = 0
            for destSquare in range(i+1, numStates-1):
                row[destSquare] = 1/6
                numAllocated += 1
            row[numStates-1] = (6-numAllocated)/6

        tMat.append(row)

    for orig, dest in snakes_ladders:
        row = [0]*numStates
        row[dest] = 1
        tMat[orig] = row

    return np.array(tMat)


def canonicalForm(tMat):


if __name__ == "__main__":
    numStates = 63

    sl = snakesLaddersTMat(numStates, [
        (2, 16),
        (24, 10),
        (19, 32),
        (31, 15),
        (27, 41),
        (47, 33),
        (39, 45),
        (57, 44),
        (46, 60)
    ])

    print(sl)
