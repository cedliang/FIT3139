import math

def stirlings_fact(n: int):
    return math.sqrt(2*math.pi*n)*(n/math.e)**n

if __name__ == "__main__":
    for n in range(1, 16):
        abs_error = (realnfact := math.factorial(n)) - (stirlings_nfact := stirlings_fact(n))
        relative_error = abs_error / realnfact

        print(n,"\t", realnfact, stirlings_nfact, abs_error, relative_error)