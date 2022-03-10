import math

from nbformat import current_nbformat

#n \in 0 <= n <= inf
def e_x_nthterm(x: int):
    def f(n: int):
        return (x**n)/math.factorial(n)
    return f



def fixed_num_terms(x: int, num_terms: int):
    current_approx = 0

    #note that range() initiates the n value for the series at 0.
    #A series that begins at 1 will require a different range term
    for n in range(num_terms):
        last_approx = current_approx
        current_approx += e_x_nthterm(x)(n)
        approx_error = (current_approx - last_approx)/current_approx

        print(approx_error)
    print("Final approximation", current_approx)



#iterates until the error for that iteration is lower than 10^tolerance_exponent
#tolerance_exponent should be a negative integer for best results, ie -12
def error_tolerance(x: int, tolerance_exponent: int):
    current_approx = 0

    n = 0
    firstLoop = True
    
    while firstLoop or approx_error > 10**tolerance_exponent:
        last_approx = current_approx
        current_approx += e_x_nthterm(x)(n)
        approx_error = (current_approx - last_approx)/current_approx

        if firstLoop:
            firstLoop = False
        
        n += 1

    print("Numterms used:", n)
    print("Final approximation", current_approx)


if __name__ == "__main__":
    #fixed_num_terms(10,40)

    error_tolerance(10, -16)


