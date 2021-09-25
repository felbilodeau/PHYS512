import numpy as np
import os

# Making sure the current directory is the directory
# of this .py file
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

def mylog2(x):
    # load the chebyshev coefficient from the text file
    coeffs = np.loadtxt("chebyshev.txt")

    # When we take the log2 of a floating point number, we can do
    # log2(a*2^b) = log2(a) + log2(2^b) = log2(a) + b
    # where a is the mantissa and b is the exponent

    # Here we separate the mantissa and the exponent
    x_mantissa, x_exponent = np.frexp(x)
    e_mantissa, e_exponent = np.frexp(np.e)

    answer = np.polynomial.chebyshev.chebval(4*(x_mantissa - 0.75), coeffs) + x_exponent
    answer /= np.polynomial.chebyshev.chebval(4*(e_mantissa - 0.75), coeffs) + e_exponent

    return answer

if __name__ == "__main__":

    x = 6
    test = mylog2(x)
    true = np.log(x)
    err = np.abs(test - true)
    print(test)
    print(true)
    print(err)