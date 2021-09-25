import numpy as np
import matplotlib.pyplot as plt
import os

# Here I'm just setting the path so the text file saves
# in the same directory as this .py file
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# So to rescale log2(x) from 0.5 to 1
# so it goes from -1 to 1, we need to calculate
# log2(x/4 + 0.75)

# First we create the x/y pairs
x = np.linspace(-1, 1, 2001)
y = np.log2(x/4 + 0.75)

# We fit the coefficients
coeffs = np.polynomial.chebyshev.chebfit(x, y, 8)
print("coefficients =", coeffs)
# I chose degree 8 because it is the first coefficient ~ 1e-7
# and the Chebyshev error is the sum of all truncated coefficients

# We can evaluate the fit and check the error real quick
y_fit = np.polynomial.chebyshev.chebval(x, coeffs)
err = np.max(np.abs(y_fit - y))
print()
print("max error =", err) # comes out to 1.29e-7, so we're good

plt.plot(x, y, '.')
plt.plot(x, y_fit)
plt.show()
plt.clf()

# Now I'm going to save the coefficients to a text file
np.savetxt("chebyshev.txt", coeffs)