import numpy as np
import matplotlib.pyplot as plt
import os
from problem1 import convolution_shift, gaussian
from problem2 import correlation

# Set up relative path handling
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# I'm going to assume here that I'm supposed to correlate a shifted
# Gaussian with a non-shifted Gaussian, because otherwise, we will just
# get the same result as we did in problem 2. Or maybe that's the point,
# to show that the correlation doesn't depend on the shift, so I'll do
# both I guess.

# We'll start with correlating with the non-shifted Gaussian

# Set up a range of x and calculate the standard Gaussian for that range
x = np.linspace(-5, 5, 1001)
y = gaussian(x, 0, 1)

# Shift the Gaussian by some arbitrary number of entries
y_shifted = convolution_shift(y, 800)

# Let's also calculate the position of the peak of the shifted Gaussian
index = np.where(y_shifted == np.max(y_shifted))[0][0]
x_shift = x[index]

# Calculate the correlation
corr = np.real(correlation(y_shifted, y))

# Plot and save to 'gauss_shift_corr_1.png'
plt.plot(x, corr, label='correlation')
plt.ylim(plt.ylim())
plt.plot([x_shift, x_shift], plt.ylim(), label='Shifted Gaussian peak position')
plt.title("Correlation between shifted Gaussian and\nnon-shifted Gaussian")
plt.xlabel(r'$y$')
plt.ylabel(r'correlation')
plt.legend()
plt.savefig("gauss_shift_corr_1.png", bbox_inches='tight')
plt.clf()

# As we can see from the plot, the peak position of the correlation function
# corresponds to the peak position of the shifted gaussian, which makes sense,
# since the correlation essentially gives us where the functions are most similar.

# Now let's do the correlation of the shifted gaussian with itself
corr = np.real(correlation(y_shifted, y_shifted))

# Plot and save to 'gauss_shift_corr_2.png'
plt.plot(x, corr, label='correlation')
plt.ylim(plt.ylim())
plt.title("Correlation of shifted Gaussian with itself")
plt.xlabel(r'$y$')
plt.ylabel(r'correlation')
plt.legend()
plt.savefig("gauss_shift_corr_2.png", bbox_inches='tight')

# Of course we see that the correlation function is now peaked at y = 0, because the
# functions are the same, therefore they are most similar when there is no shift, hence
# the peak at y = 0.