import numpy as np
import matplotlib.pyplot as plt
import os

# Here is how I did it:
# First, we need to find some function g(x) which has the effect of
# shifting any function upon convolution, which means that
# integral(f(y)*g(x - y) dy) = f(x - a) for some shift a
# Thus, we need g(x - y) = delta(y - (x - a)) = delta(-(x - y) + a)
# Therefore, g(x) = delta(x - a). To do the convolution easily,
# we of course use Fourier transforms, and the Fourier transform of
# a delta function is F[delta(x - a)] = exp(-ika).
# So we do this and then invert the Fourier transform to get the shifted
# array

# This function shifts func_array by shift, where shift corresponds
# to a shift in indices. This shift can in principle be fractional.
# If the shift is fractional though, you will get some serious oscillations
# at function discontinuities, so the advice is to leave it as an integer
def convolution_shift(func_array, shift):
    # First we need the length of the array to create the delta FT
    n = len(func_array)

    # Then we calculate the FT of the array
    func_ft = np.fft.fft(func_array, n)

    # Extract the corresponding frequencies in order
    freqs = np.fft.fftfreq(n)

    # Calculate the delta FT as exp(-2*pi*i*freqs*shift)
    delta_ft = np.exp(-2*np.pi*1j*freqs*shift)

    # Do the convolution in Fourier space
    result_ft = func_ft * delta_ft

    # Return the inverse Fourier transform
    return np.fft.ifft(result_ft)

# Just defining a gaussian function here
def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-1/2 * ((x - mu) / sigma)**2)

if __name__ == '__main__':
    # Set up relative path handling
    path = os.path.realpath(os.path.dirname(__file__))
    os.chdir(path)

    # Define an x range and calculate the standard gaussian at 0
    x = np.linspace(-5, 5, 1001)
    y = gaussian(x, 0, 1)

    # Calculate the function shifted by half the array length
    # Taking the real part because the imaginary part will 
    # be like 1e-17
    y_mod = np.real(convolution_shift(y, len(x)/2))

    # Plot and save this graph
    plt.plot(x, y, label="original gaussian")
    plt.plot(x, y_mod, label="shifted gaussian")
    plt.title("Gaussian shifted by convolution")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.savefig("gauss_shift.png", bbox_inches='tight')