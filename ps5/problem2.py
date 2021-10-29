import numpy as np
import matplotlib.pyplot as plt
from problem1 import gaussian

# Okay so here is the function I wrote
def correlation(f, g):
    # First we take the Fourier transform of f
    f_ft = np.fft.fft(f)

    # Then we take the conjugate of the FT of g
    g_ft_conj = np.conjugate(np.fft.fft(g))

    # We take the correlation in Fourier space
    result_ft = f_ft * g_ft_conj

    # For some reason I have to fftshift this, otherwise everything
    # is inverted. I'm not sure I understand why I have to do this but
    # it works I guess...
    return np.fft.fftshift(np.fft.ifft(result_ft))

if __name__ == '__main__':
    # Define a range and calculate the standard gaussian
    x = np.linspace(-5, 5, 1001)
    y = gaussian(x, 0, 1)

    # Calculate the correlation
    corr = np.real(correlation(y, y))
    print(corr[len(x)//2])  # exactly 100 times what the integral gives

    # For some reason, corr seems to be exactly 100 times larger than if I did
    # the integral instead. So after some testing it seems to be that the result
    # I get is pretty much exactly (len(x) - 1) / (np.max(x) - np.min(x)) larger
    # than what the direct integral gives, up to 12 digits after the decimal, 
    # so it has something to do with the range and the number of points within
    # that range. I'm not sure if that really matters though.

    # Here we plot the correlation and save it to 'gauss_corr.png'
    plt.plot(x, corr, label="correlation")
    plt.xlabel(r"$y$")
    plt.ylabel(r"correlation")
    plt.title("Correlation of standard gaussian with itself")
    plt.legend()
    plt.savefig('gauss_corr.png', bbox_inches='tight')