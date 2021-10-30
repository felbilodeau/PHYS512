import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Well, I seem to have missed the class where we did that random walk
# problem, so I guess I'll just do part b) of this question.

# Generate the random walk
N = 1001
random_walk = np.cumsum(np.random.randn(N))

# Define the window
window = 0.5*(1 - np.cos(2*np.pi*np.arange(0,N)/N))

# Get the FT with the window
FT = np.fft.fftshift(np.fft.fft(random_walk * window))

# Calculate the power spectrum (and remove the k=0 mode)
power_spectrum = np.zeros(N - 1)
power_spectrum[:(N-1)//2] = np.abs(FT[:(N-1)//2])**2
power_spectrum[(N-1)//2:] = np.abs(FT[(N+1)//2:])**2

# Define the function we want to test
def fun(k, a, b):
    return a/k**2 + b

# Define the k values
k = np.linspace(-N/2, N/2, 1000)

# Fit the function to the power spectrum
popt, pcov = curve_fit(fun, k, power_spectrum)
print("a =", popt[0], "+/-", np.sqrt(pcov[0,0]))
print("percent error in a =", np.sqrt(pcov[0,0])/popt[0]*100, "%")  # so this hovers around 0.2% - 2%, so it's not bad

# Plot the power spectrum and save to 'power_spectrum.png'
plt.plot(k, power_spectrum, '.')
# Let's look somewhere interesting at least
plt.xlim(0,30)
plt.ylim(0, 2e6)
plt.plot(k, fun(k, *popt), 'r-')
plt.title("Power spectrum of random walk")
plt.xlabel(r'$k$')
plt.ylabel("Power")
plt.savefig('power_spectrum.png', bbox_inches='tight')

# Okay so the relationship looks okay, depending on the random walk we get.