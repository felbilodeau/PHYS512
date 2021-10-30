import numpy as np
import matplotlib.pyplot as plt
import os

# IMPORTANT: See problem5.pdf to know where this is coming from

# Set up relative path handling
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Part c)

# Calculating the non-integer sine
m = np.arange(0, 501, 1)
q = 1e-3
y = np.sin(2*np.pi*q*m)

# Taking the DFT with numpy
fft = np.fft.fftshift(np.fft.fft(y))

# Taking the DFT with my analytical solution
N = len(m)
k = np.arange(0, N, 1)
fft_analytical = (1-np.exp(-2*np.pi*1j*(k-N*q)))/(1-np.exp(-2*np.pi*1j*(k-N*q)/N))
fft_analytical -= (1-np.exp(-2*np.pi*1j*(k+N*q)))/(1-np.exp(-2*np.pi*1j*(k+N*q)/N))
fft_analytical /= 2j
fft_analytical = np.fft.fftshift(fft_analytical)

# Printing the max difference
print(np.max(np.abs(fft - fft_analytical))) # 2.2596473161880468e-11 <- within machine error

# Plotting and saving to 'sine_dft.png'
plt.plot(k, np.abs(fft), label='np.fft.fft')
plt.plot(k, np.abs(fft_analytical), label='my solution')
plt.xlabel(r'$k$')
plt.ylabel("DFT")
plt.title("Analytical vs NumPy DFT")
plt.legend()
plt.savefig("sine_dft.png", bbox_inches='tight')
plt.clf()

# Part d)

# We will now multiply by a window
window = 0.5*(1 - np.cos(2*np.pi*m/N))
y_window = window*y
window_fft = np.fft.fftshift(np.fft.fft(y_window))

# Plotting and saving to 'sine_dft_window.png'
plt.plot(k, np.abs(fft), label = 'no window')
plt.plot(k, np.abs(window_fft), label = 'window')
plt.title("Window vs No Window DFT")
plt.xlabel(r"$k$")
plt.ylabel('DFT')
plt.legend()
plt.savefig('sine_dft_window.png', bbox_inches='tight')
plt.clf()

# Part e)

# Now let's take the Fourier transform of the window
window_fft = np.fft.fft(window)
window_theoretical = np.zeros(len(window_fft))
window_theoretical[0] = N/2
window_theoretical[1] = -N/4
window_theoretical[-1] = -N/4

# Print the max difference
print(np.max(np.abs(window_theoretical - np.real(window_fft)))) # 2.842170943040401e-14

# So the max error is like 3e-14, so we can say that this is correct