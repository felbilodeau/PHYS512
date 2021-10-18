import numpy as np
import matplotlib.pyplot as plt
import os
import uncertainties
from uncertainties import ufloat

# Set the path handling to relative
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Load the chain from the file we generated in problem3.py
text_file = np.loadtxt("planck_chain.txt", float, '#').transpose()

# Separate the file into its components
chisq = text_file[0]
params = text_file[1:]

# Plot the chain for each parameter
for i, param in enumerate(params):
    plt.plot(param)
    plt.xlabel("step number")
    plt.ylabel("param " + str(i+1))
    plt.title("parameter " + str(i+1) + " chain")
    plt.show()
    plt.clf()

# Calculate the Fourier transform for each parameter
n = params.shape[1]
params_fft = np.array([np.fft.fft(param, n) for param in params])

# Get the frequencies
k = np.fft.fftfreq(n)

# Sort both array with respect to the frequencies
indices = k.argsort()
k = k[indices]
params_fft = params_fft[:,indices]

# Set the k=0 component to 0 just so we can see what happens at low k
params_fft[:,n//2] = 0

# Plot Fourier transforms for all parameters
for i,param_fft in enumerate(params_fft):
    plt.plot(k, np.abs(param_fft))
    plt.xlabel("k")
    plt.ylabel("param " + str(i+1) + " fft")
    plt.title("parameter " + str(i+1) + " chain fft")
    plt.show()
    plt.clf()

# Okay so all the parameter ffts look pretty much like a spike around k = 0
# and drop off very fast, I think this means that the chain is converged.
# Also, the chain looks like a random walk which is consistent with a converged
# chain.

# Now we can calculate the parameters and their uncertainties using the mean and std
# of the chain:
params_from_chain = np.mean(params, axis = 1)
params_uncertainties = np.std(params, axis = 1)

# I'm going to use the uncertainties module here, make sure you have it installed
# you can install it by typing 'pip install uncertainties' in your python command line
params_ufloat = np.zeros((len(params_from_chain))).astype(uncertainties.core.Variable)

for i in range(len(params_ufloat)):
    params_ufloat[i] = ufloat(params_from_chain[i], params_uncertainties[i])

# Now let's print our optimal parameters
print("H0 =", params_ufloat[0])     # 68.44 +/- 0.08
print("ombh2 =", params_ufloat[1])  # 0.022369 +/- 0.000014
print("omch2 =", params_ufloat[2])  # 0.11738 +/- 0.00015
print("tau =", params_ufloat[3])    # 0.0808 +/- 0.0031
print("As =", params_ufloat[4])     # (2.200 +/- 0.014)e-09
print("ns =", params_ufloat[5])     # 0.9725 +/- 0.0007

# Now to calculate Omega_Lambda, we need h, Omega_b, Omega_C
h = params_ufloat[0] / 100
Omega_b = params_ufloat[1]
Omega_C = params_ufloat[2]

# Then, Omega_Lambda = 1 - Omega_b - Omega_C
Omega_Lambda = 1 - Omega_b - Omega_C
print("Omega_Lambda =", Omega_Lambda)   # 0.86025 +/- 0.00015