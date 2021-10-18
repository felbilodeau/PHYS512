import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

text_file = np.loadtxt("planck_chain.txt", float, '#').transpose()

chisq = text_file[0]
params = text_file[1:]

H0 = params[0]
ombh2 = params[1]
omch2 = params[2]
tau = params[3]
As = params[4]
ns = params[5]

for i, param in enumerate(params):
    plt.plot(param)
    plt.xlabel("step number")
    plt.ylabel("param " + str(i+1))
    plt.title("parameter " + str(i+1) + " chain")
    plt.show()
    plt.clf()

n = params.shape[1]
params_fft = np.array([np.fft.fft(param, n) for param in params])
k = np.fft.fftfreq(n)
indices = k.argsort()
k = k[indices]
params_fft = params_fft[:,indices]
params_fft[:,n//2] = 0

for i,param_fft in enumerate(params_fft):
    plt.plot(k, np.abs(param_fft))
    plt.xlabel("k")
    plt.ylabel("param " + str(i+1) + " fft")
    plt.title("parameter " + str(i+1) + " chain fft")
    plt.show()
    plt.clf()