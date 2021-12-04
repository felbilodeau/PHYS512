import numpy as np
from conjgrad import conjugate_grad_solve
from potential import sum_neighbors
import matplotlib.pyplot as plt

G = np.loadtxt("potential.txt", float, '#', ';')
G_FT = np.fft.fft(G)

def laplace(rho, mask):
    global G_FT
    # Copy the charge matrix
    temp = np.copy(rho)*0

    # Set the charge off the mask to 0
    temp[mask] = rho[mask]

    # Apply the convolution
    output = np.fft.ifft(G_FT * np.fft.fft(temp))

    # Return the potential
    return np.real(output)

def get_RHS(V, mask):
    # Copy the potential matrix and set everything to 0
    temp = np.copy(V) * 0

    # Restore the values on the mask
    temp[mask] = V[mask]

    # Since our RHS is the potential we can just return temp
    return temp

V = np.loadtxt("mask.txt", int, '#', ';')
mask = np.where(V == 1)

b = get_RHS(V, mask)
V_new = conjugate_grad_solve(laplace, mask, b, 0*b, 1e-9, V.shape[0]*3)
V_new[mask] = V[mask]

plt.imshow(V_new)
plt.show()