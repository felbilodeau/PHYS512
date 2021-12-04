import numpy as np
from conjgrad import conjugate_grad_solve
from potential import sum_neighbors
import matplotlib.pyplot as plt

G = np.loadtxt("potential.txt", float, '#', ';')
G_FT = np.fft.fft(G)

def laplace(V, mask):
    # Copy the potential and set it to zero on the mask
    temp = np.copy(V)
    temp[mask] = 0

    # Calculate the sum of the neighbours, and set it to 0 on the mask
    total = sum_neighbors(temp)
    total[mask] = 0

    # Return the charge
    return temp - total / 4

def get_RHS(V, mask):
    # Copy the potential and set it to 0 everywhere
    temp = 0*V

    # Restore the values on the mask
    temp[mask] = V[mask]

    # Calculate the sum of the neighbours on the on the mask
    RHS = sum_neighbors(V) / 4

    # Set the RHS on the mask to 0
    RHS[mask] = 0
    return RHS

V = np.loadtxt("mask.txt", int, '#', ';')
mask = np.where(V == 1)

b = get_RHS(V, mask)
V_raw = conjugate_grad_solve(laplace, mask, b, 0*b, 1e-9, V.shape[0]*3)
V_new = np.copy(V_raw)
V_new[mask] = b[mask]

np.savetxt("potential_raw.txt", V_raw, '%.18e', ';')

plt.imshow(V_new)
plt.show()