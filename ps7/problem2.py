import numpy as np
from conjgrad import conjugate_grad_solve
from potential import sum_neighbors
import os

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

if __name__ == '__main__':
    # Set up relative path handling
    path = os.path.realpath(os.path.dirname(__file__))
    os.chdir(path)

    # Loading the mask / potential we want
    V = np.loadtxt("mask.txt", int, '#', ';')
    mask = np.where(V == 1)

    # Run the conjugate gradient solver
    b = get_RHS(V, mask)
    V_raw = conjugate_grad_solve(laplace, mask, b, 0*b, 1e-9, V.shape[0]*3)

    # Calculate the charge in raw and corrected forms
    V_new = np.copy(V_raw)
    V_new[mask] = b[mask]
    rho_raw = V_raw - sum_neighbors(V_raw)/4
    rho_new = V_new - sum_neighbors(V_new)/4

    # Save to text file for plotting (see plot.py)
    np.savetxt("potential_raw.txt", V_raw, '%.18e', ';')
    np.savetxt("potential_new.txt", V_new, '%.18e', ';')
    np.savetxt("charge_raw.txt", rho_raw, '%.18e', ';')
    np.savetxt("charge_new.txt", rho_new, '%.18e', ';')

    # So I tried to make the solver work to find the charge instead of the potential,
    # but it didn't seem to want to converge for some reason. I think it's probably because
    # my guess for the RHS potential is like only a square, but obviously the potential
    # is not zero everywhere except on the sides of the box, so that didn't work...
    # I wasn't super sure about how to calculate the residuals properly and get the
    # proper gradient descent if I was looking for the charge but my boundary condition
    # was a potential.