import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Here I'm just plotting the charge I got
rho = np.loadtxt("charge_new.txt", float, '#', ';')
# Here I'm summing over 3 values since my charge is not totally localized
# to the box, it bled out into the two neighbouring areas
rho_slice = np.sum(rho[445:556,449:452], axis=1)
fig, ax = plt.subplots(1,1)
ax.plot(rho_slice)
ax.set_xlabel("Position along the box")
ax.set_ylabel("Charge")
ax.set_title("Charge along one of the sides of the box")
fig.savefig("charge_new.pdf", bbox_inches='tight')
plt.clf()

# Now let's calcualte the potential with the charge
G = np.loadtxt("potential.txt", float, '#', ';')
G_FT = np.fft.fft2(G)
rho_FT = np.fft.fft2(rho)
V = np.fft.fftshift(np.real(np.fft.ifft2(G_FT * rho_FT)))

plt.imshow(V)
plt.colorbar()
plt.title("Potential everywhere in space with\nthe charge we calculated")
plt.savefig("potential_everywhere.pdf", bbox_inches='tight')

# If we now look inside the box to see how constant the potential is:
V_box = V[450:551,450:551]
V_avg = np.mean(V_box)
V_std = np.std(V_box)
V_max = np.max(V_box)
V_min = np.min(V_box)

print("Average potential in the box =", V_avg)  # -1.0350146110968898e-11
print("Standard deviation =", V_std)            # 1.7403058455915807e-11
print("Maximum =", V_max)                       # 3.4826622313337e-11
print("Minimum =", V_min)                       # -8.063084766347144e-11

# So it's not very constant at all, the program seems to have just shoved everything
# to 0 instead of 1... Well I'm not sure what to do about this...