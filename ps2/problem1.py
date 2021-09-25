import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scpi

# Well I'm not too sure what "my integrator" is since we didn't
# write one yet, so I'm just going to take the one we did in class
# I guess...

# This is the function we need to integrate
def fun(theta, z):
    # We will assume here that Q / 8 pi epsilon = R = 1 for simplicity
    return (z - np.cos(theta)) * np.sin(theta) / (z**2 - 2*z*np.cos(theta) + 1)**(3/2)


# Here we set our values of z:
z = np.linspace(0,2,1001)
print(z)
E_custom = np.zeros((len(z)))
E_quad = np.zeros((len(z)))

for i in range(len(z)):
    E_quad[i] = scpi.quad(fun, 0, np.pi, z[i])[0]

plt.plot(z, E_quad)
plt.show()