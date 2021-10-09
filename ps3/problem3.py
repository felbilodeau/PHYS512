import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from uncertainties import ufloat

# Setting the current working directory to the current folder
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Ok so we have z - z0 = a((x - x0)^2 + (y - y0)^2)
# => z = a(x^2 - 2xx0 + x0^2 + y^2 - 2yy0 + y0^2) + z0
# => z = ax^2 - 2ax0x + ax0^2 + ay^2 - 2ay0y + ay0^2 + z0
# => z = a*(x^2 + y^2) + (-2ax0)*x + (-2ay0)*y + 1*(ax0^2 + ay0^2 + z0)

# So we can choose the parameters:
# alpha = a                     => associated with x^2 + y^2
# beta = -2ax0                  => associated with x
# gamma = -2ay0                 => associated with y
# sigma = ax0^2 + ay0^2 + z0    => associated with 1

# Such that we have z = alpha*x^2 + alpha*y^2 + beta*x + gamma*y + sigma
# We can then combine these to find the original parameters

# First let's load the data from the file

data = np.loadtxt('dish_zenith.txt', float, '#', ' ').transpose()

x = data[0]
y = data[1]
z = data[2]

# For the A matrix we need 1, x, y, x^2 + y^2
A = np.zeros((len(x), 4))
A[:,0] = 1
A[:,1] = x
A[:,2] = y
A[:,3] = x**2 + y**2

# The d matrix is just the z matrix
m = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ z
print("m =", m)

sigma = m[0]    # -1.51231182e+03
gamma = m[1]    #  4.53599028e-04
beta = m[2]     # -1.94115589e-02
alpha = m[3]    #  1.66704455e-04

# We can now calculate the original, non-linear parameters
a  = alpha
x0 = beta / (-2*a)
y0 = gamma / (-2*a)
z0 = sigma - a*(x0**2 + y0**2)

print("a =", a)    # 0.00016670445477401445
print("x0 =", x0)  # 58.221476081579354
print("y0 =", y0)  # -1.3604886221979873
print("z0 =", z0)  # -1512.8772100367912

# Let's plot the result to see if it works well
X = np.linspace(-3000, 3000, 1001)
Y = np.linspace(-3000, 3000, 1001)
X, Y = np.meshgrid(X, Y)
Z = a*((X - x0)**2 + (Y - y0)**2) + z0

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, alpha = 0.8)
ax.scatter(x, y, z, color = 'black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
plt.clf()

# Seems to work pretty well, but let's see the maximum error
z_fit = a*((x - x0)**2 + (y - y0)**2) + z0
err = np.max(np.abs(z - z_fit))
print("max_err =", err)      # 99.11245496528983 <- that's kinda high, 10 cm?!

# So to find the noise, we find the standard error:
std_err = np.sqrt(1 / (len(z) - 4) * np.sum((z - z_fit)**2))
print("std_err =", std_err)     # 40.984949945891884 <- I mean okay, with noise that high that makes sense

# and the mean of the noise should be 0:
mean_noise = np.mean(z - z_fit)
print("mean_noise =", mean_noise)   # -0.30210434430354516 <- yeah that's pretty good

# So to find the covariance of the parameters, we need to fill the N array
# now that we know what the noise is
variances = [std_err**2 for i in range(len(z))]
N = np.diag(variances)

# Really we should have done this before we fit the data, but we would have needed
# to know the uncertainties in the data beforehand to do so

# Then we calculate the covariance matrix
covariance = np.linalg.inv(A.transpose() @ np.linalg.inv(N) @ A)
uncertainties = np.sqrt(np.diag(covariance))
print("uncertainties =", uncertainties)

# The uncertainty in the parameter 'a' is then:
uncertainty_a = uncertainties[3]
print("uncertainty_a =", uncertainty_a)     # 7.017171577523507e-07

# I'm going to use the 'uncertainties' module here to calculate the focal length
# If you don't already have it, just use 'pip install uncertainties'
a_ufloat = ufloat(a, uncertainty_a)

# We know that a = 1 / 4f => f = 1 / 4a
focal_length = 1 / 4 / a_ufloat
print("focal_length =", focal_length)   # focal_length = 1500 +/- 6

# So as we can see we get exactly what was expected for the focal length,
# with an uncertainty of +/- 6 mm which is a relative uncertainty of 0.4%