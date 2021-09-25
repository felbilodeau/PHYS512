import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scpi
import legendre

# Well I'm not too sure what "my integrator" is since we didn't
# write one yet, so I'm just going to take the one in the
# file "integrate_things.py" on github I guess...

# This is the function we need to integrate
def fun(theta, z):
    # We will assume here that Q / 8 pi epsilon = R = 1 for simplicity
    return (z - np.cos(theta)) * np.sin(theta) / (z**2 - 2*z*np.cos(theta) + 1)**(3/2)

# This is the function I'm taking from github
def integrate(fun,xmin,xmax,dx_targ,z,ord=2,verbose=False):
    coeffs=legendre.integration_coeffs_legendre(ord+1)
    if verbose: #should be zero
        print("fractional difference between first/last coefficients is "+repr(coeffs[0]/coeffs[-1]-1))

    npt=int((xmax-xmin)/dx_targ)+1
    nn=(npt-1)%(ord)
    if nn>0:
        npt=npt+(ord-nn)
    assert(npt%(ord)==1)
    npt=int(npt)

    x=np.linspace(xmin,xmax,npt)
    dx=np.median(np.diff(x))    
    dat=fun(x,z)

    #we could have a loop here, but note that we can also reshape our data, then som along columns, and only then
    #apply coefficients.  Some care is required with the first and last points because they only show up once.
    mat=np.reshape(dat[:-1],[(npt-1)//(ord),ord]).copy()
    mat[0,0]=mat[0,0]+dat[-1] #as a hack, we can add the last point to the first
    mat[1:,0]=2*mat[1:,0] #double everythin in the first column, since each element appears as the last element in the previous row

    vec=np.sum(mat,axis=0)
    tot=np.sum(vec*coeffs[:-1])*dx
    return tot


# Here we set our values of z:
z = np.linspace(0,2,1001)

# We initialize the output arrays
E_custom = np.zeros((len(z)))
E_quad = np.zeros((len(z)))

# Do the integrations
for i in range(len(z)):
    # I chose a dx of 0.001 for no reason in particular
    E_custom[i] = integrate(fun, 0, np.pi, 0.001, z[i])
    E_quad[i] = scpi.quad(fun, 0, np.pi, z[i])[0]


fix, ax = plt.subplots(1,1)
ax.set_xlabel("Distance from centre of the sphere")
ax.set_ylabel("Electric Field")
ax.plot(z, E_quad, label="scipy.integrate.quad")
ax.plot(z, E_custom, label="github integrate")
ax.legend()
plt.show()
plt.clf()

# So it seems that scipy.integrate.quad does not care about the singularity
# at z = R, it just goes right through. However, the integrate function
# we saw in class gives us 2 division by 0 warnings.