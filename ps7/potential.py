import numpy as np

# This takes the sum of the neighbors of each point
def sum_neighbors(mat):
    tot=0
    for i in range(len(mat.shape)):
        tot=tot+np.roll(mat,1,axis=i)
        tot=tot+np.roll(mat,-1,axis=i)
    return tot

# Set the size of the Green's function and calculate the origin
# I'm making this very big so I can make a big box later, you may
# want to make this smaller to run it
size = 1001
origin = (size - 1) // 2

# Create the potential array and seed it at the origin
V = np.zeros((size, size))
V[origin,origin] = 1

# Iterate niter times
# You can make this smaller if you want to run it because it takes
# a while to run with such a large niter
niter = 10000
for i in range(niter):
    # Just print something to see the progress
    print("iteration", i+1, "out of", niter)

    # Set all points to the average of their neighbours
    avg = sum_neighbors(V)/4
    V = avg

    # Make sure the charge at the origin is 1
    V[origin,origin] += 1

# Shift the potential so we have V[0,0] = 1
shift = 1 - V[origin,origin]
V += shift

# Recalculate the average to check my answers
avg = sum_neighbors(V)/4

# Print some things                                          These are the results of my prints
print("rho[0,0] =", V[origin,origin] - avg[origin,origin])  # rho[0,0] = 1.0000000000000004
print("V[1,0] =", V[origin+1,origin])                       # V[1,0] = -4.440892098500626e-16
print("V[2,0] =", V[origin+2,origin])                       # V[2,0] = -0.45339362157990104
print("V[5,0] =", V[origin+5,origin])                       # V[5,0] = -1.0508459260774505

# Save the potential to a file so we can load it without having to calculate it each time
# If you change niter or size you'll want to comment this out otherwise the rest will
# now work with the new parameters for this calculation
np.savetxt("potential.txt", V, '%.18e', ';')