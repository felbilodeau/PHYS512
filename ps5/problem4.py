import numpy as np
import matplotlib.pyplot as plt
import os

# Set up relative path handling
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Okay so this is the equivalent of np.convolve.
# Here is what we are doing:
def conv_safe(f, g):
    # First, get the lengths of both arrays
    nf = len(f)
    ng = len(g)

    # We are doing a 'full' convolve, so the output will be of size nf + ng - 1
    output = np.zeros(nf +  ng - 1)

    # The first array is just f padded on both sides with ng - 1 zeros
    array_1 = np.zeros(nf + 2*(ng - 1))
    array_1[ng - 1:ng + nf - 1] = f[:]

    # The second array is just g but reversed so [6 1 3 2] -> [2 3 1 6]
    array_2 = g[::-1]

    # Now we line up the arrays and take the dot products like so:
    #
    # Let's say f = [6 1 3 2] and g = [2 4]
    # array_1 = [0 6 1 3 2 0], array_2 = [4 2]
    #
    # [4 2]
    # [0 6 1 3 2 0]     => output[0] = 4*0 + 2*6 = 12
    #
    #   [4 2]
    # [0 6 1 3 2 0]     => output[1] = 4*6 + 2*1 = 26
    # 
    # and so on until we get to the last step:
    #         [4 2]
    # [0 6 1 3 2 0]     => output[-1] = 4*2 + 2*2 = 8
    for i in range(nf + ng - 1):
        output[i] = np.dot(array_1[i:i+ng],array_2)

    # Finally we return the output
    return np.array(output)

# Note that this type of convolution may not be ideal for all applications.
# As you can see from the plot in 'conv_safe_test.png', we have edge errors
# with the 'full' method, while the 'valid' method doesn't pad with zeros

# Here we generate the convolution with conv_safe
x1 = np.linspace(-5, 5, 1001)
x2 = np.linspace(-5, 5, 501)
y1 = x1**2
y2 = np.exp(-1*np.abs(x2))
n_deltas = 1001 + 501 - 1
deltas = np.linspace(-n_deltas / 2, n_deltas / 2, n_deltas)
conv = conv_safe(y1, y2)

# The 'valid' method allows to ignore edge effects,
# but I had to double the range and size to show this.
# Otherwise, the 'valid' method creates much less data points
x1 = np.linspace(-10, 10, 2001)
x2 = np.linspace(-10, 10, 1001)
y1 = x1**2
y2 = np.exp(-1*np.abs(x2))
conv_true = np.convolve(y1, y2, 'valid')
n_deltas_trues = len(conv_true)
deltas_true = np.linspace(-n_deltas_trues/2, n_deltas_trues/2, n_deltas_trues)

plt.plot(deltas, conv, label="'full' method (conv_safe)")
plt.plot(deltas_true, conv_true, label = "'valid' method (np.convolve)")
plt.title("Comparison of 'full' and 'valid' convolutions")
plt.legend()
plt.savefig('conv_safe_test.png', bbox_inches='tight')