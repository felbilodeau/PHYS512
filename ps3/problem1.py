import numpy as np
import matplotlib.pyplot as plt

def rk4_step(fun, x, y, h):
    # Calculate the slope at the starting point
    k1 = fun(x, y)

    # Estimate the slope at the midpoint
    k2 = fun(x + h/2, y + k1*h/2)

    # Get another estimate of the slope at the midpoint
    k3 = fun(x + h/2, y + k2*h/2)

    # Estimate the slope at the endpoint
    k4 = fun(x + h, y + k3*h)

    # Return the estimate for the new y value
    return y + h*(k1/6 + k2/3 + k3/3 + k4/6)



if __name__ == "__main__":

    # Define the derivative function
    def dydx(x, y):
        return y / (1 + x**2)

    # Define our values of x
    x = np.linspace(-20, 20, 201)

    # Set up the y array with the initial condition
    y = np.zeros(201)
    y[0] = 1

    # Calculate h so we have 200 steps
    h = (x[-1] - x[0]) / 200

    # Calculate each value of y in sequence using rk4_step
    for i in range(1, 201):
        y[i] = rk4_step(dydx, x[i - 1], y[i - 1], h)

    # Calculate the true value (solution from Wolfram Alpha)
    x_true = np.linspace(-20, 20, 2001)
    y_true = np.exp(np.arctan(x_true) + np.arctan(20))

    # Calculate the max error we have and print it for good measure
    err = np.max(y - np.exp(np.arctan(x) + np.arctan(20)))
    print("max error with rk4_step =", err)

    # Plot both
    plt.plot(x, y)
    plt.plot(x_true, y_true)
    plt.show()