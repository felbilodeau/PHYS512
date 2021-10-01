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

    # Estimate dy
    dy = h*(k1/6 + k2/3 + k3/3 + k4/6)

    # Return the estimate for the new y value
    return y + dy

def rk4_stepd(fun, x, y, h):
    # We take a full step, and then two half steps
    dy1 = rk4_step(fun, x, y, h) - y
    y2a = rk4_step(fun, x, y, h/2)
    y2b = rk4_step(fun, x + h/2, y2a, h/2)

    # Note: this uses 3 times as many function calls as rk4_step

    # We now calculate the total step we took from the
    # two half steps
    dy2 = y2b - y

    # dy1 = dy_true + err
    # dy2 = dy_true + err / 16
    # 16dy2 - dy1 = 15dy_true
    # dy_true = (16dy2 - dy1) / 15
    dy = (16*dy2 - dy1) / 15
    return y + dy

if __name__ == "__main__":

    # Define the derivative function
    def dydx(x, y):
        return y / (1 + x**2)

    # Define our values of x
    x = np.linspace(-20, 20, 201)
    x_rk5 = np.linspace(-20, 20, 68) # 68 because 200 / 3 = 66.67 approx. 67, +1 = 68 so we do 67 steps

    # Set up the y arrays with the initial conditions
    y = np.zeros(len(x))
    y[0] = 1

    y_rk5 = np.zeros(len(x_rk5))
    y_rk5[0] = 1

    # Calculate each value of y in sequence using rk4_step
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        y[i] = rk4_step(dydx, x[i - 1], y[i - 1], h)

    # Calculate each value of y in sequence using rk4_stepd
    for i in range(1, len(x_rk5)):
        h = x_rk5[i] - x_rk5[i - 1]
        y_rk5[i] = rk4_stepd(dydx, x_rk5[i - 1], y_rk5[i - 1], h)

    # Calculate the true value (solution from Wolfram Alpha)
    x_true = np.linspace(-20, 20, 2001)
    y_true = np.exp(np.arctan(x_true) + np.arctan(20))

    # Calculate the max errors we have and print them for good measure
    err = np.max(y - np.exp(np.arctan(x) + np.arctan(20)))
    print("max error with rk4_step =", err)         # comes out to 3.57e-7

    err_rk5 = np.max(y_rk5 - np.exp(np.arctan(x_rk5) + np.arctan(20)))
    print("max error with rk4_stepd =", err_rk5)    # comes out to 4.28e-6

    print("error ratio =", err_rk5 / err)           # about 12 times as large

    # So as we can see, the error is actually higher with rk4_stepd with the
    # same number of function calls, since it uses 3 times as many calls we
    # are forced to have a step size 3 times as large, which means our error
    # increases regardless.

    # Plot all solutions
    plt.plot(x, y, label = "rk4_step")
    plt.plot(x_rk5, y_rk5, label = "rk4_stepd")
    plt.plot(x_true, y_true, label = "true solution")
    plt.legend()
    plt.show()
    plt.clf()