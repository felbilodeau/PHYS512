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