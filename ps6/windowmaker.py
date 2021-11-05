# This just contains functions to produces windows and flat windows

import numpy as np

# Make a simple cosine window
def make_window(n):
    x=np.linspace(-np.pi,np.pi,n)
    return (1+np.cos(x))/2

# Make a window with a flat center
def make_flat_window(n,m):
    # Create the cosine window with a length of m
    cosine_window=make_window(m)

    # Set the flat window as a series of ones
    flat_window=np.ones(n)

    # Calculate half of m to separate the tapers
    p=m//2
    
    # Set the first p elements of the flat window to the first half of
    # the cosine window and same with the last p elements
    flat_window[:p]=cosine_window[:p]
    flat_window[-p:]=cosine_window[-p:]

    # Return the flat window
    return flat_window