import numpy as np
import camb
import matplotlib.pyplot as plt
import os

# Set the path handling to relative
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# First we need the function to get the prediction from CAMB
def get_spectrum(pars,lmax=2508):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    spectrum = tt[2:]
    return spectrum[:(lmax - 1)]

# Then we need a function to calculate the chi-squared with the params
# and uncertainties
def get_chisq(params, data, noise = None):
    # Get the prediction
    prediction = get_spectrum(params)

    # Check if noise was provided
    if noise == None:
        # If not, just sum the residuals squared
        return np.sum((data - prediction)**2)
    else:
        # If it was, sum the residuals divided by the uncertainty squared
        return np.sum(((data - prediction)/noise)**2)

# Now we can write the mcmc driver
def run_mcmc(start_params, step_size, data, chisq_fun, nstep = 100, noise = None):
    # Initialize some arrays
    params = start_params.copy()
    nparam = len(start_params)
    chisq_chain = np.zeros((nstep))
    params_chain = np.zeros((nparam, nstep))
    chisq = chisq_fun(params, data, noise)

    # Now we do the steps
    for i in range(nstep):
        # Get a random step and calculate the new chisq
        new_params = params + step_size * np.random.randn(nparam)
        new_chisq = chisq_fun(new_params, data, noise)

        # Calculate the delta_chisq and probability
        delta_chisq = new_chisq - chisq
        probability = np.exp(-0.5 * delta_chisq)

        # Decide if we accept the step of not
        accept = np.random.rand(1) < probability

        # Check if it was accepted, update if it was
        if accept:
            params = new_params
            chisq = new_chisq

        # Update the chain
        params_chain[:,i] = params
        chisq_chain[i] = chisq

    # Return the params and chisq chains
    return params_chain, chisq_chain

if __name__ == '__main__':
    # Load the data to be compared to
    text_file = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", float, '#').transpose()
    
    ell = text_file[0]
    data = text_file[1]
    noise = 0.5 * (text_file[2] + text_file[3])

    # Load the parameters we found from problem 2 as our starting point
    text_file = np.loadtxt("planck_fit_params.txt", float, '#')

    start_params = text_file[0]
    noise = text_file[1]