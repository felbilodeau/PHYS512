import numpy as np
import os
import camb

path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Okay so I'm basically copying the code from problem3.py except I will add
# the prior in the chisq calculation and start with the parameters i got from 
# the last chain as well as their uncertainties as the step size

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
def get_chisq(params, data, noise = None, par_priors = None, par_errs = None):
    # Get the prediction
    prediction = get_spectrum(params)

    # Check if noise was provided
    if noise is None:
        # If not, just sum the residuals squared
        chisq = np.sum((data - prediction)**2)
    else:
        # If it was, sum the residuals divided by the uncertainty squared
        chisq = np.sum(((data - prediction)/noise)**2)

    # Calculate the prior chisq
    if par_priors is not None:
        params_shift = params - par_priors
        chisq += np.sum((params_shift/par_errs)**2)

    return chisq
        
# Now we can write the mcmc driver
def run_mcmc(start_params, step_size, data, chisq_fun, nstep = 1000, noise = None, par_priors = None, par_errs = None):
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
        new_chisq = chisq_fun(new_params, data, noise, par_priors, par_errs)

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
        print("step", i+1, "out of", nstep, "done")

    # Return the params and chisq chains
    return params_chain, chisq_chain

if __name__ == '__main__':
    # Now we do similarly to what we did in paroblem3.py
    # Load the data to be compared to
    text_file = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt", float, '#').transpose()
    
    ell = text_file[0]
    data = text_file[1]
    noise = 0.5 * (text_file[2] + text_file[3])

    # Get the starting params as the ones we got from problem3.py
    start_params = np.array([68.44, 0.022369, 0.11738, 0.0808, 2.200e-9, 0.9725])
    nparam = len(start_params)

    # Define the step size the uncertainties we got from problem3.py
    step_size = np.array([0.08, 0.000014, 0.00015, 0.0031, 0.014e-9, 0.0007])

    # Define the number of steps
    nstep = 1000

    # Except now we add the prior
    par_priors = np.zeros((nparam))
    par_priors[3] = 0.0540

    par_errs = 1e20 * np.ones((nparam))
    par_errs[3] = 0.0074

    # Run the mcmc
    params_chain, chisq_chain = run_mcmc(start_params, step_size, data, get_chisq, nstep, noise, par_priors, par_errs)

    # Save the chain with the chisq as the first column
    output = open("planck_chain_tauprior.txt", "w")

    output.write("# chisq H0 ombh2 omch2 tau As ns\n")
    for i in range(nstep):
        output.write("%.18e " % chisq_chain[i])
        for j in range(len(start_params)):
            output.write("%.18e" % params_chain[j,i])
            if j != (len(start_params) - 1):
                output.write(" ")

        if i != (nstep - 1):
            output.write("\n")

    output.close()

    # Please check chain_check_tauprior.py for the analysis of this chain