from textwrap import wrap
import numpy as np
import camb
import matplotlib.pyplot as plt
import os

path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# I took this from the example just to load the CMB data
# and generate the model spectrum:
def get_spectrum(H0, ombh2, omch2, tau, As, ns,lmax=3000):
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    return tt[2:]


# Here we write the numerical differentiator
def get_derivative(fun, m0, dm = None):
    # Choose the optimal dm
    if dm == None:
        # we want f'''(m0) so we need
        # f''(m0 - h), f''(m0 + h)
        # f'(m0 - 2h), f'(m0), f'(m0 + 2h)

        h = 0.05*m0     # 5% difference

        # Calculate crudely all the derivatives we need
        dfdm_minus2h = (fun(m0 - h) - fun(m0 - 3*h)) / (2*h)
        dfdm_0 = (fun(m0 + h) - fun(m0 - h)) / (2*h)
        dfdm_plus2h = (fun(m0 + 3*h) - fun(m0 + h)) / (2*h)

        d2fdm2_minush = (dfdm_0 - dfdm_minus2h) / (2*h)
        d2fdm2_plush = (dfdm_plus2h - dfdm_0) / (2*h)

        d3fdm3 = (d2fdm2_plush - d2fdm2_minush) / (2*h)

        # Estimate the optimal dm
        epsilon = 1e-15
        dm = np.mean(np.abs(np.cbrt(fun(m0) * epsilon / d3fdm3)))

    # Calculate the derivative with the optimal dm
    derivative = (fun(m0 + dm) - fun(m0 - dm)) / (2*dm)
    return derivative[:2507]

def newton_step(fun, m, data, N_inverse):
    # Initialize A_m
    A_m = np.zeros((len(data), len(m)))
    
    # We need to create wrapper functions for each parameter derivative
    def wrapper_fun_H0(H0):
        return fun(H0, m[1], m[2], m[3], m[4], m[5])

    def wrapper_fun_ombh2(ombh2):
        return fun(m[0], ombh2, m[2], m[3], m[4], m[5])

    def wrapper_fun_omch2(omch2):
        return fun(m[0], m[1], omch2, m[3], m[4], m[5])

    def wrapper_fun_tau(tau):
        return fun(m[0], m[1], m[2], tau, m[4], m[5])

    def wrapper_fun_As(As):
        return fun(m[0], m[1], m[2], m[3], As, m[5])

    def wrapper_fun_ns(ns):
        return fun(m[0], m[1], m[2], m[3], m[4], ns)

    # Then we calculate all the derivatives
    A_m[:,0] = get_derivative(wrapper_fun_H0, m[0])
    A_m[:,1] = get_derivative(wrapper_fun_ombh2, m[1])
    A_m[:,2] = get_derivative(wrapper_fun_omch2, m[2])
    A_m[:,3] = get_derivative(wrapper_fun_tau, m[3])
    A_m[:,4] = get_derivative(wrapper_fun_As, m[4])
    A_m[:,5] = get_derivative(wrapper_fun_ns, m[5])

    # Calculate the residuals
    r = fun(m[0], m[1], m[2], m[3], m[4], m[5])[:2507] - data

    # Calculate delta_m and return
    delta_m = np.linalg.inv(A_m.transpose() @ N_inverse @ A_m) @ A_m.transpose() @ N_inverse @ r
    return delta_m

if __name__ == '__main__':

    # Set up the initial parameters
    m0 = np.array([69,0.022,0.12,0.06,2.1e-9,0.95])

    # Load the power spectrum from the file
    power_spectrum = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt', float, '#')
    ell = power_spectrum[:,0]
    data = power_spectrum[:,1]
    uncertainties = 0.5 * (power_spectrum[:,2] + power_spectrum[:,3])

    # Define the function
    def fun(H0, ombh2, omch2, tau, As, ns):
        return get_spectrum(H0, ombh2, omch2, tau, As, ns, 2508)

    # Calculate N_inverse and initialize delta_m
    N_inverse = np.diag(1 / uncertainties**2)

    delta_m = np.zeros((len(m0)))
    delta_m[:] = 1

    # Tolerance for delta_m as a fraction of the parameter
    tol = 6     # <- with this tolerance it manages to do one step, it diverges on the second step

    # Loop and update m
    while(np.max(np.abs(delta_m / m0)) >= tol):
        delta_m = newton_step(fun, m0, data, N_inverse)
        m0 += delta_m
        print(delta_m / m0)

    print("m0 =", m0)
    np.savetxt("planck_fit_params.txt", m0)

    # Well I've spent an embarassing amount of time on this and I can't get it to converge at all...
    # This program is super slow because of the dm calculation in get_derivative, and
    # the value for tau seems to diverge after the first step if tol <= 5.62 or something...
    # And now I'm out of time to do this assignment. So I guess that's all folks!