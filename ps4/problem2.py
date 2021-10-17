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
        f_minush = fun(m0 - h)
        f_minus3h = fun(m0 - 3*h)
        f_plush = fun(m0 + h)
        f_plus3h = fun(m0 + 3*h)

        dfdm_minus2h = (f_minush - f_minus3h) / (2*h)
        dfdm_0 = (f_plush - f_minush) / (2*h)
        dfdm_plus2h = (f_plus3h - f_plush) / (2*h)

        d2fdm2_minush = (dfdm_0 - dfdm_minus2h) / (2*h)
        d2fdm2_plush = (dfdm_plus2h - dfdm_0) / (2*h)

        d3fdm3 = (d2fdm2_plush - d2fdm2_minush) / (2*h)

        # Estimate the optimal dm
        epsilon = 1e-15
        dm = np.mean(np.abs(np.cbrt(fun(m0) * epsilon / d3fdm3)))

    # Calculate the derivative with the optimal dm
    derivative = (fun(m0 + dm) - fun(m0 - dm)) / (2*dm)
    return derivative

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
    r = data - fun(m[0], m[1], m[2], m[3], m[4], m[5])

    # Wow okay sorry I need to vent, I spent like 6 hours debugging this program
    # because it was running but the chi-squared kept increasing with each step instead
    # of decreasing and you know what the problem was?! In the line above, I wrote
    # r = fun(m[0], m[1], m[2], m[3], m[4], m[5]) - data instead and that ruined
    # everything and it took me so long to catch I am so mad at myself oh my god
    # plz send help :'(

    # Calculate delta_m and return
    delta_m = np.linalg.inv(A_m.transpose() @ N_inverse @ A_m) @ A_m.transpose() @ N_inverse @ r
    return delta_m, A_m

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
        spectrum = get_spectrum(H0, ombh2, omch2, tau, As, ns, 2508)[:2507]
        return spectrum

    # Calculate N_inverse
    N_inverse = np.diag(1 / uncertainties**2)

    # Tolerance for delta_chisq
    tol = 0.01

    model = fun(m0[0], m0[1], m0[2], m0[3], m0[4], m0[5])
    chisq = (data - model).transpose() @ N_inverse @ (data - model)
    delta_chisq = tol + 1 # Just to make sure the loop runs at least once
    print("intial chisq =", chisq)  # ~3272

    # Loop and update m
    while(delta_chisq >= tol):
        # Calculate delta_m for this step
        delta_m, A_m = newton_step(fun, m0, data, N_inverse)

        # We keep track of A_m to calculate the uncertainties later

        # Update m0 to m0 + delta_m
        m0 += delta_m

        # Calculate the new chisq from the new parameters
        model = fun(m0[0], m0[1], m0[2], m0[3], m0[4], m0[5])
        new_chisq = (data - model).transpose() @ N_inverse @ (data - model)

        # Calculate delta_chisq and update chisq to new_chisq
        delta_chisq = np.abs(new_chisq - chisq)
        chisq = new_chisq

        print("delta_chisq =", delta_chisq)

    # Print the final chisq
    print("final chisq =", chisq)   # ~2576

    # Calculate the covariance and uncertainties using A_m and N_inverse
    covariance = np.linalg.inv(A_m.transpose() @ N_inverse @ A_m)
    par_uncertainties = np.sqrt(np.diag(covariance))

    # Print the parameters we obtained
    print("m0 =", m0)

    # Open a file to save the parameters and their uncertainties
    output = open("planck_fit_params.txt", 'w')

    output.write("# H0 ombh2 omch2 tau As ns\n")
    output.write("%.18e %.18e %.18e %.18e %.18e %.18e\n" % (m0[0], m0[1], m0[2], m0[3], m0[4], m0[5]))
    output.write("%.18e %.18e %.18e %.18e %.18e %.18e" % (par_uncertainties[0], par_uncertainties[1],
                     par_uncertainties[2], par_uncertainties[3], par_uncertainties[4], par_uncertainties[5]))
    
    output.close()

    # So we're not quite in range of the optimal chi-squared value of 2501 +/- 70 with this fit,
    # but with a final chi-squared of 2576 we're very nearly there, and this might be close
    # to the best we can do with this fit given that the delta_chisq for the last step is
    # around 4e-4 and should only become smaller as the steps increase.

    # Also it asks to keep track of the curvature matrix but I'm not sure what that means,
    # am I supposed to calculate the Hessian matrix or something? I don't know, I might
    # come back to this later if I find out.