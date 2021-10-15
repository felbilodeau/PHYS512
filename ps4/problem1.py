import numpy as np
import camb
from matplotlib import pyplot as plt

# Okay so here I've copied the text script and added another block for
# the new parameters

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
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
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]


#plt.ion()  <- had to remove this because the figures were not showing

pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec-model
chisq=np.sum( (resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.") # We get a chi-squared of 15267.937150261601
#read in a binned version of the Planck PS for plotting purposes
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
plt.plot(ell,model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.show()

# I'm adding this to show it's not a great fit
degs_freedom = len(resid)-len(pars)
print("Chi-squared mean:", degs_freedom)
print("Chi-squared uncertainty:", np.sqrt(2*degs_freedom))
print("Chi-squared range: [", degs_freedom - np.sqrt(2*degs_freedom), ",", degs_freedom + np.sqrt(2*degs_freedom), "]")
print()

plt.clf()

# This is the block I copied with the new parameters
# I removed the redundant lines

pars=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec-model
chisq=np.sum( (resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.") # We get a chi-squared of 3272.205355920244
plt.plot(ell,model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.show()

# I'm adding this to show that it's a better fit
degs_freedom = len(resid)-len(pars)
print("Chi-squared mean:", degs_freedom)
print("Chi-squared uncertainty:", np.sqrt(2*degs_freedom))
print("Chi-squared range: [", degs_freedom - np.sqrt(2*degs_freedom), ",", degs_freedom + np.sqrt(2*degs_freedom), "]")

# So in both cases, the fit is not within the 1-sigma chi-squared range. Of course, it's much worse with
# the first set of parameters, but it's still not within the range with the better set of parameters.
# Therefore, I suppose neither of them would be considered an acceptable fit, although the second
# set of parameters is objectively better.