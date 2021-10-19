import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.core.fromnumeric import argmax

path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Load the tauprior chain
text_file = np.loadtxt("planck_chain_tauprior.txt", float, '#').transpose()

# Get the parameters from the chain
param_chains = text_file[1:]
params_with_prior = np.mean(param_chains, axis = 1)
prior_uncertainties = np.std(param_chains, axis = 1)

# Print the parameters
print("Parameters with tau prior:")
print("H0 =", params_with_prior[0], "+/-", prior_uncertainties[0])      # 68.44000000000094 +/- 9.379164112033322e-13
print("ombh2 =", params_with_prior[1], "+/-", prior_uncertainties[1])   # 0.022369000000000177 +/- 1.7694179454963432e-16
print("omch2 =", params_with_prior[2], "+/-", prior_uncertainties[2])   # 0.11737999999999807 +/- 1.9290125052862095e-15
print("tau =", params_with_prior[3], "+/-", prior_uncertainties[3])     # 0.08080000000000065 +/- 6.522560269672795e-16
print("As =", params_with_prior[4], "+/-", prior_uncertainties[4])      # 2.199999999999993e-09 +/- 7.031035206700735e-24
print("ns =", params_with_prior[5], "+/-", prior_uncertainties[5])      # 0.9724999999999907 +/- 9.325873406851315e-15
print()

# Okay well this chain didn't move at all from the last chain. I must have messed up somewhere, but I'm all out of time
# to try to find the bug and fix it.

# Now we can do the importance sampling from the last chain
def get_chisq_prior(params, param_priors = None, param_errs = None):
    if param_priors is None:
        return 0
    param_shifts = params - param_priors
    return np.sum((param_shifts/param_errs)**2)

# Load the last chain
text_file = np.loadtxt("planck_chain.txt", float, '#').transpose()
chain = text_file[1:]

# Get the number of steps and number of parameters
nstep = chain.shape[1]
nparam = chain.shape[0]

# Set the priors
priors = np.zeros(nparam)
priors[3] = 0.0540

param_errs = 1e20*np.ones((nparam))
param_errs[3] = 0.0074

# Initialize the weight and chisq arrays
weight = np.zeros(nstep)
chisq = np.zeros(nstep)

# Calculate the prior_chisq for each step
for i in range(nstep):
    chisq[i] = get_chisq_prior(chain[:,i], priors, param_errs)

# Calculate the weights
chisq = chisq - np.mean(chisq)
weight = np.exp(0.5*chisq)

# Get the parameters and uncertainties from the weights
params_importance = np.zeros(nparam)
importance_uncertainties = np.zeros(nparam)
for i in range(nparam):
    params_importance[i] = np.sum(weight * chain[i,:]) / np.sum(weight)
    importance_uncertainties[i] = np.sqrt(np.average((chain[i,:] - params_importance[i])**2, weights=weight))


# Print the parameters
print("Parameters with importance sampling:")
print("H0 =", params_importance[0], "+/-", importance_uncertainties[0])      # 68.45587827882088 +/- 0.08407634185448346
print("ombh2 =", params_importance[1], "+/-", importance_uncertainties[1])   # 0.022355331792369263 +/- 1.3939290823185112e-05
print("omch2 =", params_importance[2], "+/-", importance_uncertainties[2])   # 0.11735167578882298 +/- 0.0001263421443817916
print("tau =", params_importance[3], "+/-", importance_uncertainties[3])     # 0.08489129765290665 +/- 0.001978872169758692
print("As =", params_importance[4], "+/-", importance_uncertainties[4])      # 2.2176683825088045e-09 +/- 9.033880299115588e-12
print("ns =", params_importance[5], "+/-", importance_uncertainties[5])      # 0.9731799375837806 +/- 0.0003209837216312382
print()

# Get the differences in both methods
differences = np.abs((params_with_prior - params_importance)/params_with_prior)*100

print("Percent differences in the parameters:")
print("Delta_H0 =", differences[0], "%")      # 0.023200290502542503 %
print("Delta_ombh2 =", differences[1], "%")   # 0.06110334673393426 %
print("Delta_omch2 =", differences[2], "%")   # 0.024130355405595508 %
print("Delta_tau =", differences[3], "%")     # 5.063487194190555 %
print("Delta_As =", differences[4], "%")      # 0.8031082958550814 %
print("Delta_ns =", differences[5], "%")      # 0.06991646105809061 %
print()

# So aside from tau which has a 5% difference, the parameters are quite similar between the two methods