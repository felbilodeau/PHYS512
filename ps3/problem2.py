import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import BDF

# Index         Elements      Half-Life         Half-Life (years)
#
# 0             U238          4.468e9 years     4.468e9
# 1             Th234         24.1 days         0.06603
# 2             Pa234         6.7 hours         7.65e-4
# 3             U234          245 500 years     245500
# 4             Th230         75 380 years      75380
# 5             Ra226         1 600 years       1600
# 6             Rn222         3.8235 days       0.0104753
# 7             Po218         3.1 minutes       5.9e-6
# 8             Pb214         26.8 minutes      5.1e-5
# 9             Bi214         19.9 minutes      3.79e-5
# 10            Po214         164.3e-6 seconds  5.2093e-12
# 11            Pb210         22.3 years        22.3
# 12            Bi210         5 015 years       5015
# 13            Po210         138.376 days      0.379112
# 14            Pb206         Stable            Stable

def fun(x, y, half_life = [4.468e9, 0.06603, 7.65e-4, 245500, 75380, 1600, 0.0104753, 5.9e-6, 5.1e-5, 3.79e-5, 5.2093e-12, 22.3, 5015, 0.379112]):

    # Create the dydx array
    dydx = np.zeros(len(half_life) + 1)

    # Calculate the divisions so we only do them once
    decay_U238 = y[0] / half_life[0]
    decay_Th234 = y[1] / half_life[1]
    decay_Pa234 = y[2] / half_life[2]
    decay_U234 = y[3] / half_life[3]
    decay_Th230 = y[4] / half_life[4]
    decay_Ra226 = y[5] / half_life[5]
    decay_Rn222 = y[6] / half_life[6]
    decay_Po218 = y[7] / half_life[7]
    decay_Pb214 = y[8] / half_life[8]
    decay_Bi214 = y[9] / half_life[9]
    decay_Po214 = y[10] / half_life[10]
    decay_Pb210 = y[11] / half_life[11]
    decay_Bi210 = y[12] / half_life[12]
    decay_Po210 = y[13] / half_life[13]

    # Populate dydx
    dydx[0] = -decay_U238
    dydx[1] = decay_U238 - decay_Th234
    dydx[2] = decay_Th234 - decay_Pa234
    dydx[3] = decay_Pa234 - decay_U234
    dydx[4] = decay_U234 - decay_Th230
    dydx[5] = decay_Th230 - decay_Ra226
    dydx[6] = decay_Ra226 - decay_Rn222
    dydx[7] = decay_Rn222 - decay_Po218
    dydx[8] = decay_Po218 - decay_Pb214
    dydx[9] = decay_Pb214 - decay_Bi214
    dydx[10] = decay_Bi214 - decay_Po214
    dydx[11] = decay_Po214 - decay_Pb210
    dydx[12] = decay_Pb210 - decay_Bi210
    dydx[13] = decay_Bi210 - decay_Po210
    dydx[14] = decay_Po210

    # Return dydx * ln(2) since these are half-lives
    return np.log(2) * dydx