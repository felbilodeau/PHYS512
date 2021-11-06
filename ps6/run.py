# This is the file which needs to be run to get the results.
# The methods are all described in more detail in their
# specific files

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from power_spectrum_density import get_average_psd
from simple_read_ligo import read_file, read_template
from windowmaker import make_flat_window, make_window
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy import signal
from whiten import whiten
from match_filter import match_filter

path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Okay so here is the gist of what we need to do to clean the data

# 1. Estimate the power spectrum density of the data
# 2. Whiten the data using the PSD
# 3. Band-pass the data to remove high frequency noise

filename = "LOSC_Event_tutorial/LOSC_Event_tutorial/H-H1_LOSC_4_V2-1126259446-32.hdf5"
template_filename = "LOSC_Event_tutorial/LOSC_Event_tutorial/GW150914_4_template.hdf5"

strain,dt,utc = read_file(filename)
th, tl = read_template(template_filename)
n = len(strain)
window = make_window(n)
dwindow = np.blackman(n)

plt.plot(dwindow)
plt.plot(window)
plt.show()
plt.clf()

fs = 4096
NFFT = fs * 4

#strain_white = whiten(strain, NFFT, dt)

SNR = match_filter(strain, th, tl, dt, fs, window, NFFT)

plt.plot(SNR)
plt.show()