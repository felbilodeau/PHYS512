# This is the file which needs to be run to get the results.
# The methods are all described in more detail in their
# specific files

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from power_spectrum_density import get_average_psd
from simple_read_ligo import read_file, read_template
from windowmaker import make_flat_window
from scipy.interpolate import interp1d
from whiten import whiten
from match_filter import match_filter

# Setup relative path handling
path = os.path.realpath(os.path.dirname(__file__))
os.chdir(path)

# Change this list to choose which events get analyzed
events_to_analyze = [0,1,2,3]
# 0: GW150914
# 1: LVT151012
# 2: GW151226
# 3: GW170104

# This is from the json file, but since it's just 4 events I figured I'd just copy them since I don't know how jsons work yet
events = [["H-H1_LOSC_4_V2-1126259446-32.hdf5", "L-L1_LOSC_4_V2-1126259446-32.hdf5", "GW150914_4_template.hdf5"],
          ["H-H1_LOSC_4_V2-1128678884-32.hdf5", "L-L1_LOSC_4_V2-1128678884-32.hdf5", "LVT151012_4_template.hdf5"],
          ["H-H1_LOSC_4_V2-1135136334-32.hdf5", "L-L1_LOSC_4_V2-1135136334-32.hdf5", "GW151226_4_template.hdf5"],
          ["H-H1_LOSC_4_V1-1167559920-32.hdf5", "L-L1_LOSC_4_V1-1167559920-32.hdf5", "GW170104_4_template.hdf5"]]

event_fbands = [[43.0,300.0],
                [43.0,400.0],
                [43.0,800.0],
                [43.0,800.0]]


# Part a)
# 1. Estimate the power spectrum density of the data by averaging over smaller bins

# The sampling rate fs is always 4096 on all events so we'll just set it here
fs = 4096

# We'll use the first event as an example for the data cleaning
data_file_H1 = events[0][0]
data_file_L1 = events[0][1]

# Load the data
strain_H1,dt,utc = read_file(data_file_H1)
strain_L1,dt,utc = read_file(data_file_L1)

# Get the power spectrum for each strains
segment_length = 4*fs     # Set the segment size to 4 seconds of data
psd_H1 = get_average_psd(strain_H1, segment_length, fs)
psd_L1 = get_average_psd(strain_L1, segment_length, fs)

# 2. Whiten the data using the PSD
# We need to interpolate the data with all the frequencies because the averaged psd has less points
freqs = np.fft.rfftfreq(len(strain_H1), dt)
psd_freqs = np.fft.rfftfreq(len(psd_H1)*2 - 1, dt)  # *2 - 1 because we are using rfftfreq

# Create interpolations
psd_interp_H1 = np.interp(freqs, psd_freqs, psd_H1)
psd_interp_L1 = np.interp(freqs, psd_freqs, psd_L1)

# Now we can whiten the data
strain_H1_white = whiten(strain_H1, psd_interp_H1, dt)
strain_L1_white = whiten(strain_L1, psd_interp_L1, dt)

# Let's look at the whitenend data
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(strain_H1*1e19, label = 'original data')
# We slice up the beginning and end of the white data because it blows up there due to the whitening
ax2.plot(strain_H1_white[10000:119000], label = 'whitened data')
ax2.set_xlabel('time (units of 1/fs)')
ax1.set_ylabel(r'strain ($1\times10^{-19}$)')
ax2.set_ylabel('strain/rt')
ax1.legend()
ax2.legend()
ax1.set_title('Comparison of original and white data for H1 detector')
plt.show()
# As we can see, the white data has much less low frequency noise compared to the original data

# We can do the same plot for L1 and see if we get the same result
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(strain_L1*1e19, label = 'original data')
# We slice up the beginning and end of the white data because it blows up there due to the whitening
ax2.plot(strain_L1_white[10000:119000], label = 'whitened data')
ax2.set_xlabel('time (units of 1/fs)')
ax1.set_ylabel(r'strain ($1\times10^{-19}$)')
ax2.set_ylabel('strain/rt')
ax1.legend()
ax2.legend()
ax1.set_title('Comparison of original and white data for L1 detector')
plt.show()
# Yep, we get much less low frequency noise again

filename = "H-H1_LOSC_4_V2-1126259446-32.hdf5"
template_filename = "GW150914_4_template.hdf5"

strain,dt,utc = read_file(filename)
th, tl = read_template(template_filename)
n = len(strain)
window = make_flat_window(n, n//5)

NFFT = fs * 4

SNR = match_filter(strain, th, dt, fs, window, NFFT)

plt.plot(SNR)
plt.show()