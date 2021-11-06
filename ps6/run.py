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
events_to_analyze = [0, 1, 2, 3]
# 0: GW150914
# 1: LVT151012
# 2: GW151226
# 3: GW170104

event_names = ["GW150914", "LVT151012", "GW151226", "GW170104"]

# This is from the json file, but since it's just 4 events I figured I'd just copy them since I don't know how jsons work yet
events = [["H-H1_LOSC_4_V2-1126259446-32.hdf5", "L-L1_LOSC_4_V2-1126259446-32.hdf5", "GW150914_4_template.hdf5"],
          ["H-H1_LOSC_4_V2-1128678884-32.hdf5", "L-L1_LOSC_4_V2-1128678884-32.hdf5", "LVT151012_4_template.hdf5"],
          ["H-H1_LOSC_4_V2-1135136334-32.hdf5", "L-L1_LOSC_4_V2-1135136334-32.hdf5", "GW151226_4_template.hdf5"],
          ["H-H1_LOSC_4_V1-1167559920-32.hdf5", "L-L1_LOSC_4_V1-1167559920-32.hdf5", "GW170104_4_template.hdf5"]]

event_fbands = [[43.0,300.0],
                [43.0,400.0],
                [43.0,800.0],
                [43.0,800.0]]


# Part a):
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
psd_H1 = get_average_psd(strain_H1, segment_length)
psd_L1 = get_average_psd(strain_L1, segment_length)

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

# Part b):
for i in events_to_analyze:
    # Extract the data file names
    data_file_H1 = events[i][0]
    data_file_L1 = events[i][1]
    template_filename = events[i][2]
    event_name = event_names[i]
    print("Event " + event_name)

    # Read the files and template
    strain_H1,dt,utc = read_file(data_file_H1)
    strain_L1,dt,utc = read_file(data_file_L1)
    th, tl = read_template(template_filename)

    # Create a window for the match filtering
    n = len(strain_H1)
    window = make_flat_window(n, n//5)

    # Set the segment length to 4 seconds of data
    segment_length = fs * 4

    # We actually whiten the data in the match filtering here, not before
    SNR_H1, template_phased_H1 = match_filter(strain_H1, th, tl, dt, fs, window, segment_length)
    SNR_L1, template_phased_L1 = match_filter(strain_L1, th, tl, dt, fs, window, segment_length)

    # Produce the time array given the dt and the number of data points
    time = np.linspace(0, n*dt, n)

    # Plot both SNR
    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.plot(time, SNR_H1, label = "H1 detector")
    ax1.set_ylabel("SNR")
    ax1.set_title("SNR for event " + event_name)
    ax1.legend()

    ax2.plot(time, SNR_L1, label = "L1 detector")
    ax2.set_ylabel("SNR")
    ax2.set_xlabel("Time (s)")
    ax2.legend()

    plt.show()

    # Extract the time at the peaks
    max_index_H1 = np.argmax(SNR_H1)
    max_index_L1 = np.argmax(SNR_L1)

    peak_time_H1 = time[max_index_H1]
    peak_time_L1 = time[max_index_L1]

    SNR_max_H1 = np.max(SNR_H1)
    SNR_max_L1 = np.max(SNR_L1)
    print("SNR for H1 at event =", SNR_max_H1)
    print("SNR for L1 at event =", SNR_max_L1)

    # Shift the templates so they line up at the correct times
    shifted_template_H1 = np.roll(template_phased_H1, max_index_H1 - n // 2)
    shifted_template_L1 = np.roll(template_phased_L1, max_index_L1 - n // 2)

    # Whiten the template and the data
    psd_H1 = get_average_psd(strain_H1, segment_length)
    psd_L1 = get_average_psd(strain_L1, segment_length)

    freqs = np.fft.rfftfreq(len(strain_H1), dt)
    psd_freqs = np.fft.rfftfreq(len(psd_H1)*2 - 1, dt)  # *2 - 1 because we are using rfftfreq

    psd_interp_H1 = np.interp(freqs, psd_freqs, psd_H1)
    psd_interp_L1 = np.interp(freqs, psd_freqs, psd_L1)

    strain_H1_white = whiten(strain_H1, psd_interp_H1, dt)
    strain_L1_white = whiten(strain_L1, psd_interp_L1, dt)

    template_white_H1 = whiten(shifted_template_H1, psd_interp_H1, dt)
    template_white_L1 = whiten(shifted_template_L1, psd_interp_L1, dt)

    # We could probably also try a band-pass filter to remove more noise
    # This is the simplest band-pass filter where I just completely suppress the modes
    # outside the frequency range
    # Taking the FTs
    H1_FT = np.fft.rfft(strain_H1_white)
    H1_template_FT = np.fft.rfft(template_white_H1)
    L1_FT = np.fft.rfft(strain_L1_white)
    L1_template_FT = np.fft.rfft(template_white_L1)

    # Getting the frequencies and the cutoff + indices
    freqs_filter = np.fft.rfftfreq(len(H1_FT), dt)
    cutoff_freq_max = event_fbands[i][1]
    cutoff_freq_min = event_fbands[i][0]
    indices_max = np.where(freqs_filter > cutoff_freq_max)[0]
    indices_min = np.where(freqs_filter < cutoff_freq_min)[0]

    # Cutting off the higher and lower frequencies
    H1_FT[indices_max] = 0
    H1_FT[indices_min] = 0
    H1_template_FT[indices_max] = 0
    H1_template_FT[indices_min] = 0

    L1_FT[indices_max] = 0
    L1_FT[indices_min] = 0
    L1_template_FT[indices_max] = 0
    L1_template_FT[indices_min] = 0

    # Reconverting
    strain_H1_white_filter = np.fft.irfft(H1_FT)
    template_white_filter_H1 = np.fft.irfft(H1_template_FT)
    strain_L1_white_filter = np.fft.irfft(L1_FT)
    template_white_filter_L1 = np.fft.irfft(L1_template_FT)

    # Setting a plotting range around the event
    start_H1 = int(max_index_H1 - 0.05*fs)
    stop_H1 = int(max_index_H1 + 0.05*fs)
    
    start_L1 = int(max_index_L1 - 0.05*fs)
    stop_L1 = int(max_index_L1 + 0.05*fs)

    # Plotting the 'fits' for both detectors
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(time[start_H1:stop_H1], strain_H1_white_filter[start_H1:stop_H1])
    ax1.plot(time[start_H1:stop_H1], template_white_filter_H1[start_H1:stop_H1])
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Event " + event_name)
    
    ax2.plot(time[start_L1:stop_L1], strain_L1_white_filter[start_L1:stop_L1])
    ax2.plot(time[start_L1:stop_L1], template_white_filter_L1[start_L1:stop_L1])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")

    plt.show()

    # So as we can see, the events match up pretty well with the expected template,
    # which is a good sign

    print()