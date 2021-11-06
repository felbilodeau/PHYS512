import numpy as np
from power_spectrum_density import get_average_psd      # Turns out my own function didn't work but this is where you'll find it
from matplotlib import mlab
from windowmaker import make_flat_window

# Function which performs match filtering on a strain time-series with a template
def match_filter(strain, template, dt, fs, window, NFFT):
    # Get the length of the strain and extract the freauencies and df we will have in the Fourier transform
    n = len(strain)
    freqs = np.fft.fftfreq(n, dt)
    df = np.abs(freqs[0] - freqs[1])

    # Calculate the FT of the template using the window, and normalized by the sampling rate fs
    template_fft = np.fft.fft(template * window) / fs

    # Okay so I wrote my own psd averaging code, seen used below in comments, but it doesn't seem to work
    # for some reason, the SNR is all over the place without a defined peak. I mean I'm sure it's because 
    # of the overlap thing, but I can't get that to work in my code, so I mean I guess
    # you can look at it but I have to use the matplotlib psd function to get meaningful results.

    # strain_psd = get_average_psd(strain, NFFT, fs)
    # psd_freqs = np.fft.fftfreq(len(strain_psd), dt)
    # psd_interp = np.interp(np.abs(freqs), psd_freqs, strain_psd)

    # Create a window for the psd
    psd_window = make_flat_window(NFFT, NFFT//5)

    # Define the overlap to be half the segment size
    noverlap = NFFT/2

    # Get the averaged psd using the matplotlib function
    strain_psd, psd_freqs = mlab.psd(strain, Fs = fs, NFFT = NFFT, window = psd_window, noverlap = noverlap)

    # Interpolate between the reduced data points in the psd to have n data points
    psd_interp = np.interp(np.abs(freqs), psd_freqs, strain_psd)

    # Calculate the FT of the strain using the same window as the template, and also scale it by the sampling rate fs
    strain_fft = np.fft.fft(strain * window) / fs

    # Perform the match filtering and invert the FT to get the time of the event
    mf_ft = strain_fft * template_fft.conj() / psd_interp
    mf = np.fft.ifft(mf_ft) * 2*fs

    # Rescale the match filter so we have the SNR
    variance = (template_fft * template_fft.conj() / psd_interp).sum() * df
    sigma = np.sqrt(variance)
    SNR_complex = mf / sigma

    # Roll the SNR data so that the peak is where the event ends instead of where it begins
    peaksample = int(strain.size / 2)
    SNR_complex = np.roll(SNR_complex, peaksample)

    # Take the absolute value and return
    SNR = np.abs(SNR_complex)
    return SNR