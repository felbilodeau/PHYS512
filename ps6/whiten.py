import numpy as np

# Whitens the data given a function interpolated_psd to interpolate a smoothed psd
def whiten(strain, interpolated_psd, dt):
    # Get the number of data points and the frequencies associated with them
    n = len(strain)
    freqs = np.fft.rfftfreq(n, dt)
    
    # Calculate the FT of the data
    strain_ft = np.fft.rfft(strain)

    # We need to make sure we normalize this properly with 
    norm = np.sqrt(2*dt)

    # Then we calculate the whitened FT
    white_data_FT = strain_ft / np.sqrt(interpolated_psd(freqs)) * norm

    # Finally, reconvert the FT to get the whitened data and return
    white_data = np.fft.irfft(white_data_FT, n)
    return white_data