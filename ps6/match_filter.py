import numpy as np
from power_spectrum_density import get_average_psd_complex, get_average_psd
from scipy.interpolate import interp1d
from matplotlib import mlab
import matplotlib.pyplot as plt
from whiten import whiten
from windowmaker import make_flat_window

def match_filter(strain, th, tl, dt, fs, window, NFFT):
    n = len(strain)
    time = np.linspace(0, n*dt, n)
    freqs = np.fft.fftfreq(n, dt)
    df = np.abs(freqs[0] - freqs[1])

    #template = th + tl*1j
    template = th

    template_fft = np.fft.fft(template * window) / fs

    # strain_psd = get_average_psd(strain, NFFT)
    # psd_freqs = np.fft.fftfreq(len(strain_psd), dt)
    # psd_interp = np.interp(np.abs(freqs), psd_freqs, strain_psd)

    psd_window = make_flat_window(NFFT, NFFT//5)
    NOVL = NFFT/2

    strain_psd, psd_freqs = mlab.psd(strain, Fs = fs, NFFT = NFFT, window = psd_window, noverlap = NOVL)
    psd_interp = np.interp(np.abs(freqs), psd_freqs, strain_psd)

    psd_mine = get_average_psd(strain, NFFT)

    plt.loglog(strain_psd)
    plt.loglog(psd_mine)
    plt.show()
    plt.clf()

    print(len(freqs))
    print(len(psd_freqs))

    strain_fft = np.fft.fft(strain * window) / fs

    mf_ft = strain_fft * template_fft.conj() / psd_interp
    mf = np.fft.ifft(mf_ft) * 2*fs

    sigmasq = (template_fft * template_fft.conj() / psd_interp).sum() * df
    sigma = np.sqrt(sigmasq)
    SNR_complex = mf / sigma

    peaksample = int(strain.size / 2)
    SNR_complex = np.roll(SNR_complex, peaksample)
    SNR = np.abs(SNR_complex)

    indmax = np.argmax(SNR)
    timemax = time[indmax]
    SNRmax = SNR[indmax]

    return SNR

    