import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
import numpy as np
from scipy.integrate import simps
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import pandas as pd
import numpy


def bin_power(X, Band, Fs):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.

    Note
    -----
    A real signal can be synthesized, thus not real.

    Parameters
    -----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    Power
        list

        spectral power in each frequency bin.

    Power_ratio
        list

        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.

    """

    C = numpy.fft.fft(X)
    C = abs(C)
    Power = numpy.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(numpy.floor(Freq / Fs * len(X))):
                int(numpy.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio


def spectral_entropy(X, Band, Fs, Power_Ratio=None):
    """Compute spectral entropy of a time series from either two cases below:
    1. X, the time series (default)
    2. Power_Ratio, a list of normalized signal power in a set of frequency
    bins defined in Band (if Power_Ratio is provided, recommended to speed up)

    In case 1, Power_Ratio is computed by bin_power() function.

    Notes
    -----
    To speed up, it is recommended to compute Power_Ratio before calling this
    function because it may also be used by other functions whereas computing
    it here again will slow down.

    Parameters
    ----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    As indicated in return line

    See Also
    --------
    bin_power: pyeeg function that computes spectral power in frequency bins

    """

    if Power_Ratio is None:
        Power, Power_Ratio = bin_power(X, Band, Fs)

    Spectral_Entropy = 0
    for i in range(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * numpy.log(Power_Ratio[i])
    Spectral_Entropy /= numpy.log(
        len(Power_Ratio)
    )  # to save time, minus one is omitted
    return -1 * Spectral_Entropy


def bandpower(data, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    fs : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    fs = 400

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * fs
        else:
            nperseg = (2 / low) * fs

        freqs, psd = welch(data, fs, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, fs, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def filter(data):

    '''
    plt.figure()
    plt.plot(data)
    '''

    fs = 400
    nyq = 0.5*fs
    cutoff = 2
    order = 2
    low = 1 / nyq
    high = 75 / nyq
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_low = signal.filtfilt(b, a, data)
    '''
    plt.plot(y_low)

    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    y_band = signal.filtfilt(b, a, y_low)
    plt.plot(y_band)
    plt.show()

    '''

    return y_low


def process(dataset):

    data = {}
    len_data = round(dataset.shape[0]/1600)

    for c in dataset.columns:
        if c != 'label':

            print("-----------{}-----------".format(c))
            data[c+'_psd_delta'] = []
            data[c+'_se_delta'] = []

            data[c+'_psd_theta'] = []
            data[c+'_se_theta'] = []

            data[c+'_psd_alpha'] = []
            data[c+'_se_alpha'] = []

            data[c+'_psd_beta'] = []
            data[c+'_se_beta'] = []

            data[c+'_psd_gamma'] = []
            data[c+'_se_gamma'] = []


            for i in range(0,len_data):

                signal = filter(dataset[c][i*1600:(i+1)*1600])
                data[c+'_psd_delta'].append(bandpower(signal, [0.5, 4], 'welch'))
                data[c+'_se_delta'].append(spectral_entropy(signal, range(1, 4), 400))

                data[c + '_psd_theta'].append(bandpower(signal, [4, 8], 'welch'))
                data[c+'_se_theta'].append(spectral_entropy(signal, range(4, 8), 400))

                data[c+'_psd_alpha'].append(bandpower(signal, [8, 14], 'welch'))
                data[c+'_se_alpha'].append(spectral_entropy(signal, range(8, 14), 400))

                data[c+'_psd_beta'].append(bandpower(signal, [14, 31], 'welch'))
                data[c+'_se_beta'].append(spectral_entropy(signal, range(14, 31), 400))

                data[c+'_psd_gamma'].append(bandpower(signal, [31, 50], 'welch'))
                data[c+'_se_gamma'].append(spectral_entropy(signal, range(31, 50), 400))

        else:
            data[c] = []
            for i in range(0,len_data):
                data[c].append(dataset[c][i*1600])

        df = pd.DataFrame(data)
        df.to_csv("../../Database/SEED-VIG/psdDeRaw.csv", sep=";", index=False)



