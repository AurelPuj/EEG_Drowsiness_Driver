import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
import numpy as np
from scipy.integrate import simps
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import pandas as pd

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
    #plt.plot(y_low)

    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    y_band = signal.filtfilt(b, a, y_low)
    #plt.plot(y_band)
    #plt.show()
    return y_band



def process(dataset):
    dict = {}
    len_data = round(dataset.shape[0]/1600)
    count = 0
    for c in dataset.columns:
        if c != 'label':
            print("{}".format(c))

            dict[c+'_delta'] = []
            dict[c+'_theta'] = []
            dict[c+'_alpha'] = []
            dict[c+'_beta'] = []
            dict[c+'_gamma'] = []

            for i in range(0,len_data):
                signal = filter(dataset[c][i*1600:(i+1)*1600])
                dict[c+'_delta'].append(bandpower(signal, [0.5, 4], 'welch'))
                dict[c + '_theta'].append(bandpower(signal, [4, 8], 'welch'))
                dict[c+'_alpha'].append(bandpower(signal, [8, 14], 'welch'))
                dict[c+'_beta'].append(bandpower(signal, [14, 31], 'welch'))
                dict[c+'_gamma'].append(bandpower(signal, [31, 50], 'welch'))

                if bandpower(signal, [0.5, 4], 'welch') < bandpower(signal, [4, 8], 'welch'):
                    count += 1
            print("{} number of sleep : {}".format(c,count))
            count=0
        else:
            dict[c]=[]
            for i in range(0,len_data):
                dict[c].append(dataset[c][i*1600])
            print("size of label : {}".format(len(dict[c])))

        df = pd.DataFrame(dict)
        df.to_csv("../../Database/SEED-VIG/psdRaw.csv", sep=";", index=False)
        #plt.bar(['delta', 'theta', 'alpha', 'beta', 'gamma'], [delta, theta, alpha, beta, gamma])
        #plt.show()



