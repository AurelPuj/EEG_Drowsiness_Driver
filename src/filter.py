import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
import numpy as np
from scipy.integrate import simps

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps

    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)
    print(psd)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def process(dataset):
    for c in dataset.columns:
        plt.figure()
        print("----------{}----------".format(c))
        data = dataset[c]
        plt.plot(data)

        plt.figure()
        data = butter_lowpass_filter(data, 200, 2)
        plt.plot(data)

        plt.figure()
        data = butter_bandpass_filter(data, 1, 75, 2)
        plt.plot(data)
        plt.show()
        '''
        win_sec = 4
        sf = 200
        # Delta/beta ratio based on the absolute power
        db = bandpower(data, sf, [0.5, 4], win_sec)
        print(" Band Power Delta : {}".format(db))
        db = bandpower(data, sf, [4, 8], win_sec)
        print(" Band Power Theta : {}".format(db))

        db = bandpower(data, sf, [8, 14], win_sec)
        print(" Band Power Alpha : {}".format(db))

        db = bandpower(data, sf, [14, 31], win_sec)
        print(" Band Power Beta : {}".format(db))

        db = bandpower(data, sf, [31, 50], win_sec)
        print(" Band Power Gamma : {}".format(db))
'''