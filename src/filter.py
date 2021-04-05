import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal
import numpy as np
from scipy.signal import welch, resample
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
    )
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

    fs = 250

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
    total = np.logical_and(freqs >= 1, freqs <= 50)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd[total], dx=freq_res)
    return bp

def filter_band(data, fs, low, high):

    nyq = 0.5*fs
    order = 2
    low = low / nyq
    high = high / nyq

    b, a = signal.iirnotch(50, 30, fs)
    y_notch = signal.filtfilt(b, a, data)

    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    y_band = signal.filtfilt(b, a, y_notch)

    return y_band

def filter_low(data, fs):

    nyq = 0.5*fs
    cutoff = 2
    order = 2
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_low = signal.filtfilt(b, a, data)

    return y_low


def filter_raw(dataset):

    data = {}
    dataset = dataset[['FT7', 'FT8', 'T7', 'P1', 'P2', 'T8', 'O2', 'O1', 'label']]

    for c in dataset.columns:
        if c != 'label':

            print("-----------{}-----------".format(c))
            secs = dataset[c].shape[0] / 400
            samps = int(secs * 250)
            print(samps)
            signal_resample = resample(dataset[c].to_numpy(), samps)
            data[c+'_delta'] = pd.Series(filter_band(signal_resample, 250, 1, 4))
            data[c + '_theta'] = pd.Series(filter_band(signal_resample, 250, 4, 8))
            data[c + '_alpha'] = pd.Series(filter_band(signal_resample, 250, 8, 14))
            data[c + '_beta'] = pd.Series(filter_band(signal_resample, 250, 14, 31))
            data[c + '_gamma'] = pd.Series(filter_band(signal_resample, 250, 31, 50))


        else:
            data[c] = []
            for i in range(round(dataset[c].shape[0]/1600)):
                for i in range(1000):
                    data[c].append(dataset[c][i*1600])


    df = pd.DataFrame(data)
    df.to_csv("../../Database/SEED-VIG/filterRaw.csv", sep=";", index=False)




def psd_raw(dataset):

    data = {}
    label = {}
    len_data = round(dataset.shape[0]/800)
    print(len(dataset['label']))

    for c in dataset.columns:
        if c != 'label':
            data[c] = []
            for i in range(0, len_data):
                sig = filter(dataset[c][i*800:(i+1)*800])
                freq, psd = signal.welch(sig)
                print(len(psd))
                data[c] = [*data[c], *psd]

            print("-----------{}-----------".format(c))

        else:
            label[c] = []
            for i in range(0, len_data):
                label[c].append(dataset[c][i * 800])


def process(dataset):

    data = {}
    dataset = dataset[['FT7','FT8','T7','P1','P2','T8','O2','O1','label']]

    sleep_data = dataset[dataset['label'] == 2]

    noise_data = sleep_data + np.random.normal(0, .1, sleep_data.shape)
    noise_data['label'] = 2
    dataset = pd.concat([dataset, noise_data], ignore_index=True)

    awake_data = dataset[dataset['label'] == 0]
    noise_data = awake_data + np.random.normal(0, .1, awake_data.shape)
    noise_data['label'] = 0
    dataset = pd.concat([dataset, noise_data], ignore_index=True)

    for c in dataset.columns:
        if c != 'label':

            print("-----------{}-----------".format(c))

            data[c + '_psd_theta_ma'] = []
            data[c + '_se_theta_ma'] = []

            data[c + '_psd_alpha_ma'] = []
            data[c + '_se_alpha_ma'] = []

            data[c + '_psd_gamma_ma'] = []
            data[c + '_se_gamma_ma'] = []

            data[c + '_psd_delta_ma'] = []
            data[c + '_se_delta_ma'] = []

            data[c + '_psd_beta_ma'] = []
            data[c + '_se_beta_ma'] = []

            data[c + '_psd_theta_relative'] = []
            data[c + '_psd_alpha_relative'] = []
            data[c + '_psd_gamma_relative'] = []
            data[c + '_psd_delta_relative'] = []
            data[c + '_psd_beta_relative'] = []

            len_data = round(len(dataset[c].to_numpy()) / 1600)
            secs = dataset[c].shape[0] / 400
            samps = int(secs * 250)
            print(samps)
            signal_resample = resample(dataset[c].to_numpy(), samps)
            signal = pd.Series(filter_band(signal_resample, 250, 1, 50))
            signal_ma = signal.rolling(window=3).mean()
            signal_ma[0] = signal[0]
            signal_ma[1] = signal[1]
            signal_ma = signal_ma.to_numpy()

            for i in range(0, len_data):

                data[c + '_psd_delta_ma'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [0.5, 4], 'welch', None))
                data[c + '_se_delta_ma'].append(spectral_entropy(signal_ma[i * 1000:(i + 1) * 1000], range(4, 8), 250))

                data[c + '_psd_theta_ma'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [4, 8], 'welch', None))
                data[c+'_se_theta_ma'].append(spectral_entropy(signal_ma[i * 1000:(i + 1) * 1000], range(4, 8), 250))

                data[c+'_psd_alpha_ma'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [8, 14], 'welch', None))
                data[c+'_se_alpha_ma'].append(spectral_entropy(signal_ma[i * 1000:(i + 1) * 1000], range(8, 14), 250))

                data[c+'_psd_beta_ma'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [14, 30], 'welch', None))
                data[c+'_se_beta_ma'].append(spectral_entropy(signal_ma[i * 1000:(i + 1) * 1000], range(14, 30), 250))

                data[c + '_psd_gamma_ma'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [30, 50], 'welch', None))
                data[c + '_se_gamma_ma'].append(spectral_entropy(signal_ma[i * 1000:(i + 1) * 1000], range(30, 50), 250))

                data[c + '_psd_delta_relative'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [0.5, 4], 'welch', None, relative=True))
                data[c + '_psd_theta_relative'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [4, 8], 'welch', None, relative=True))
                data[c + '_psd_alpha_relative'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [8, 14], 'welch', None, relative=True))
                data[c + '_psd_beta_relative'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [14, 30], 'welch', None, relative=True))
                data[c + '_psd_gamma_relative'].append(bandpower(signal_ma[i * 1000:(i + 1) * 1000], [30, 50], 'welch', None, relative=True))



        else:
            data[c] = []
            for i in range(0,len_data):
                data[c].append(dataset[c][i*1600])


    df = pd.DataFrame(data)
    print(df)
    df.to_csv("../../Database/SEED-VIG/dataset.csv", sep=";", index=False)


def process_bpci_data(dataset):

    data = {}
    print(dataset)

    data['FT7'] = dataset['Channel1']
    data['FT8'] = dataset['Channel2']
    data['T7'] = dataset['Channel3']
    data['CP1'] = dataset['Channel4']
    data['CP2'] = dataset['Channel5']
    data['T8'] = dataset['Channel6']
    data['O2'] = dataset['Channel7']
    data['O1'] = dataset['Channel8']

    for k in data.keys():
        secs = data[k].shape[0] / 250
        samps = int(secs * 400)
        signal_resample = resample(data[k].to_numpy(), samps)
        signal = pd.Series(filter(signal_resample,400))
        data[k] = signal

    df = pd.DataFrame(data)
    print(df)
    df.head(1600*20).to_json("../../Database/SEED-VIG/test.json")

    df.to_csv("../../Database/SEED-VIG/test.csv", sep=";", index=False)


def filter_api(dict):

    secs = dict.shape[0] / 250
    n_sample = int(secs * 400)
    filter_dict = {}

    for c in dict.keys():

        signal_resample = resample(dict[c].to_numpy(), n_sample)
        filter_dict[c] = pd.Series(filter(signal_resample, 400))

    dataset = pd.DataFrame(filter_dict)
    return dataset
