import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from matplotlib.ticker import ScalarFormatter

# Function to return power spectrum for mean or max
def powerspectrum(patch, t_step, method="welch"):
    if len(patch)%2:
        patch = patch[:-1]
    if method=="welch":
        freq, Pxx_den = scipy.signal.welch(patch, fs = 1/t_step)
    elif method=="power":
        freq, Pxx_den = scipy.signal.periodogram(patch,fs=1/t_step, scaling="density")
    return(freq, Pxx_den)

def plotspectra(dir, metric=3, norm=True):
    '''
    dir: Relative path to directory. Start with "../measurements/"
    metric: Use 1 for MEAN. Use 3 for MAX.
    norm: normalize each power spectrum so that total power is 1? 
    Else allow differences in total power.
    '''
    all_den = []
    fig, ax = plt.subplots()
    norm = True
    tstep = 5
    for fname in os.listdir(dir):
        if ".csv" in fname:
            print(fname)
            meas = np.genfromtxt(dir+fname, delimiter=',')
            patch = meas[1:,metric] # first row is labels
            freq, Pxx_den = powerspectrum(patch, tstep)
            period = 1/freq
            if norm:
                Pxx_den = Pxx_den/np.mean(Pxx_den)
            all_den.append(Pxx_den)
            plt.plot(period, Pxx_den, alpha=0.5)
    plt.plot(period, np.mean(all_den, axis=0), linewidth=3, color="black")
    plt.title("Spectral density of polarity patch oscillations (Welch's method)")
    plt.xlabel("Period (seconds)")
    plt.ylabel("Power density")
    plt.xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    #plt.ylim(0,5)
    ax.invert_xaxis()
    return plt.show()


dir = "../measurements/2024_04_30/"
plotspectra(dir)

# Write data as line in csv
