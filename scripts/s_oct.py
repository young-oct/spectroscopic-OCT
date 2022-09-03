# -*- coding: utf-8 -*-
# @Time    : 2022-09-02 23:00
# @Author  : young wang
# @FileName: s_oct.py
# @Software: PyCharm
import pickle
from pathlib import Path
from scipy import ndimage
from skimage.morphology import disk,square,star,diamond,octagon
from skimage.morphology import dilation, erosion
from skimage import filters
from scipy.signal import find_peaks
from skimage import feature
from scipy.ndimage import gaussian_filter
from numpy.fft import fft, fftshift, ifft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib

def Aline_R(data,start):
    A_line = ifft(data, axis=1)
    return A_line[dwell * start:dwell * (start + 512), -350:-20].T

def load_raw(file_path):
    if Path(file_path).is_file():

        temp = np.load(file_path)
        raw = temp['arr_1']
        return raw

def mean_remove(s,decimation_factor):
    s = s - np.mean(s, axis=1)[:, np.newaxis]
    # (3) sample every decimation_factor line,
    s = s[:, ::decimation_factor]
    return s

dwell = 20


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 16,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    file_path = '../data/finger(raw).npz'
    raw = load_raw(file_path)
    start, decimation_factor = 420, 20
    rvmin, vmax = 15, 55  # dB

    s_r = mean_remove(Aline_R(raw, start), decimation_factor)
    sr_log = 20 * np.log10(abs(s_r))

    fig = plt.figure(figsize=(16, 9),constrained_layout=True)

    gs = fig.add_gridspec(ncols=2, nrows=1)
    ax = fig.add_subplot(gs[0])
    ax.set_title('(a) standard B-mode')

    ax.imshow(sr_log, 'gray',
              aspect=sr_log.shape[1] / sr_log.shape[0],
              vmin=rvmin, vmax=vmax, interpolation='none')
    plt.show()
