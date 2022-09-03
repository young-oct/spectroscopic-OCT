# -*- coding: utf-8 -*-
# @Time    : 2022-09-02 23:00
# @Author  : young wang
# @FileName: s_oct.py
# @Software: PyCharm
import pickle
from pathlib import Path
from scipy import ndimage
from skimage.morphology import disk, square, star, diamond, octagon
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
from sklearn import preprocessing


def range_convert(img, target_type_min, target_type_max, ):
    """
    converts image range from input range to a given range for
    standardized comparison
    :param img: 2D array input
    :return: 2D array input in given range
    """
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(img.dtype)
    return new_img


def Aline(data, decimation_factor):
    """

    :param data: 2D array of processed interferogram
    :param decimation_factor: average number of A-lines to improve SNR
    :return: 2D array of A-lines
    """
    if data.shape[1] > data.shape[0]:
        pass
    else:
        np.swapaxes(data, 0, 1)

    A_line = ifft(data, axis=0)

    return A_line[:, ::decimation_factor]


def data_loader(file_path, cross = True, norm = True):
    """
    load raw OCT interferogram
    :input file_path:
    :return: OCT interferogram array
    in the format of [axial length, lateral length]
    """
    if Path(file_path).is_file():
        raw = np.load(file_path)

    # swap axis of the array to
    # [axial length, lateral length]
    raw_proc = np.swapaxes(raw['arr_1'], 0, 1)

    if cross:
        raw_proc = corr_term(raw_proc, norm = norm)
    else:
        pass
    return raw_proc


def corr_term(raw, norm = True):
    """
    remove the DC term of the raw interferogram
    :param norm: normalization flag
    :param raw: 2D array of raw interferogram
    :return: 2D array of processed interferogram
    """
    if raw.shape[1] > raw.shape[0]:
        pass
    else:
        np.swapaxes(raw, 0, 1)

    # obtain the DC term
    raw_proc = raw - np.mean(raw, axis=1)[:, np.newaxis]

    if norm:
        raw_proc = preprocessing.normalize(raw_proc, norm='l2', axis=1)
    else:
        pass

    return raw_proc

def Aline2Bmode(alines, start =420):
    roi = alines[-350:-20,start:int(start+512)]

    img_log = 20 * np.log10(abs(roi))

    return range_convert(img_log,0,255)



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
    raw = data_loader(file_path)
    bmode = Aline2Bmode(Aline(raw,decimation_factor=20))

    p_factor = 0.65
    dvmin, dvmax = np.max(bmode) * p_factor, np.max(bmode)
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)

    gs = fig.add_gridspec(ncols=2, nrows=1)
    ax = fig.add_subplot(gs[0])
    ax.set_title('(a) standard B-mode')

    ax.imshow(bmode, 'gray',
              aspect=bmode.shape[1] / bmode.shape[0],
              vmin=dvmin, vmax=dvmax, interpolation='none')
    plt.show()
