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

def load_raw(file_path):
    if Path(file_path).is_file():

        temp = np.load(file_path)
        raw = temp['arr_1']
        return raw

if __name__ == '__main__':
    file_path = '../data/finger(raw).npz'
    a = load_raw(file_path)