# -*- coding: utf-8 -*-
# @Time    : 2022-09-04 17:57
# @Author  : young wang
# @FileName: utilities.py
# @Software: PyCharm

from pathlib import Path
from numpy.fft import fft, fftshift, ifft
import numpy as np
from sklearn import preprocessing
import cv2

def range_convert(img, target_type_min, target_type_max, target_type):
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
    new_img = (a * img + b).astype(target_type)
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


def data_loader(file_path, cross=True, norm=True):
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
    raw_proc = np.swapaxes(raw['arr_0'], 0, 1)

    if cross:
        raw_proc = corr_term(raw_proc, norm=norm)
    else:
        pass
    return raw_proc


def corr_term(raw, norm=True):
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


def Aline2Bmode(alines, range=True):
    img_log = 20 * np.log10(abs(alines))
    if range:
        img_log = range_convert(img_log,
                                target_type_min = 0,
                                target_type_max= 255,
                                target_type= img_log.dtype)
    else:
        pass
    return img_log


def formHSV(spe_img, b_img, sp_thres_low, sp_thres_high,
            st_thres=0.65, saturation=200,start = 100):
    sp_inf = range_convert(spe_img, 0, 1, target_type=np.float64)
    st_inf = range_convert(b_img, 0, 1, target_type=np.float64)

    sp_inf = np.clip(sp_inf, a_min=sp_thres_low, a_max=sp_thres_high)
    st_inf = np.clip(st_inf, a_min=st_thres, a_max=1)

    # convert 3D array into opencv styple HSV image
    img_h = range_convert(sp_inf, 0, 180, target_type=np.uint8)
    img_v = range_convert(st_inf, 0, 255, target_type=np.uint8)
    img_s = np.full((img_v.shape), saturation, dtype=np.uint8)

    temp_img = cv2.merge((ROI(img_h, start=start),
                          ROI(img_v, start=start),
                          ROI(img_s, start=start)))

    out = cv2.cvtColor(temp_img, cv2.COLOR_HSV2RGB)

    return cv2.cvtColor(out, cv2.COLOR_RGB2HSV)


def ROI(image, start=100):
    return image[-350:-20, start: int(start + 512)]

