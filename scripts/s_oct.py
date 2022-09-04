# -*- coding: utf-8 -*-
# @Time    : 2022-09-02 23:00
# @Author  : young wang
# @FileName: s_oct.py
# @Software: PyCharm

from pathlib import Path
from numpy.fft import fft, fftshift, ifft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import cv2
import glob
from natsort import natsorted
from utilities.utilities import range_convert,Aline,data_loader,Aline2Bmode,formHSV,ROI

#
# def range_convert(img, target_type_min, target_type_max, target_type):
#     """
#     converts image range from input range to a given range for
#     standardized comparison
#     :param img: 2D array input
#     :return: 2D array input in given range
#     """
#     imin = img.min()
#     imax = img.max()
#
#     a = (target_type_max - target_type_min) / (imax - imin)
#     b = target_type_max - a * imax
#     new_img = (a * img + b).astype(target_type)
#     return new_img
#
#
# def Aline(data, decimation_factor):
#     """
#
#     :param data: 2D array of processed interferogram
#     :param decimation_factor: average number of A-lines to improve SNR
#     :return: 2D array of A-lines
#     """
#     if data.shape[1] > data.shape[0]:
#         pass
#     else:
#         np.swapaxes(data, 0, 1)
#
#     A_line = ifft(data, axis=0)
#
#     return A_line[:, ::decimation_factor]
#
#
# def data_loader(file_path, cross=True, norm=True):
#     """
#     load raw OCT interferogram
#     :input file_path:
#     :return: OCT interferogram array
#     in the format of [axial length, lateral length]
#     """
#     if Path(file_path).is_file():
#         raw = np.load(file_path)
#
#     # swap axis of the array to
#     # [axial length, lateral length]
#     raw_proc = np.swapaxes(raw['arr_0'], 0, 1)
#
#     if cross:
#         raw_proc = corr_term(raw_proc, norm=norm)
#     else:
#         pass
#     return raw_proc
#
#
# def corr_term(raw, norm=True):
#     """
#     remove the DC term of the raw interferogram
#     :param norm: normalization flag
#     :param raw: 2D array of raw interferogram
#     :return: 2D array of processed interferogram
#     """
#     if raw.shape[1] > raw.shape[0]:
#         pass
#     else:
#         np.swapaxes(raw, 0, 1)
#
#     # obtain the DC term
#     raw_proc = raw - np.mean(raw, axis=1)[:, np.newaxis]
#
#     if norm:
#         raw_proc = preprocessing.normalize(raw_proc, norm='l2', axis=1)
#     else:
#         pass
#
#     return raw_proc
#
#
# def Aline2Bmode(alines, range=True):
#     img_log = 20 * np.log10(abs(alines))
#     if range:
#         img_log = range_convert(img_log,
#                                 target_type_min = 0,
#                                 target_type_max= 255,
#                                 target_type= img_log.dtype)
#     else:
#         pass
#     return img_log
#
#
# def formHSV(spe_img, b_img, sp_thres_low, sp_thres_high,
#             st_thres=0.65, saturation=200,start = 100):
#     sp_inf = range_convert(spe_img, 0, 1, target_type=np.float64)
#     st_inf = range_convert(b_img, 0, 1, target_type=np.float64)
#
#     sp_inf = np.clip(sp_inf, a_min=sp_thres_low, a_max=sp_thres_high)
#     st_inf = np.clip(st_inf, a_min=st_thres, a_max=1)
#
#     # convert 3D array into opencv styple HSV image
#     img_h = range_convert(sp_inf, 0, 180, target_type=np.uint8)
#     img_v = range_convert(st_inf, 0, 255, target_type=np.uint8)
#     img_s = np.full((img_v.shape), saturation, dtype=np.uint8)
#
#     temp_img = cv2.merge((ROI(img_h, start=start),
#                           ROI(img_v, start=start),
#                           ROI(img_s, start=start)))
#
#     out = cv2.cvtColor(temp_img, cv2.COLOR_HSV2RGB)
#
#     return cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
#
#
# def ROI(image, start=100):
#     return image[-350:-20, start: int(start + 512)]


if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 16,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_list = natsorted(glob.glob('../data/*.npz'))

    for i in range(len(data_list)):
        if i == 0:
            start = 100
        else:
            start = 100
        raw = data_loader(data_list[i])
        bmode = Aline2Bmode(Aline(raw, decimation_factor=20), range=True)

        length = raw.shape[0]

        band_no = 2
        hann = np.hanning(1.05 * length / band_no)
        band1 = np.pad(hann, (0, length - len(hann) % length), 'constant')
        band2 = np.pad(hann, (length - len(hann) % length, 0), 'constant')

        band1 = band1[:, np.newaxis]
        band2 = band2[:, np.newaxis]

        bands = [band1, band2]

        spectral_arr = np.zeros((raw.shape[0], raw.shape[1], 2))
        for i in range(spectral_arr.shape[-1]):
            temp = np.copy(raw)
            spectral_arr[:, :, i] = temp * bands[i]

        spc_log = np.zeros((raw.shape[0], int(raw.shape[1] / 20), 2))
        for i in range(spc_log.shape[-1]):
            spc_log[:, :, i] = Aline2Bmode(Aline(spectral_arr[:, :, i],
                                                 decimation_factor=20),
                                           range=True)

        spe_contrast = np.diff(spc_log, axis=2).squeeze()
        spe_contrast_norm = spe_contrast / np.clip(bmode,
                                                   a_min= 0.05*np.max(bmode),
                                                   a_max= np.max(bmode))

        v_img = np.mean(spc_log, axis=2)

        hsv_img = formHSV(spe_contrast_norm, v_img,
                          sp_thres_low=0.01,
                          sp_thres_high=1,
                          st_thres=0.65,
                          saturation=25,
                          start = start)

        img_list = [range_convert(ROI(bmode,start ), 0, 255, np.float64),
                    range_convert(ROI(spe_contrast_norm,start), 0, 255, np.float64), hsv_img]
        title_list = ['conventional OCT image',
                      'spectral signal image',
                      'spectroscopic OCT']

        fig, axs = plt.subplots(1, int(len(img_list)), figsize=(16, 9))
        for n, (ax, img, title) in enumerate(zip(axs.flat,
                                                 img_list, title_list)):

            if n == 0:
                ax.imshow(img, 'gray', vmin=0.65 * np.max(img), vmax=np.max(img))
            elif n == 1:
                ax.imshow(img, 'hot', vmin=0.1 * np.max(img), vmax=np.max(img))
            else:
                ax.imshow(img)

            ax.set_title(title)
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()
