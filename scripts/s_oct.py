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

        img_list = [range_convert(ROI(bmode,start), 0, 255, np.float64),
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
