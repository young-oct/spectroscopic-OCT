# -*- coding: utf-8 -*-
# @Time    : 2022-09-06 14:39
# @Author  : young wang
# @FileName: stft_method.py
# @Software: PyCharm

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from scipy import signal
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
    images = []
    titles = []

    for i in range(len(data_list)):
        if i == 0:
            start = 100
        else:
            start = 100
        raw = data_loader(data_list[i])
        bmode = Aline2Bmode(Aline(raw, decimation_factor=20), range=True)

        img_contrast = np.zeros_like(raw)
        for j in range(raw.shape[-1]):
            line_raw = raw[:, j]
            _, _, banded_signal = signal.stft(line_raw,
                                              nperseg=int(len(line_raw))/3,
                                              nfft = len(line_raw)*2-1)
            energy = [np.sum(abs(banded_signal[:,i]) ** 2) for i in range(banded_signal.shape[-1])]
            img_contrast[:,j] = signal.resample(energy, raw.shape[0])

        decimation_factor = 20
        start = 100
        cor_log = img_contrast[:, ::decimation_factor]
        cor_log = Aline2Bmode(cor_log)

        v_img = Aline2Bmode(Aline(raw, decimation_factor=20), range=True)

        hsv_img = formHSV(cor_log, v_img,
                          sp_thres_low=0.01,
                          sp_thres_high=1,
                          st_thres=0.65,
                          saturation=25,
                          start=start)

        img_list = [range_convert(ROI(v_img, start), 0, 255, np.float64),
                    range_convert(ROI(cor_log, start), 0, 255, np.float64), hsv_img]

        title_list = ['conventional OCT image',
                              'spectral signal image',
                              'spectroscopic OCT']

        images.append(img_list)
        titles.append(title_list)

    fig, axs = plt.subplots(len(images), len(images[0]), figsize=(16, 9))
    for k in range(len(images)):
        temp_img = images[k]
        temp_title = titles[k]

        for n in range(len(temp_img)):
            if n == 0:
                axs[k,n].imshow(temp_img[n], 'gray',
                            vmin= 0.65 * np.max(temp_img[n]), vmax=np.max(temp_img[n]))
            elif n == 1:
                axs[k, n].imshow(temp_img[n], 'hot',
                                 vmin=0.1 * np.max(temp_img[n]), vmax=np.max(temp_img[n]))
            else:
                axs[k, n].imshow(temp_img[n])
            axs[k,n].set_title(temp_title[n])
            axs[k,n].set_axis_off()
    fig.suptitle('sftt method')
    plt.tight_layout()

    # figure_folder = '../figure'
    # if not os.path.isdir(figure_folder):
    #     os.mkdir(figure_folder)
    # fig_path = os.path.join(figure_folder,'figure.jpeg')
    # plt.savefig(fig_path, format='jpeg',
    #     bbox_inches=None, pad_inches=0)
    plt.show()