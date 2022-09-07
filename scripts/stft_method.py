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
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(Mappable, Orientation='vertical'):

    Ax = Mappable.axes
    fig = Ax.figure
    divider = make_axes_locatable(Ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)

    return fig.colorbar(
        mappable=Mappable,
        cax=cax,
        use_gridspec=True,
        orientation=Orientation)


"""
spectral analysis of OCT images based on methods described here

U. Morgner, W. Drexler, F. X. Kärtner,
X. D. Li, C. Pitris, E. P. Ippen, 
and J. G. Fujimoto, 
"Spectroscopic optical coherence tomography," 
Opt. Lett. 25, 111-113 (2000)

"""

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
            length = len(line_raw)

            f, _, Zxx = signal.stft(line_raw, window=('gaussian', int(length // 20)),
                                    nperseg=int(length // 5),
                                    noverlap=int(length // 10),
                                    nfft=length,
                                    return_onesided=True)
            centroid = np.empty(Zxx.shape[-1])
            for t in range(Zxx.shape[-1]):
                centroid[t] = np.sum(abs(Zxx[:, t]) * f) / np.sum(abs(Zxx[:, t]))
                cent_resample = signal.resample(centroid, len(line_raw))

            img_contrast[:,j] = signal.resample(cent_resample, raw.shape[0])
    #
        decimation_factor = 20
        start = 100
        cor_log = img_contrast[:, ::decimation_factor]
        cor_log = Aline2Bmode(cor_log)

        v_img = Aline2Bmode(Aline(raw, decimation_factor=20), range=True)

        hsv_img = formHSV(cor_log, v_img,
                          sp_thres_low=0.05,
                          sp_thres_high=0.8,
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
                            vmin= np.mean(temp_img[n]),
                                vmax=np.max(temp_img[n]))
            elif n == 1:
                ax = axs[k, n].imshow(temp_img[n],
                                      'hot',
                                 vmin = np.mean(temp_img[n]),
                                      vmax=np.max(temp_img[n]))
                colorbar(ax)
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
