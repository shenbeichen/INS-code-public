#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from bragg_peaks_position import *

raw_dat = np.loadtxt('HKL_op.txt')
peaks = np.flipud(raw_dat[:, :3])

infile = "/global/cscratch1/sd/yshen/v6_700K/QEI_no_nan_with_bg.csv"
good_pks = get_braggpeaks_position(infile, peaks, cut_length=1.0, cut_width=0.02, \
    cut_thickness=0.2, energy_window=5.0, model_name='p', show=False, save=True)
print(good_pks)
res = np.array(good_pks)
np.savetxt("gpks.txt", res)

