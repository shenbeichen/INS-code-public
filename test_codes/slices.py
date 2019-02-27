#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def locate(dat_point, starts, ends, steps):
    """
        To get the bin index for the data point
        @paras:
            dat_point: [H, K, L, energy]
            starts: [H_start, K_start, L_start, energy_start]
            ends: [H_end, K_end, L_end, energy_end]
            steps: [H_step, K_step, L_step, energy_step]
        @returns:
            If the data point is within the range, return the index of the bin
            it falls into, otherwise return False.          
    """
    # check the consistence
    assert len(dat_point) == len(starts) == len(ends) == len(steps)
    
    if np.all((dat_point-starts)>=0) and np.all((dat_point-ends)<0):
        return np.floor((dat_point-starts) / steps) # within the range
    else:
        return False # not in the range

def get_centers(starts, ends, steps):
    """
        To get the bin centers
        @paras:
            starts: [H_start, K_start, L_start, energy_start]
            ends: [H_end, K_end, L_end, energy_end]
            steps: [H_step, K_step, L_step, energy_step]
        @returns:
            bin_centers: List of arrays, format:
            [[H_bin_centers], [K_bin_centers], [L_bin_centers], [energy_bin_centers]].
    """
    N = len(starts)
    assert len(ends) == len(steps) == N
    return [np.arange(starts[i]+0.5*steps[i], ends[i]-0.5*steps[i]+1e-5, steps[i]) for i in xrange(N)]
    

def cut(I, QE_info):
    """
    To get the slice of the data
    @paras:
        I: the data set. Format: numpy array
            H_1, K_1, L_1, energy_1, I_1
            ..., ..., ..., ........, ...
            ...
            ..., ..., ..., ........, ...
            H_n, K_n, L_n, energy_n, I_n 
        QE_info: list or numpy array
            [[H_start, H_end, H_step],
             [K_start, K_end, K_step],
             [L_start, L_end, L_step],
             [energy_start, energy_end, energy_step]]
            If no step value or the step value is 0, means only one bin 
            [start, end) for that component.
    @returns:
        bin_centers: List of arrays, format:
            [[H_bin_centers], [K_bin_centers], [L_bin_centers], [energy_bin_centers]].
        cumulated intensities: arrary has the shape of 
            (num_H_bins, num_K_bins, num_L_bins, num_energy_bins)
    """
    if type(QE_info) == np.ndarray:
        N_dimension = QE_info.shape[0]
    else:
        N_dimension = len(QE_info)
    assert N_dimension == I.shape[1]-1
    
    # gather the info of statrs, ends, steps
    starts, ends, steps = np.zeros(N_dimension), np.zeros(N_dimension), np.zeros(N_dimension)
    for i in xrange(N_dimension):
        starts[i] = QE_info[i][0]
        ends[i] = QE_info[i][1]
        if len(QE_info[i]) == 2 or QE_info[i][2] == 0:
            steps[i] = starts[i] - ends[i] 
        else:
            steps[i] = QE_info[i][2]
            
    # get the bin centers    
    bin_centers = get_centers(starts, ends, steps)
    
    # initialize 
    cum = np.zeros([len(b) for b in bin_centers])
    counts = np.zeros_like(cum)
    
    for i in xrange(I.shape[0]):
        ix_bin = locate(I[i], starts, ends, steps)
        if ix_bin:
            cum[ix_bin] += I[i, -1]
            counts[ix_bin] += 1
    cum[counts==0] = np.nan # no intensity data at that point
    cum /= counts
    return bin_centers, cum
    
            
            
        