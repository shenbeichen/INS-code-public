#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, SkewedVoigtModel
from slices import cut
from icp import icp

def fit_1peak_1direction(I, pos, cut_length, cut_width, cut_thickness, \
        energy_window, ix_direction, model_name, verbose=False):
    """
    To fit the peaks along one direction.
    @paras:
        I: the dataset
        pos: [H0, K0, L0] the initial postion of the peak
        cut_length, cut_width, cut_thickness, energy_window, ix_direction:
            All these are cutting setting.
            For example, if given
            ----------------------------------
            pos = [1, 2, 3], cut_length=1, cut_width=0.05, cut_thickness=0.2, 
            energy_window=4, ix_direction=2.
            ----------------------------------
            These will make cuts along [0K0] directions, giving the slice as
            H: [0.9, 1.1]            #cut_thickness=1.1-0.9=0.2
            K: [1.5, 2.5, step=0.05] #cut_length=2.5-1.5=1, cut_width=step=0.05
            L: [2.9, 3.1]            #cut_thickness=3.1-2.9=0.2
            energy: [-2, 2]          #energy_window=2-(-2)=4meV
        model_name: the name of the model to fit the peak. Default: Gaussian.
            Options: Gaussian, Lorentzian, Voigt, SkewedVoigt. 
            Only the first letter in the name matters.
        verbose: print/save the fitting curves: Default: False.
    @returns:
        the peak position along that direction
    """
    
    """
    To gather the information to make cut of the data. Format:
    QE_info: list or numpy array
            [[H_start, H_end, H_step],
             [K_start, K_end, K_step],
             [L_start, L_end, L_step],
             [energy_start, energy_end, energy_step]]
            If no step value or the step value is 0, means only one bin 
            [start, end) for that component.
    """
    hist_QE_info = np.zeros((4, 3))
    hist_QE_info[:3, 0] = pos
    hist_QE_info[:3, 1] = pos
    tmp = 0.5 * [cut_length if i == ix_direction else cut_thickness for i in xrange(3)]
    hist_QE_info[:, 0] -= tmp
    hist_QE_info[:, 1] += tmp
    hist_QE_info[ix_direction, 2] = cut_width
    hist_QE_info[3, 0] = -0.5*energy_window
    hist_QE_info[3, 1] = 0.5*energy_window
    
    # cut
    data = cut(I, hist_QE_info)
    x, y = data[ix_direction], data[-1]
    
    # fit the peak
    flag = model_name[0].upper()
    if  flag == 'G':
        mod = GaussianModel()
    elif flag == 'L':
        mod = LorentzianModel()
    elif flag == 'V':
        mod = VoigtModel()
    elif flag == '':
        mod = SkewedVoigtModel()
    else:
        print('cannot identify the model name, use Gaussain model anyway...')
        mod = GaussianModel()
    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    
    if verbose:
        # TODO: save the plot
        pass 
    
    return pars.valuesdict()['center'] # return the center of the peak

def get_1peak_position(I, pos, cut_length, cut_width, cut_thickness, \
        energy_window, model_name, threshold=0.1, verbose=False, max_iterations=10):
    """
    To get the position of one peak.
    """
    
    pre_pos = np.array(pos)
    for i in xrange(max_iterations):
        post_pos = np.zeros(3)
        
        # update the position
        for ix_direction in xrange(3):
            post_pos[ix_direction] = fit_1peak_1direction(I, pos, cut_length, \
                cut_width, cut_thickness, energy_window, ix_direction, model_name)
                
        # check the changes
        if np.norm(post_pos - pre_pos) <= threshold:
            return post_pos
        else:
            pre_pos = post_pos
    return False # cannot get a good fitting result
    
def get_braggpeaks_position(I, ideal_positions, cut_length, cut_width, \
        cut_thickness, energy_window, model_name='Gaussian', search_limit=0.5):
    """
    To get the positions of bragg peaks in the data set.
    @paras:
        I: the dataset
        ideal_positions: ideal positions. Also these are the initial positions
            we start to look for the bragg peaks.
        search_limit: the search limit to get the bragg peaks. Default: 0.5. 
            For example, if we search for a bragg peaks near [0, 0, 0]. We will
            drop the result if we find the peak is at [0.6, 0.55, 0.6].
    @returns:
        The positions of "good" peaks.
    """
    ix_bad_peaks, good_peaks = [], []
    for i, pos in enumerate(ideal_positions):
        real_pos = get_1peak_position(I, pos, cut_length, cut_width, cut_thickness, energy_window, model_name)
        if real_pos and np.max(real_pos-pos) < search_limit:
            good_peaks.append(real_pos)
        else:
            ix_bad_peaks.append(i)
    print('list of peaks that are not found:', ix_bad_peaks)
    return good_peaks
    
def get_linear_transform_matrix(ideal_pos, real_pos):
    """
    To get the linear transformation matrix mapping the Bragg peaks in the data
    set to their theoretical positions, using the Iterative Closest Point method.
    """
    src, dst = np.array(real_pos), np.array(ideal_pos)
    return icp(src, dst)[0]
    
def transfer(I, transform_matrix):
    """
    To transfer the whole data set.
    """
    num_points = I.shape[0]
    tmp = np.vstack((I[:, :3].T, np.ones(num_points)))
    I[:, :3] = np.dot(transform_matrix, tmp)[:3].T
        
    
    
    
    
    
        
            
        
    