#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from numpy.linalg import norm
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, SkewedVoigtModel
from slice_spark import cut
import matplotlib.pyplot as plt
import time
from math import floor, log10

from pyspark.sql import SparkSession
from pyspark.sql.types import *

def get_bin_centers(starts, ends, steps):
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
    return np.arange(starts+0.5*steps, ends-0.5*steps+1e-5, steps)


def fit_1peak_1direction(df, pos, cut_length, cut_width, cut_thickness, \
                         energy_window, ix_direction, model_name, \
                         show=False, save=False, outdir='./peaks_reports/fig/', suffix=''):
    """
    To fit the peaks along one direction.
    @paras:
        infile: the dataset file path.
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
    QE_info = np.zeros((4, 3))
    QE_info[:3, 0] = pos
    QE_info[:3, 1] = pos
    tmp = 0.5 * np.array([cut_length if i == ix_direction else cut_thickness for i in range(3)])
    QE_info[:3, 0] -= tmp
    QE_info[:3, 1] += tmp
    QE_info[ix_direction, 2] = cut_width
    QE_info[3, 0] = -0.5*energy_window
    QE_info[3, 1] = 0.5*energy_window
    
    # cut
    y = cut(df, QE_info, ignore_energy=True)
    x = get_bin_centers(*QE_info[ix_direction])
    if np.sum(np.isnan(y)) >= 0.5 * y.size:
        return np.nan
    ix_not_nan = np.isfinite(y)#np.logical_not(np.isnan(y))
    y = y[ix_not_nan]
    x = x[ix_not_nan]
    
    # fit the peak
    flag = model_name[0].upper()
    if  flag == 'G':
        mod = GaussianModel()
    elif flag == 'L':
        mod = LorentzianModel()
    elif flag == 'V':
        mod = VoigtModel()
    elif flag == 'P':
        mod = PseudoVoigtModel()
    elif flag == 'S':
        mod = SkewedVoigtModel()
    else:
        print('cannot identify the model name, use Gaussain model anyway...')
        mod = GaussianModel()
    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    if not out.success:
        return np.nan
    
    if show or save:
        
        x_labels = ["Q_"+chr(120+i)+" (r.l.u.)" for i in range(3)]
        plt.scatter(x, y, c='b')
        plt.plot(x, out.best_fit, 'r-', label=mod.name+ \
                 '\n$\\vdash$ center={:.3e}'.format(out.params.valuesdict()['center'])+ \
                 '\n$\\vdash$ red-chisqr={:.3e}'.format(out.redchi))
        plt.xlabel(x_labels[ix_direction])
        plt.ylabel("Intensity (arb. units)")
        plt.legend(loc='best')
        
        if save:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            fname = "peak_" + '_'.join(map(str, np.round(pos).astype(int))) \
                    + "_along_" + chr(120+ix_direction) + suffix
            plt.savefig(outdir+fname, dpi=300)
            
        if show:
            plt.show()
        plt.close()
    print("Fit result: center =", out.params.valuesdict()['center'])
    print("Fit Statistics: chi-sqr = {:.3e},".format(out.chisqr))
    print("                reduce chi-sqr = {:.3e}".format(out.redchi))
    print("                Akaike info crit = {:.3e}".format(out.aic))
    print("                Bayesian info crit = {:.3e}".format(out.bic))
        
    return out.params.valuesdict()['center']

def get_1peak_position(df, pos, cut_length, cut_width, cut_thickness, \
                       energy_window, model_name, threshold=0, max_iterations=10, \
                       show=False, save=False, report=False, outdir='./peaks_reports/fig/'):
    """
    To get the position of one peak.
    """
    
    pre_pos = np.array(pos)
    direction = ['X', 'Y', 'Z']
    
    # If the user didn't specify the threshold, use cut_width.
    if not threshold:
        threshold = cut_width
    print("Peak:", pos)    
    for i in range(max_iterations):
        post_pos = np.zeros(3)
        
        # update the position
        for ix_direction in range(3):
            print("iter {}, direction {}...".format(i+1, direction[ix_direction]))
            post_pos[ix_direction] = fit_1peak_1direction(df, pre_pos, cut_length, \
                                                          cut_width, cut_thickness, energy_window, ix_direction, model_name, \
                                                          show=show, save=save, suffix='_iter'+str(i+1), outdir=outdir)
            if np.isnan(post_pos[ix_direction]):
                print("Data not available or too much noise... Stop searching for this peak.")
                return np.nan
        # check the changes
        if norm(post_pos - pre_pos) <= threshold:
            print("Final position:", post_pos)
            if report: # create pdf report
                # TODO
                pass
            return post_pos
        else:
            print("Position update:", post_pos)
            pre_pos = post_pos
    else:
        print("Position update:", post_pos)
        print("Maximum number of iterations reached... Stop searching for this peak.")
                    
    return np.nan # cannot get a good fitting result


def get_braggpeaks_position(infile, ideal_positions, cut_length, cut_width, \
                            cut_thickness, energy_window, model_name='', search_limit=0.5, \
                            show=False, save=False, report=False, outdir='./peaks_reports/fig/'):
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
    job_size_order = floor (log10 (ideal_positions.shape[0])) + 1
    T1 = time.time()
    spark = SparkSession.builder.appName("slice").getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    dataschema = StructType([StructField("H", FloatType(), False), \
                             StructField("K", FloatType(), False), \
                             StructField("L", FloatType(), False), \
                             StructField("E", FloatType(), False), \
                             StructField("I", FloatType(), False), \
                             StructField("I_bg", FloatType(), False)])
    df = spark.read.csv(infile, sep=",", schema=dataschema)
    df = df.filter((df.E >= -0.5 * energy_window) & (df.E < 0.5 * energy_window)).drop("E", "I_bg")
    num_succeed_bad_peaks = 0
    for i, pos in enumerate(np.array(ideal_positions)):
        t1 = time.time()
        print("-"*25+'Peak', i+1, '-'*25)
        
        radius = max(0.5 * cut_length + search_limit, 0.5 * cut_thickness) * 1.1
        boundry = np.vstack((pos - radius, pos + radius))
        df_near_peak = df.filter((df.H >= boundry[0, 0]) & (df.H < boundry[1, 0]) & \
                                 (df.K >= boundry[0, 1]) & (df.K < boundry[1, 1]) & \
                                 (df.L >= boundry[0, 2]) & (df.L < boundry[1, 2])).cache()
        
        real_pos = get_1peak_position(df_near_peak, pos, cut_length, cut_width, cut_thickness, energy_window, \
                                      model_name, show=show, save=save, report=report, \
                                      outdir=outdir+'/peak_'+str(i+1).zfill(job_size_order)+'/')
        if np.all(np.isfinite(real_pos)) and np.all((np.abs(real_pos-pos)) < search_limit):
            with open("log.txt", 'ab') as f:
                np.savetxt(f, real_pos[None], delimiter=',')
            good_peaks.append(real_pos)
            num_succeed_bad_peaks = 0
        else:
            ix_bad_peaks.append(i)
            num_succeed_bad_peaks += 1
            if num_succeed_bad_peaks >= 20:
                print("Reach the maximum 20 succeeding bad peaks, stop searching...")
                break
        t2 = time.time()
        print("Time used: {:.2f} seconds.".format(t2-t1))
    spark.stop()
    T2 = time.time()
    print("="*60)
    print("Total time collapsed: {:.2f} seconds.".format(T2-T1))
    print('list of peaks that are not found:', ix_bad_peaks)
    print("="*57+"END")
    return good_peaks
