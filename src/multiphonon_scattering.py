#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np

from pyspark.sql.functions import udf
from pyspark.sql.types import *

from math import exp, factorial, pow
from scipy.interpolate import interp1d
from scipy.integrate import quad

import matplotlib.pyplot as plt

from atomic_NIST_data import *


u2kg = 1.66054e-27
barn2m_square = 1.0e-24 * 0.01 * 0.01
meV2Joule = 1.60217656535e-22
hbar = 1.0545718e-34 # unit: J*s

class MultiPhononScattering():
    
    def __init__(self, dos_file, dict_atoms, temperature, dft_energy_rescale=1.0, max_n=5):
        self.T = temperature
        self.kBT = 8.6173303e-5 * self.T * 1000 # unit: meV
        self.num_atom_species = len(dict_atoms)
        self.atom_names = [*dict_atoms] # For Python >=3.5
        self.mole_fractions = np.array([dict_atoms[atom_name] for atom_name in dict_atoms], dtype='float')
        
        self.get_pdos(dos_file, dft_energy_rescale)
        self.sigma = np.array([Atom[atom_name].sigma * barn2m_square for atom_name in dict_atoms]) #unit: m^2
        self.mass = np.array([Atom[atom_name].mass * u2kg for atom_name in dict_atoms]) # unit: kg
        
        self.neutron_weights = self.sigma / self.mass * self.mole_fractions
        self.neutron_weights /= np.sum(self.neutron_weights)
        
        self.max_n = max_n
        self.A = []
    
    def get_pdos(self, infile, scale):
        dos = np.nan_to_num(np.loadtxt(dos_file))
        
        assert dos.ndim == 2, "Invalid dos file format, at least 2 columns as (e, g1, g2...)!"
        assert self.num_atom_species == dos.shape[1] - 1, "Number of atom species does not match!"
        
        if np.any(dos[:, 0] < 0):
            print("Attention: the negative part will be truncated and the whole dos will be renormalized.")
        start_ix = np.argmax(energy > 0)
        dos = dos[start_ix:]
        self.dos_energy = dos[:, 0] * scale
        self.pdos = []
        for d in range(self.num_atom_species):
            self.pdos.append(dos[:, d+1] / np.trapz(dos[:, d+1], self.dos_energy))
        
        return None
    
    def get_higher_order_phonon_spectra(self, show=False, save=True, outdir='./higher_order_spectra_res/'):
        T1 = time.time()
        
        self.A = [[] for d in range(self.num_atom_species)]
        n = 1
        
        print("calculating A"+str(n)+" ....")
        t1 = time.time()
        for d in range(self.num_atom_species):
            y_pos = self.pdos[d] / (self.dos_energy * (1. - np.exp(-self.dos_energy /  self.kBT)))
            y_neg = self.pdos[d] / (self.dos_energy * (np.exp(self.dos_energy /  self.kBT) -1))
            y1 = np.hstack((np.flipud(y_neg), y_pos))
            x1 = np.hstack((-np.flipud(energy), energy))
            y1 = y1 / np.trapz(y1, x1)
            A1 = interp1d(x1, y1, bounds_error=False, fill_value=0.)
            self.A[d].append(A1)
            
            if save or show:
                plt.plot(x1, y1, label='A'+str(n)+' of atom '+ self.atom_names[d])
                plt.xlabel("Energy (meV)")
                plt.ylabel(str(n)+"-phonon spectra (1/meV)")
                plt.legend(loc='upper left')
                if save:
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    fname = 'A'+str(n)+' of atom '+ self.atom_names[d]
                    plt.savefig(outdir+fname, dpi=300)
            
                if show:
                    plt.show()
                plt.close()
        t2 = time.time()
        print("A"+str(n)+"done, time used: {:.2f} seconds.".format(t2-t1))
        
        x_old = x1
        while n < self.max_n:
            n += 1
            print("calculating A"+str(n)+" ....")
            t1 = time.time()
            x_new = np.linspace(x1.min()+x_old.min(), x1.max()+x_old.max(), num=100*n, endpoint=True)
            for d in range(self.num_atom_species):
                y_new = np.zeros(x_new.shape[0])
                for i in range(x_new.shape[0]):
                    if n > 2:
                        y_new[i] = 0.5 * (quad(lambda x: self.A[d][0](x_new[i]-x) * self.A[d][-1](x), x_old.min(), x_old.max())[0]\
                            + 1./n * quad(lambda x: self.A[d][0](x_new[i]-x) * self.A[1-d][-1](x), x_old.min(), x_old.max())[0]\
                            + (n-1) / float(n) * quad(lambda x: self.A[1-d][0](x_new[i]-x) * self.A[d][-1](x), x_old.min(), x_old.max())[0])
                    else:
                        y_new[i] = 0.5 * (quad(lambda x: self.A[d][0](x_new[i]-x) * self.A[d][0](x), x_old.min(), x_old.max())[0]\
                            + quad(lambda x: self.A[d][0](x_new[i]-x) * self.A[1-d][0](x), x_old.min(), x_old.max())[0])  
                y_new = y_new / np.trapz(y_new, x_new)
                A_new = interp1d(x_new, y_new, bounds_error=False, fill_value=0.)
                self.A[d].append(A_new)
                
                if save or show:
                    plt.plot(x_new, y_new, label='A'+str(n)+' of atom '+ self.atom_names[d])
                    plt.xlabel("Energy (meV)")
                    plt.ylabel(str(n)+"-phonon scattering spectra (1/meV)")
                    plt.legend(loc='upper left')
                    if save:
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        fname = 'A'+str(n)+' of atom '+ self.atom_names[d]
                        plt.savefig(outdir+fname, dpi=300)
            
                    if show:
                        plt.show()
                    plt.close()
            x_old = x_new
            t2 = time.time()
            print("A"+str(n)+"done, time used: {:.2f} seconds.".format(t2-t1))
            
        T2 = time.time()
        print("Finish calculating the phonon scattering spectra. Total time used: {:.2f} seconds.".format(T2-T1))
        return None
    
    def pre_twoW_DebyeWaller_factor(self):
        """
        2W = pre_twoW * (Q^2) # unit of Q: m
        """
        self.pre_twoW = []
        for d in range(self.num_atom_species):
            integral = self.pdos[d] /  self.dos_energy / np.tanh(self.dos_energy / 2. / self.kBT)
            self.pre_twoW.append(hbar ** 2 / 2. / self.mass[d] * np.trapz(integral, self.dos_energy) / meV2Joule)
        return None
    
    def cal_multiphonon_scattering_point(self, Q_square, E):
        res = 0
        two_W = self.pre_twoW * Q_square
        for n in range(2, max_n+1):
            for d in range(self.num_atom_species):
                res += self.neutron_weights[d] * exp(-two_W) * pow(two_W, n) * factorial(n) * self.A[d][n-1](E)
        return res
    
    def get_multiphonon_scattering(self, df, scaled_W): # scale = 2pi/a
        
        t1 = time.time()
        print("Start: multiphonon scattering calculation...")
        
        # TODO
        # calculate Q_square with scale = 2pi/a
        # df.Q_square needed
        
        udf_S = udf(lambda Q_square, E: self.cal_multiphonon_scattering_point(Q_square, E), floatType())
        new_df = df.withColumn("I_bg_mps", udf_S(df.Q_square, df.E))
        
        t2 = time.time()
        print("Finish: multiphonon scattering calculation. Total time used: {:.2f} seconds.".format(t2-t1))
        
        return new_df





