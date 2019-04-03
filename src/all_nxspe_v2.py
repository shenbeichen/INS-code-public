#!/usr/bin/env python3
# coding: utf-8

import sys
import h5py as h5
import numpy as np
from numpy.linalg import inv


import time
import math

try:
    import cPickle as pickle
except:
    import pickle
import gzip
import pickletools


"""
    Lambda function with math will be faster here
"""
get_k = lambda energy : math.sqrt( energy ) * 0.694692092176654

class nxspe:
    def __init__(self, infile_name, override_psi=0, ein=0):
        
        self.fname = infile_name
        try:
            f = h5.File(self.fname,'r')
            entry = list(f.keys())[0]
            self.incident_energy = ein if ein else f[entry+'/NXSPE_info/fixed_energy'][0]
            if not override_psi:
                self.psi = np.deg2rad( f[entry+'/NXSPE_info/psi'] )
            else:
                self.psi = np.deg2rad(override_psi)
            # convert to numpy arrays, do all of them just in case
            self.azimuthal = np.deg2rad( f[entry+'/data/azimuthal'] )
            self.polar = np.deg2rad( f[entry+'/data/polar'] )
            self.intensity = np.array( f[entry+'/data/data'] ) 
            self.ph_energy_boundries = np.array( f[entry+'/data/energy'] )
            f.close()
        except OSError:
            print('Cannot open', self.fname)
            raise
        except ValueError:
            print("Cannot convert data in", self.fname)
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0], ' while reading', self.fname)
            raise
        else:
            self.ne = self.ph_energy_boundries.size-1
            self.ph_energy_centers = (self.ph_energy_boundries[:-1]+self.ph_energy_boundries[1:])*0.5
            self.masked_region = np.isnan(self.intensity)

    # Throw zeros only for testing!!!
    def write_to_csv_while_converting( self, inv_UB, outdir='./', \
                                  throw_nan=True, keep_zeros=True, W=np.eye(3)):
    
        k_i = get_k(self.incident_energy)
        sin_psi, cos_psi = math.sin(self.psi), math.cos(self.psi)
        spec_to_uv = np.array([[ cos_psi, sin_psi, 0], \
                               [-sin_psi, cos_psi, 0], \
                               [       0,       0, 1]])
        spec_to_rlu = np.matmul(inv_UB, spec_to_uv)
        
        if not throw_nan and not keep_zeros:
            raise ValueError("Do not keep nan value while removing zeros!")
    
        for i in range(self.ne):
            k_f = get_k(self.incident_energy-self.ph_energy_centers[i])
        
            if throw_nan:
                if keep_zeros:
                    ix_keep = np.isfinite(self.intensity[:, i])
                else:
                    ix_keep = np.nonzero(np.nan_to_num(self.intensity[:, i]))
                theta, phi = self.polar[ix_keep], self.azimuthal[ix_keep]
                I_col = np.expand_dims(self.intensity[:, i][ix_keep], axis=1)
            else:
                # This will not waste time, they are not np.copy()
                theta, phi = self.polar, self.azimuthal
                I_col = np.expand_dims(self.intensity[:, i], axis=1)

            sin_theta = np.sin(theta)
            ex = np.cos(theta)
            ey = sin_theta * np.cos(phi)
            ez = sin_theta * np.sin(phi)
        
            q1 = k_i - ex * k_f
            q2 = -ey * k_f
            q3 = -ez * k_f
            
            Q_in_rlu = np.matmul( spec_to_rlu, np.vstack((q1, q2, q3)) ).T
            
            if not np.allclose(W, np.eye(3)):
                inv_W = inv(W)
                Q_in_rlu = np.matmul(inv_W, Q_in_rlu.T).T
        
            E_col = np.full( (Q_in_rlu.shape[0], 1), self.ph_energy_boundries[i] )
            
            fname = outdir + ('QEI_no_nan.csv' if keep_zeros else 'QEI_no_nan_no_zero.csv')
            
            with open(fname, 'ab+') as f:
                np.savetxt(f, np.hstack((Q_in_rlu, E_col, I_col)), fmt='%.4f,%.4f,%.4f,%.2f,%.4e')


    def temp_store_data ( self, inv_UB, outdir='./', W=np.eye(3)):
        k_i = get_k(self.incident_energy)
        sin_psi, cos_psi = math.sin(self.psi), math.cos(self.psi)
        spec_to_uv = np.array([[ cos_psi, sin_psi, 0], \
                               [-sin_psi, cos_psi, 0], \
                               [       0,       0, 1]])
        spec_to_rlu = np.matmul(inv_UB, spec_to_uv)
        
        for i in range(self.ne):
            k_f = get_k(self.incident_energy-self.ph_energy_centers[i])
            theta, phi = self.polar, self.azimuthal
            I_col = np.expand_dims(self.intensity[:, i], axis=1)
                                                                       
            sin_theta = np.sin(theta)
            ex = np.cos(theta)
            ey = sin_theta * np.cos(phi)
            ez = sin_theta * np.sin(phi)
            q1 = k_i - ex * k_f
            q2 = -ey * k_f
            q3 = -ez * k_f
            Q_in_rlu = np.matmul( spec_to_rlu, np.vstack((q1, q2, q3)) ).T
            if not np.allclose(W, np.eye(3)):
                inv_W = inv(W)
                Q_in_rlu = np.matmul(inv_W, Q_in_rlu.T).T
            
            E_col = np.full( (Q_in_rlu.shape[0], 1), self.ph_energy_boundries[i] )
            #with open(fname, 'ab+') as f:
            fname = outdir + ('QEI')
#            with gzip.GzipFile(fname, 'ab+') as f:
            with open(fname, 'ab+') as f:
                QEI_pickle = pickle.dump( np.hstack((Q_in_rlu, E_col, I_col)) , f )
                pickletools.optimize(QEI_pickle)
                





