#!/usr/bin/env python3
# coding: utf-8

import sys
import h5py as h5
import numpy as np
from numpy.linalg import inv
import time
from math import sqrt

get_k = lambda energy : sqrt( energy ) * 0.694692092176654

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
        sin_psi, cos_psi = np.sin(self.psi), np.cos(self.psi)
        self.sepc_to_uv = np.array([[ cos_psi, sin_psi, 0], \
                                    [-sin_psi, cos_psi, 0], \
                                    [       0,       0, 1]])
        # uv_to_rlu = inv_UB
        spec_to_rlu = np.matmul(inv_UB, self.sepc_to_uv)
        
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


def check_nxspe_consistency(n1, n2):
    
    assert n1.incident_energy == n2.incident_energy, "Incident energy does not match!"
    assert np.allclose(n1.azimuthal, n2.azimuthal) and np.allclose(n1.polar, n2.polar), "Geometry does not match!"
    assert np.allclose(n1.ph_energy_boundries, n2.ph_energy_boundries), "Energy bins do not match!"
    assert np.all(n1.masked_region == n2.masked_region), "Masked detectors do not match!"
    
    return None
            
def get_bg_nxspe(bg_data_files, ein):
    
    bg_n = nxspe(bg_data_files[0], ein=ein)
    
    for fname in bg_data_files[1:]:
        tmp_n = nxspe(fname, ein=ein)
        check_nxspe_consistency(bg_n, tmp_n)
        bg_n.intensity += tmp_n.intensity
    bg_n.intensity /= len(bg_data_files)
    
    return bg_n
        
class nxspe_with_bg:
    
    def __init__(self, sc_fname, bg_data, override_psi=0, ein=0):
        self.sc_n = nxspe(sc_fname, override_psi, ein)
        self.bg_n = bg_data
        check_nxspe_consistency(self.sc_n, self.bg_n)
        self.incident_energy = self.sc_n.incident_energy
        if not override_psi:
            self.psi = self.sc_n.psi
        else:
            self.psi = np.deg2rad(override_psi)
        self.azimuthal = self.sc_n.azimuthal
        self.polar = self.sc_n.polar
        self.ne = self.sc_n.ne
        self.ph_energy_centers = self.sc_n.ph_energy_centers
        
        sin_psi, cos_psi = np.sin(self.psi), np.cos(self.psi)
        self.sepc_to_uv = np.array([[ cos_psi, sin_psi, 0], \
                                    [-sin_psi, cos_psi, 0], \
                                    [       0,       0, 1]])
    
    def write_to_csv_while_converting( self, inv_UB, outdir='./', \
                                      throw_nan=True, keep_zeros=True, W=np.eye(3)): 
        
        if not throw_nan and not keep_zeros:
            raise ValueError("Do not keep nan value while removing zeros!")
            
        k_i = get_k(self.incident_energy)
        
        inv_W = inv(W)
        
        # uv_to_rlu = inv_UB
        spec_to_rlu = np.matmul(inv_UB, self.sepc_to_uv)
        
        if not throw_nan and not keep_zeros:
            raise ValueError("Do not keep nan value while removing zeros!")
        
        for i in range(self.ne):
            k_f = get_k(self.incident_energy-self.ph_energy_centers[i])
            
            if throw_nan:
                if keep_zeros:
                    ix_keep = np.isfinite(self.sc_n.intensity[:, i])
                else:
                    ix_keep = np.nonzero(np.nan_to_num(self.sc_n.intensity[:, i]))
                theta, phi = self.polar[ix_keep], self.azimuthal[ix_keep]
                sc_I_col = np.expand_dims(self.sc_n.intensity[:, i][ix_keep], axis=1)
                bg_I_col = np.expand_dims(self.bg_n.intensity[:, i][ix_keep], axis=1)
            else:
                # This will not waste time, they are not np.copy()
                theta, phi = self.polar, self.azimuthal
                sc_I_col = np.expand_dims(self.sc_n.intensity[:, i], axis=1)
                bg_I_col = np.expand_dims(self.bg_n.intensity[:, i], axis=1)
            
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
                
            E_col = np.full( (Q_in_rlu.shape[0], 1), self.ph_energy_centers[i] )
         
            fname = outdir + ('QEI_no_nan_with_bg.csv' if keep_zeros else 'QEI_no_nan_no_zero_with_bg.csv')
            
            with open(fname, 'ab+') as f:
                np.savetxt(f, np.hstack((Q_in_rlu, E_col, sc_I_col, bg_I_col)), fmt='%.4f,%.4f,%.4f,%.2f,%.4e,%.4e')
