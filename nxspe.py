#!/usr/bin/env python3

import sys
import h5py as h5
import numpy as np


def get_k(energy, e_to_k = 0.694692092176654):
    """
    To get neutron wave-vector k from energy/energies.
    """
    return np.sqrt(energy) * e_to_k


class nxspe:
    
    def __init__(self, infile_name, override_psi=0):
        # 1. all degrees are in rad; 2. keep the origin, nan or 0 as it is.
        self.fname = infile_name
        try:
            f = h5.File(self.fname,'r')
            entry = list(f.keys())[0] # get the entry key value
            self.incident_energy = f[entry+'/NXSPE_info/fixed_energy'][0]
            if not override_psi:
                self.psi = np.deg2rad( f['__OWS/NXSPE_info/psi'] )
            else:
                self.psi = np.deg2rad(override_psi) # user-defined psi
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
            sin_psi, cos_psi = np.sin(self.psi), np.cos(self.psi)
            """
                spec_to_uv:
                convert from spectrometer coords to orthormal frame
                defined by notional directions of u, v.
            """
            self.sepc_to_uv = np.array([[ cos_psi, sin_psi, 0], \
                                        [-sin_psi, cos_psi, 0], \
                                        [       0,       0, 1]])
            self.ne = self.ph_energy_boundries.size-1 #number of energy bins
            self.ph_energy_centers = (self.ph_energy_boundries[:-1]+self.ph_energy_boundries[1:])*0.5
            
    
    def write_to_csv_while_converting( self, inv_UB, outdir='./', throw_nan=True, keep_zeros=True):
        # WARNING: Throw zeros only for testing!!!
        # uv_to_rlu is the matrix inv_UB.
        
        k_i = get_k(self.incident_energy)
        
        spec_to_rlu = np.matmul(inv_UB, self.sepc_to_uv)
        
        if not throw_nan and not keep_zeros:
            raise ValueError("Do not keep nan value while removing zeros!")
        
        for i in range(self.ne):
            k_f = get_k(self.incident_energy-self.ph_energy_centers[i])
            
            # get (theta, phi) we want to keep, and write corresponding I in column vector format
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
            E_col = np.full( (Q_in_rlu.shape[0], 1), self.ph_energy_boundries[i] )
         
            fname = outdir + ('QEI_no_nan.csv' if keep_zeros else 'QEI_no_nan_no_zero.csv')
            """
            output csv file format:
                delimeter: ","
                no headings(will be sepcified when we load it in Spark)
                cols=(H,K,L,E,I)
            """
            with open(fname, 'ab+') as f:
                np.savetxt(f, np.hstack((Q_in_rlu, E_col, I_col)), fmt='%.4f,%.4f,%.4f,%.2f,%.4e')
