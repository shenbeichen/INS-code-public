#!/usr/bin/env python3

import os
import h5py as h5
import numpy as np
from itertools import chain
from numpy import linalg as LA
import crystal_structure as cs

class nxspe:
    
    def __init__( self, filename ):
        self.filename = filename
        f = h5.File(self.filename,'r')
        self.incident_energy = np.array( f['data/NXSPE_info/fixed_energy'] )[0]
        self.psi = (np.array(  f['data/NXSPE_info/psi'] )[0] * np.pi / 180.)
        # convert to numpy arrays, do all of them just in case
        self.azimuthal = (np.nan_to_num( np.array( f['data/data/azimuthal']) ) * np.pi / 180.)
        self.polar = (np.nan_to_num( np.array( f['data/data/polar'] ) ) * np.pi / 180.)
        self.intensity = np.nan_to_num( np.array( f['data/data/data'] ) )
        self.energy = np.nan_to_num( np.array( f['data/data/energy'] ) )
        f.close()
        self.np = self.polar.size
        self.ne = self.energy.size-1
        self.rebinned_energy = (self.energy[0:self.ne]+self.energy[1:self.ne+1])*0.5
        self.ql1 = np.zeros( self.np * self.ne )
        self.ql2 = np.zeros( self.np * self.ne )
        self.ql3 = np.zeros( self.np * self.ne )
        self.hkl = np.zeros( (self.np * self.ne, 3) )


    def convert_to_hkl( self, Binv, Uinv ):
        # Conversion factor: 1 mev --> 0.6947 1/A for cold neutrons
        # This value might be wrong
        unitscaling = 0.694692092176654
        ql1 = []
        ql2 = []
        ql3 = []
        BU = np.matmul(Binv,Uinv)
        k_i = self.incident_energy*unitscaling
        for energy in self.rebinned_energy:
            k_f = energy*unitscaling
            ql1.append((k_i-k_f*np.cos(self.polar))/(2.*np.pi))
            ql2.append(-k_f*np.sin(self.polar)*np.cos(self.azimuthal)/(2.*np.pi))
            ql3.append(-k_f*np.sin(self.polar)*np.sin(self.azimuthal)/(2.*np.pi))
        self.ql1 = np.fromiter( chain.from_iterable(ql1),dtype=float )
        self.ql2 = np.fromiter( chain.from_iterable(ql2),dtype=float )
        self.ql3 = np.fromiter( chain.from_iterable(ql3),dtype=float )
        self.hkl = np.matmul( BU, [self.ql1, self.ql2, self.ql3] ).T

    def write_to_file( self ):
        # This is where we write data to hdf5 groups
        exists = os.path.isfile('fractional_coordinates.hdf5')
        if exists:
            # Store configuration file values
            f = h5.File( 'fractional_coordinates.hdf5', 'a' )
        else:
            # Keep presets
            f = h5.File( 'fractional_coordinates.hdf5', 'w' )
 
        g = f.create_group( self.filename )
        g.create_dataset('hkl', data=self.hkl,compression='gzip')
        g.create_dataset('energy', data=self.incident_energy-self.energy,compression='gzip')
        g.create_dataset('intensity', data=self.intensity,compression='gzip')
        f.close()










