#!/usr/bin/env python3

import h5py as h5
import numpy as np
from itertools import chain
from numpy import linalg as LA

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
        self.np = self.polar.size
        self.ne = self.energy.size-1
        self.rebinned_energy = (self.energy[0:self.ne]+self.energy[1:self.ne+1])*0.5
        self.h = np.zeros( self.np * self.ne )
        self.k = np.zeros( self.np * self.ne )
        self.l = np.zeros( self.np * self.ne )


    def convert_to_hkl( self ):
        # Conversion factor: 1 mev --> 0.6947 1/A for cold neutrons
        # This value might be wrong
        unitscaling = 0.694692092176654
        h = []
        k = []
        l = []
        k_i = self.incident_energy*unitscaling
        for energy in self.rebinned_energy:
            k_f = energy*unitscaling
            h.append(-k_f*np.sin(self.polar)*np.cos(self.azimuthal))
            k.append(-k_f*np.sin(self.polar)*np.sin(self.azimuthal))
            l.append(k_i-k_f*np.cos(self.polar))
        self.h = np.array( chain.from_iterable(h) )
        self.k = np.array( chain.from_iterable(k) )
        self.l = np.array( chain.from_iterable(l) )

    def get_B_matrix( self, a, b, c  ):
        # read in primitive vectors
        # convert to reciprocal but drop the 2*pi
        a_star = np.cross(b,c)/np.dot(a,np.cross(b,c))
        b_star = np.cross(c,a)/np.dot(b,np.cross(c,a))
        c_star = np.cross(a,b)/np.dot(c,np.cross(a,b))

    def get_U_matrix( self, u , v ):
        t1 = u / LA.norm(u)
        t2 = v / LA.norm(v)
        t3 = np.cross(t1,t2)
        U_mat = [[t1[0]*np.sin(self.psi), t2[0]*np.cos(self.psi),0.],
                 [0., 0.,t3[2]*1.],
             [t1[2]*u[2]*np.cos(self.psi), -t2[2]*np.sin(self.psi),0.]]
        return U_mat





