#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm, det, inv

class crystal_structure:


    def __init__ (self, filename):
        self.fname = filename
        self.crystal_info = np.loadtxt(self.fname)
        self.check()
        
        if self.crystal_info.shape[0] == 4:
            self.alatt = self.crystal_info[0]
            self.ang = np.deg2rad(self.crystal_info[1])
            self.u = self.crystal_info[2]
            self.v = self.crystal_info[3]
        
        sin_ang, cos_ang = np.sin(self.ang), np.cos(self.ang)
        V = np.sqrt(np.abs(1 - np.sum(cos_ang**2) + 2 * np.prod(cos_ang)))
        
        self.alatt_ = 2 * np.pi * sin_ang / self.alatt / V
        self.ang_ = np.arccos( (np.roll(cos_ang, 1) * np.roll(cos_ang, 2) - cos_ang) / \
                    (np.roll(sin_ang, 1) * np.roll(sin_ang, 2)) )     
    
    def check(self):
        assert (self.crystal_info.shape[1] == 3), "Error: Wrong crystal info format!"
        if np.any(self.crystal_info[0]<=0):
            raise ValueError("Lattice constants should be positive!")
        if np.any(self.crystal_info[1] <= 0) or np.any(self.crystal_info[1] >= 180):
            raise ValueError("Lattice angles should be in the range (0, 180) deg!")
        return None

    def get_UB_matrix_inv(self):
        a_, b_, c_ = self.alatt_
        alpha_, beta_, gamma_ = self.ang_
        self.B = np.array([[a_, b_ * np.cos(gamma_),  c_ * np.cos(beta_)], \
                           [ 0, b_ * np.sin(gamma_), -c_ * np.sin(beta_) * np.cos(self.ang[0]) ], \
                           [ 0,                   0,  2 * np.pi / self.alatt[-1]]])
        
        u_col, v_col = np.expand_dims(self.u, 1), np.expand_dims(self.v, 1)
        u_c, v_c = np.squeeze(np.matmul(self.B, [u_col, v_col]))
        
        e1 = u_c / norm(u_c)
        e3 = np.cross(u_c, v_c) / norm(np.cross(u_c, v_c))
        e2 = np.cross(e3, e1)
        
        U = np.vstack((e1, e2, e3))
        self.U = U / det(U)
        
        return inv(np.matmul(self.U, self.B))



