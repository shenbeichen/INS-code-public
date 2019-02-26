#!/usr/bin/env python3
import numpy as np
from numpy import linalg as LA

class crystal_structure:


    def __init__ ( self, filename ):
        self.filename = filename
        with open( self.filename ) as f:
            lines = f.read().splitlines()
        # Read and parse the input
        self.a = [float(i) for i in lines[0].split(' ')]
        self.b = [float(i) for i in lines[1].split(' ')]
        self.c = [float(i) for i in lines[2].split(' ')]
        self.u = [float(i) for i in lines[3].split(' ')]
        self.v = [float(i) for i in lines[4].split(' ')]

        self.alpha = np.arccos(np.dot(self.b,self.c)/(LA.norm(self.b)*LA.norm(self.c)))
        self.beta  = np.arccos(np.dot(self.a,self.c)/(LA.norm(self.a)*LA.norm(self.c)))
        self.gamma = np.arccos(np.dot(self.a,self.b)/(LA.norm(self.a)*LA.norm(self.b)))
        
        self.a_ = 2*np.pi*np.cross(self.b,self.c)/np.dot(self.a,np.cross(self.b,self.c))
        self.b_ = 2*np.pi*np.cross(self.c,self.a)/np.dot(self.b,np.cross(self.c,self.a))
        self.c_ = 2*np.pi*np.cross(self.a,self.b)/np.dot(self.c,np.cross(self.a,self.b))
        self.alpha_ = np.arccos((np.cos(self.beta)*np.cos(self.gamma)-np.cos(self.alpha))/(np.sin(self.beta)*np.sin(self.gamma)))
        self.beta_  = np.arccos((np.cos(self.alpha)*np.cos(self.gamma)-np.cos(self.beta))/(np.sin(self.alpha)*np.sin(self.gamma)))
        self.gamma_ = np.arccos((np.cos(self.alpha)*np.cos(self.beta)-np.cos(self.gamma))/(np.sin(self.alpha)*np.sin(self.beta)))

    
    def get_U_matrix( self, psi):
        t1 = self.u / LA.norm(self.u)
        t2 = self.v / LA.norm(self.v)
        t3 = np.cross(t1,t2)
        U_mat = [[t2[0]*np.cos(psi)  ,t1[0]*np.sin(psi) ,0.       ],
                 [0.                 ,0.                , t3[2]*1.],
                 [-t2[2]*np.sin(psi) ,t1[2]*np.cos(psi) ,0.       ]]
        return U_mat

    def get_B_matrix( self ):
        B_mat = [[LA.norm(self.a_), LA.norm(self.b_)*np.cos(self.gamma_), LA.norm(self.c_)*np.cos(self.beta_)],
                 [0.              , LA.norm(self.b_)*np.sin(self.gamma_), -LA.norm(self.c_)*np.sin(self.beta_)*np.cos(self.alpha)],
                 [0.              , 0.                                  ,1/LA.norm(self.c)]]
        return B_mat

    def get_inverse( self, mat ):
        return LA.inv(mat)





