#!/usr/bin/env python
# coding: utf-8


# One of the DANSE packages contains everything about neutron scattering.
# We can use that later. Or simply copy the table they showed in their reference.
nist_data ={
    'Na': [22.989769, 3.28],
    'Br': [79.904, 5.9],
    'Cu': [63.546, 8.03],
    'O' : [15.999, 4.232]
    'Si': [28.0855, 2.167]
}

class Atom:
    def __init__(self, name):
        self.name = name
        self.mass = nist_data[name][0]
        self.sigma = nist_data[name][1]
    
    def get_mass(self):
        return self.mass
    
    def get_total_scattering_cross_section(self):
        return self.sigma





