#!/usr/bin/env python
# coding: utf-8
import pandas as pd

# One of the DANSE packages contains everything about neutron scattering.
# We can use that later. Or simply copy the table they showed in their reference.
"""
nist_data ={
    'Na': [22.989769, 3.28],
    'Br': [79.904, 5.9],
    'Cu': [63.546, 8.03],
    'O':  [15.999, 4.232]
}
"""

df1 = pd.read_csv('./neutron_sigmas.csv', names=["Z-Symbol-A", "concentration/half-life", \
    "spin_I", "b_c", "bp", "bm", "c", "coherent", "incoherent", "total", "absorption"])

df2 = pd.read_csv('./mass.csv', names=["isotope", "m", "p", "avg"])

#df3 = pd.read_csv('./atomic_numbers.csv', names=["Z", "symbol", "element"])

class Atom:
    def __init__(self, name):
        self.name = name
        #self.mass = nist_data[name][0]
        #self.sigma = nist_data[name][1]
        if name[-1].isdigit():
            self.mass = self._parse(df2[df2["isotope"].str.contains(name)].iloc[0]["m"])
        else:
            self.mass = self._parse(df2[df2["isotope"].str.contains(name)].iloc[0]["avg"])
        self.sigma = self._parse(df1[df1["Z-Symbol-A"].str.contains(name)].iloc[0]["total"])
    
    def _parse(self, s):
        idx = s.find('(')
        if idx > 0: # value(uncertainty)
            return float(s[:idx])
        if s.startswith('['): # [nominal]
            return int(s[1:-1])
        if s == "": # missing
            return 0
        return float(s)
    
    def get_mass(self):
        return self.mass
    
    def get_total_scattering_cross_section(self):
        return self.sigma





