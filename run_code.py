#!/usr/bin/env python3

from setup1 import *
from nxspe import *
from crystal_structure import *
import time

path = set_working_directory()
files = get_file_list()
c = crystal_structure( 'crystal_structure.info' )
Binv = c.get_inverse(c.get_B_matrix())
for file in files:
    start = time.time()
    n = nxspe(file)
    Uinv = c.get_inverse(c.get_U_matrix(n.psi))
    n.convert_to_hkl(Binv,Uinv)
    end = time.time()
    print ('Processed ', file, ' in ', end-start, ' seconds.')

