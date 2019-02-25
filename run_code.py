#!/usr/bin/env python3

from setup1 import *
from nxspe import *

set_working_directory()
files = get_file_list()

for file in files:
    print ('Reading file', file)
    n = nxspe(file)
    n.convert_to_hkl()
#print(n.incident_energy)
#    n.set_nxspe_info()
#    n.set_data()

