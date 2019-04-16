#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

import time
import numpy as np
import itertools
from multiphonon_scattering import *
from cut_along_path import *

from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("cutting").getOrCreate()
spark.sparkContext.setLogLevel("OFF")
spark.sparkContext.addPyFile("multiphonon_scattering.py")
spark.sparkContext.addPyFile("cut_along_path.py")
spark.sparkContext.addPyFile("atomic_NIST_data.py")

infile = '/global/cscratch1/sd/yshen/v6_700K/q-offset-all/folded_dat_700K_all'
dataschema = StructType([StructField("x", FloatType(), False), \
                         StructField("y", FloatType(), False), \
                         StructField("z", FloatType(), False), \
                         StructField("Q2", FloatType(), False), \
                         StructField("E", FloatType(), False), \
                         StructField("I", FloatType(), False), \
                         StructField("I_bg", FloatType(), False)])
df = spark.read.csv(infile, sep=",", schema=dataschema).cache()
df.show()

dos_file = '/global/cscratch1/sd/yshen/v6_700K/cut_with_mps/pdos_700K.txt'
dict_atoms = {'Na':1, 'Br':1}
temperature = 700
energy_ratio = 8.30769 / 7.09459
a = 6.1383
mps = MultiPhononScattering(dos_file, dict_atoms, temperature, a, dft_energy_rescale=energy_ratio)
#mps.get_higher_order_phonon_spectra(show=False, save=True)
energies = np.arange(0.15, 27, 0.3)
#mps.cal_higher_order_phonon_spectra_at_energies(energies)
mps.load_higher_order_phonon_spectra_at_energies(energies)

tmp = [0.5, -0.5]
l_points = list(map(list, itertools.product(tmp, tmp, tmp)))
print("-"*25+" Cut along GL "+"-"*25)
for i in range(len(l_points)):
    print("(Task {:d}/{:d})Processing Gamma to".format(i+1, len(l_points)), l_points[i])
    t1 = time.time()
    df_in_range = cut_along_path(df, [0.0, 0, 0], l_points[i], 0.1, num_intervals=50).cache()
    df_in_range.show()
    df_in_range_mps = mps.get_multiphonon_scattering(df_in_range).cache()
    df_in_range_mps.show()
    res = np.array(df_in_range_mps.groupBy("ix", "E").avg('I', 'I_bg', 'I_bg_mps').collect())
    np.savetxt('GL'+str(i+1)+'.txt', res)
    t2 = time.time()
    print("Done, time used: {:.2f} seconds.".format(t2-t1))

x_points = np.vstack((np.eye(3), -np.eye(3)))
print("-"*25+" Cut along GX "+"-"*25)
for i in range(6):
    print("(Task {:d}/{:d}) Processing Gamma to".format(i+1, len(x_points)), x_points[i])
    t1 = time.time()
    df_in_range = cut_along_path(df, [0.0, 0, 0], x_points[i], 0.1, num_intervals=50).cache()
    df_in_range.show()
    df_in_range_mps = mps.get_multiphonon_scattering(df_in_range).cache()
    df_in_range_mps.show()
    res = np.array(df_in_range_mps.groupBy("ix", "E").avg('I', 'I_bg', 'I_bg_mps').collect())
    np.savetxt('GX'+str(i+1)+'.txt', res)
    t2 = time.time()
    print("Done, time used: {:.2f} seconds.".format(t2-t1))

