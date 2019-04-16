#!/usr/bin/env python
# coding: utf-8
import numpy as np
from icp import *
from correct_q_offset import *

from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("folding").getOrCreate()
spark.sparkContext.addPyFile("fold_back.py")

from fold_back import *

infile_peaks = "/global/cscratch1/sd/yshen/v6_700K/bps/gpks.txt"
actual_positions = np.loadtxt(infile_peaks)#, delimiter=',')
ideal_positions = np.round(actual_positions)

T,_,_ = icp(actual_positions, ideal_positions)
print(T)

infile = "/global/cscratch1/sd/yshen/v6_700K/QEI_no_nan_with_bg.csv"
dataschema = StructType([StructField("H", FloatType(), False), \
                         StructField("K", FloatType(), False), \
                         StructField("L", FloatType(), False), \
                         StructField("E", FloatType(), False), \
                         StructField("I", FloatType(), False), \
                         StructField("I_bg", FloatType(), False)])
df = spark.read.csv(infile, sep=",", schema=dataschema)
df = df.filter(df.E >= 0)
df.show()
corrected_df = correct_q_offset(df, T).cache()
corrected_df.show()

W = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1.]])
folded_df = fold(corrected_df, W, ideal_positions).cache()
folded_df.show()
#print(folded_df.count())
folded_df.write.csv('/global/cscratch1/sd/yshen/v6_700K/q-offset/folded_dat_700K')

