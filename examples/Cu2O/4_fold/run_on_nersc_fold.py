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

infile_peaks = "/global/cscratch1/sd/yshen/v2/src4/log.txt"
actual_positions = np.loadtxt(infile_peaks, delimiter=',')
ideal_positions = np.round(actual_positions)

T,_,_ = icp(actual_positions, ideal_positions)
print(T)

infile = "/global/cscratch1/sd/yshen/v2/QEI_no_nan.csv"
dataschema = StructType([StructField("H", FloatType(), False), \
                         StructField("K", FloatType(), False), \
                         StructField("L", FloatType(), False), \
                         StructField("E", FloatType(), False), \
                         StructField("I", FloatType(), False)])
df = spark.read.csv(infile, sep=",", schema=dataschema)
df.show()
corrected_df = correct_q_offset(df, T).cache()
corrected_df.show()

W = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1.]])
folded_df = fold(corrected_df, W, ideal_positions).cache()
folded_df.show()



