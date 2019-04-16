#!/usr/bin/env python
# coding: utf-8

import numpy as np
import itertools
import pyspark.sql.functions as func
from pyspark.sql.functions import udf
from pyspark.sql.types import *

def neighboors(x):
    f_ = np.floor(x).astype("int")
    c_ = np.ceil(x).astype("int")
    fc_ = np.vstack((f_, c_)).T
    return [list(i) for i in set(itertools.product(*fc_))]


def fold_all(df, W):
    schema = StructType([StructField("x", FloatType(), False), \
                         StructField("y", FloatType(), False), \
                         StructField("z", FloatType(), False), \
                         StructField("Q2", FloatType(), False)]) 
    def in_BZs(H, K, L):
        arr_HKL = np.array([H, K, L])
        near_HKLs = neighboors(arr_HKL) # is a list
        arr_Q = np.matmul(arr_HKL, W.T)
        near_Qs = np.matmul(np.array(near_HKLs), W.T)
        dist = np.sum((near_Qs - arr_Q) ** 2, axis=1)
        
        ix_min = np.argmin(dist)
        BZ_HKL = near_HKLs[ix_min]
        reduced_Q = arr_Q - near_Qs[ix_min]
        return (*list(map(float, reduced_Q)), float(np.sum(arr_Q ** 2)))
    udf_in_BZs = udf(in_BZs, schema)
    
    folded_df = df.withColumn("BZ", func.explode(func.array(udf_in_BZs(df.H, df.K, df.L))))
    folded_df = folded_df.select("BZ.*", "E", "I", "I_bg").cache()
    return folded_df

