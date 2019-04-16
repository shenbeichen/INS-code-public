#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import norm
from pyspark.sql.functions import udf
import pyspark.sql.functions as func
from math import floor
from pyspark.sql.types import *

def cut_along_path(df, start, end, thickness, step=0, num_intervals=0): 
    assert bool(step) != bool(num_intervals), "Invalid step or num_intervals. Do not set both of them."
    
    if isinstance(start, list):
        start = np.array(start)
    else:
        assert isinstance(start, np.ndarray), "Start point must be a list or numpy array!"
    if isinstance(end, list):
        end = np.array(end)
    else:
        assert isinstance(end, np.ndarray), "End point must be a list or numpy array!"
    
    schema = StructType([StructField("in", IntegerType(), False), StructField("ix", IntegerType(), False)])
    r0 = end - start
    norm_r0 = norm(r0)
    def find_bin_ix(H, K, L):
        point = np.array([H, K, L])
        r = point - start
        norm_r = norm(r)
        theta = np.arccos( np.dot(r, r0) / (norm_r * norm_r0) )
        if theta > np.pi /2:
            return (0, 0)
        h = norm_r * np.sin(theta)
        if h > thickness:
            return (0, 0)
        l = norm_r * np.cos(theta)
        if l >= norm_r0:
            return (0, 0)
        elif step:
            return (1, floor(l / step))
        else:
            return (1, floor(l / (norm_r0 / num_intervals)))
    udf_find_bin_ix = udf(find_bin_ix, schema)
    
    df_in_range = df.withColumn("bins", func.explode(func.array(udf_find_bin_ix(df.x, df.y, df.z))))
    df_in_range = df_in_range.select("bins.*","E", "Q2", "I", "I_bg")
    df_in_range = df_in_range.filter(df_in_range["in"] == 1).drop("in")
    return df_in_range
