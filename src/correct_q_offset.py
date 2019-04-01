#!/usr/bin/env python
# coding: utf-8

import time

def correct_q_offset(df, T):
    t1 = time.time()
    print("Start: q-offset correction...")
    corrected_df = df.withColumn("H_new", T[0, 0] * df.H + T[0, 1] * df.K + T[0, 2] * df.L + T[0, 3])\
                     .withColumn("K_new", T[1, 0] * df.H + T[1, 1] * df.K + T[1, 2] * df.L + T[1, 3])\
                     .withColumn("L_new", T[2, 0] * df.H + T[2, 1] * df.K + T[2, 2] * df.L + T[2, 3])\
                     .drop("H", "K", "L")\
                     .withColumnRenamed("H_new", "H")\
                     .withColumnRenamed("K_new", "K")\
                     .withColumnRenamed("L_new", "L")
    t2 = time.time()
    print("Finish: q-offset correction. Time used: {:.2f} seconds.".format(t2-t1))
    return corrected_df




