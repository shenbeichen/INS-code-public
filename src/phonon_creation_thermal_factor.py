#!/usr/bin/env python
# coding: utf-8


from math import exp
import pyspark.sql.functions as func

def correct_thermal_factor(df, input_cols, T, replace=False):
    new_df = df
    kBT = 8.6173303e-5 * T * 1000 # unit: meV
    for col_name in input_cols:
        new_df = new_df.withColums("corrected_"+col_name, col(col_name) * (func.exp(df.E/kBT) - 1.0))
    if replace:
        print("Replace", input_cols, "with corrected values...")
        new_df = new_df.drop(*input_I_cols)
        for col_name in input_cols:
            new_df = new_df.withColumnRenamed("corrected_"+col_name, col_name)
    else:
        output_cols = ["corrected_"+col_name for col_name in input_cols]
        print("Add corrected intensity data as new column(s): ", output_cols)
    return new_df

