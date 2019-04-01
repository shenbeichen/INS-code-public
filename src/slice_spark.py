#!/usr/bin/env python
# coding: utf-8

from pyspark.sql.functions import col
import pyspark.sql.functions as func

import numpy as np
from itertools import zip_longest


def convert_to_ses(QE_info):
    """
        convert the QE information to starts, ends and steps.
    """
    if isinstance(QE_info, list):
        arr_QE = np.array(list(zip_longest(*QE_info, fillvalue=0))).T
    elif type(QE_info) == np.ndarray:
        arr_QE = QE_info
    else:
        raise ValueError("Cannot read QE information for the cut!")
    assert arr_QE.shape == (4, 3), "Wrong QE information for the cut!"
    
    if np.any(arr_QE[:, 0] > arr_QE[:, 1]):
        print("Reminder: starting > ending porints. They'll be switched. You can force quit if you want.")
        starts, ends = np.amin(arr_QE[:, :2], axis=1), np.amax(arr_QE[:, :2], axis=1)
    else:
        starts, ends = arr_QE[:, 0], arr_QE[:, 1]
    
    steps = np.abs(arr_QE[:, 2])
    if np.any(steps > ends - starts):
        raise ValueError("Invalid steps!")
        
    return starts, ends, steps


def cut(df, QE_info, sorted_res=True, ignore_energy=False):
    starts, ends, steps = convert_to_ses(QE_info)
    if ignore_energy:
        heads = ['H', 'K', 'L']
        df_in_range = df.filter((df.H>=starts[0]) & (df.H<ends[0]) & \
                                (df.K>=starts[1]) & (df.K<ends[1]) & \
                                (df.L>=starts[2]) & (df.L<ends[2]))
    else:
        heads = ['H', 'K', 'L', 'E']
        df_in_range = df.filter((df.H>=starts[0]) & (df.H<ends[0]) & \
                                (df.K>=starts[1]) & (df.K<ends[1]) & \
                                (df.L>=starts[2]) & (df.L<ends[2]) & \
                                (df.E>=starts[3]) & (df.E<ends[3]))
                            
    res_heads, res_shape = [], []
    for col_ix, col_name in enumerate(heads):
        if steps[col_ix] != 0 and steps[col_ix] != ends[col_ix] - starts[col_ix]:
            res_heads.append(col_name+'_bin_ix')
            res_shape.append( int(round((ends[col_ix] - starts[col_ix]) / steps[col_ix])) )
            df_in_range = df_in_range.withColumn(col_name+'_bin_ix', \
                                                 func.floor( (col(col_name)-starts[col_ix])/steps[col_ix] ))

    if not res_heads: # means 0-Dimension
        return np.array(df.groupBy().avg('I').collect())
        
    raw_res = np.array(df_in_range.groupBy(*res_heads).agg({'I': 'mean'}).collect())
    
    if sorted_res:
        res = np.full((*res_shape), np.nan)
        if len(res_shape) == 1: # means 1-Dimension
            for row in raw_res:
                res[int(row[0])] = row[1]
        else:
            for row in raw_res:
                res[tuple(row[:-1].astype(int))] = row[-1]
        return res
    else:
        return raw_res
