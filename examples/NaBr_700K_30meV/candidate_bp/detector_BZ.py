#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct
import pyspark.sql.functions as func
from pyspark.sql.types import *

import numpy as np
from numpy.linalg import inv


def occupancy_BZ(infile, W=np.eye(3)):
    spark = SparkSession.builder.appName("slice").getOrCreate()
    dataschema = StructType([ StructField("H", FloatType(), False), \
                              StructField("K", FloatType(), False), \
                              StructField("L", FloatType(), False), \
                              StructField("E", FloatType(), False), \
                              StructField("I", FloatType(), False), \
                              StructField("I_bg", FloatType(), False)])
    df = spark.read.csv(infile, sep=",", schema=dataschema)
    if np.allclose(W, np.eye(3)):
        df_BZ = df.withColumn("BZ_H", func.round(df.H)).withColumn("BZ_K", func.round(df.K)).withColumn("BZ_L", func.round(df.L))
    else:
        print("Transform applied...")
        inv_W = inv(W)
        df_BZ = df.withColumn("BZ_H", func.round(inv_W[0, 0] * df.H + inv_W[0, 1] * df.K + inv_W[0, 2] * df.L))\
                  .withColumn("BZ_K", func.round(inv_W[1, 0] * df.H + inv_W[1, 1] * df.K + inv_W[1, 2] * df.L))\
                  .withColumn("BZ_L", func.round(inv_W[2, 0] * df.H + inv_W[2, 1] * df.K + inv_W[2, 2] * df.L))
    stat_BZ = np.array(df_BZ.groupBy("BZ_H", "BZ_K", "BZ_L").count().collect()).astype(int)
    spark.stop()
    return stat_BZ[stat_BZ[:, -1].argsort()]


def occupancy_BZ_near(infile, W=np.eye(3), grid=0.1):
    spark = SparkSession.builder.appName("slice").getOrCreate()
    dataschema = StructType([ StructField("H", FloatType(), False), \
                              StructField("K", FloatType(), False), \
                              StructField("L", FloatType(), False), \
                              StructField("E", FloatType(), False), \
                              StructField("I", FloatType(), False), \
                              StructField("I_bg", FloatType(), False)])
    df = spark.read.csv(infile, sep=",", schema=dataschema)
    if not np.allclose(W, np.eye(3)):
        print("Transform applied...")
        inv_W = inv(W)
        df = df.withColumn("H_", inv_W[0, 0] * df.H + inv_W[0, 1] * df.K + inv_W[0, 2] * df.L)\
               .withColumn("K_", inv_W[1, 0] * df.H + inv_W[1, 1] * df.K + inv_W[1, 2] * df.L)\
               .withColumn("L_", inv_W[2, 0] * df.H + inv_W[2, 1] * df.K + inv_W[2, 2] * df.L)\
               .drop('H', 'K', 'L')\
               .withColumnRenamed('H_', 'H').withColumnRenamed('K_', 'K').withColumnRenamed('L_', 'L')
    df_BZ = df.filter(df.E>=0)\
              .withColumn("BZ_H", func.round(df.H)).withColumn("BZ_K", func.round(df.K)).withColumn("BZ_L", func.round(df.L))\
              .withColumn("sub_BZ_H", func.round( (col("H")-col("BZ_H"))/grid ))\
              .withColumn("sub_BZ_K", func.round( (col("K")-col("BZ_K"))/grid ))\
              .withColumn("sub_BZ_L", func.round( (col("L")-col("BZ_L"))/grid ))
    stat_BZ = np.array(df_BZ.groupBy("BZ_H", "BZ_K", "BZ_L")\
                            .agg(countDistinct("sub_BZ_H", "sub_BZ_K", "sub_BZ_L"))\
                            .collect()\
                       ).astype(int)
    spark.stop()
    return stat_BZ[stat_BZ[:, -1].argsort()]


#test:

infile = "/global/cscratch1/sd/yshen/v6_700K/QEI_no_nan_with_bg.csv"
res = occupancy_BZ_near(infile, grid=0.05)
print(res)
np.savetxt("HKL_op.txt", res)





