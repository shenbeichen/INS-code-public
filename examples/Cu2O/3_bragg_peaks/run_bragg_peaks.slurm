#!/bin/bash

#SBATCH -q regular 
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 03:00:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH --image=nersc/spark-2.3.0:v1
#SBATCH --volume="/global/cscratch1/sd/yshen/tmpfiles:/tmp:perNodeCache=size=200G"

module load python3
module load spark/2.3.0

start-all.sh

shifter spark-submit run_on_nersc_bragg_peaks.py

stop-all.sh
