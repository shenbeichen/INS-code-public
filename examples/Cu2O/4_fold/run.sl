#!/bin/bash

#SBATCH -q debug
#SBATCH -N 3
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH --image=nersc/spark-2.3.0:v1
#SBATCH --volume="/global/cscratch1/sd/yshen/tmpfiles:/tmp:perNodeCache=size=200G"

module load python3
module load spark/2.3.0

start-all.sh

shifter spark-submit run_on_nersc_fold.py

stop-all.sh
