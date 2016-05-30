#!/bin/bash

# Copyright 2015 Ankur Sinha 
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# File : nest-runsim.sh
#

#PBS -l walltime=48:00:00
#PBS -l nodes=50
#PBS -m abe
#PBS -N nest-v-s

module unload mpi/mpich-x86_64
module load mvapich2-1.7

SOURCE_PATH="/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016/src/Sinha2016.py"
RESULT_PATH="/stri-data/asinha/results/"
SIM_TIME=$(date +%Y%m%d%H%M)

echo ------------------------------------------------------
echo 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

echo "ANKUR>> Begun at $SIM_TIME"
echo "ANKUR>> Script: ${0}"

mkdir $RESULT_PATH/$SIM_TIME
cd $RESULT_PATH/$SIM_TIME

/usr/local/bin/mpiexec -n 50 python $SOURCE_PATH

END_TIME=$(date +%Y%m%d%H%M)
echo "ANKUR>> Ended at $END_TIME"
