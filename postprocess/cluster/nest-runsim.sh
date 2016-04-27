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

#PBS -l walltime=10:00:00
#PBS -l nodes=20
#PBS -m abe
#PBS -N "nest-timetest"

module unload mpi/mpich-x86_64
module load openmpi

SOURCE_PATH="/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016/src/Sinha2016.py"
RESULT_PATH="/stri-data/asinha/results/"
SIM_TIME=$(date +%Y%m%d%H%M)

echo "ANKUR>> Begun at $SIM_TIME"

mkdir $RESULT_PATH/$SIM_TIME
cd $RESULT_PATH/$SIM_TIME

mpirun -n 20 python $SOURCE_PATH

END_TIME=$(date +%Y%m%d%H%M)
echo "ANKUR>> Ended at $END_TIME"
