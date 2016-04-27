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

#PBS -l walltime=00:01:00
#PBS -l nodes=64
#PBS -m abe
#PBS -N "nest-test64"

module list
module unload mpi/mpich-x86_64
module load openmpi

SOURCE_PATH="/home/asinha/Documents/00_Code/00_repos/00_mine/Sinha2016/postprocess/cluster/tests"

cd /stri-data/asinha/results/tests/

echo $LD_LIBRARY_PATH

mpirun -n 64 python $SOURCE_PATH/nest-test.py
