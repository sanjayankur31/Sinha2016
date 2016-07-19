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

module load mpi/openmpi-x86_64

SIM_PATH="/home/asinha/cluster-data/"
SIM_TIME=""
PROGRAM_PATH="$SIM_PATH""$SIM_TIME""/Sinha2016/src/Sinha2016.py"
RESULT_PATH="$SIM_PATH""$SIM_TIME""/result/"
NUM_NODES=20

echo "ANKUR>> Begun at $SIM_TIME"
echo "ANKUR>> Script: ${0}"

cd $RESULT_PATH

mpiexec -n $NUM_NODES python $PROGRAM_PATH

END_TIME=$(date +%Y%m%d%H%M)
echo "ANKUR>> Ended at $END_TIME"
