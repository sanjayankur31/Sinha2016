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
# File : scripts/cluster/start-job.sh
#
# Queue's up a new job for me.

SOURCE_PATH="/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016/"
GIT_COMMIT=""
SIM_PATH="/home/asinha/cluster-data/"
SIM_TIME=$(date +%Y%m%d%H%M)
RUN_SCRIPT="scripts/mac/nest-runsim.sh"
RUN_NEW=""
ERROR="no"
NUM_NODES=20
CUR_SIM_PATH=""

function run_task
{
    pushd "$CUR_SIM_PATH"
        sh "$RUN_NEW"
    popd
}

function setup_env
{
    CUR_SIM_PATH="$SIM_PATH""$SIM_TIME"
    echo "This simulation will run in: $CUR_SIM_PATH"
    mkdir -pv "$CUR_SIM_PATH"

    pushd "$CUR_SIM_PATH"
        echo "Cloning source repository..."
        git clone "$SOURCE_PATH" "Sinha2016"

        pushd "Sinha2016"
            echo "Checking out commit $GIT_COMMIT..."
            git checkout -b this_sim "$GIT_COMMIT"
            if [ "$?" -ne 0 ]
            then
                echo "Error occured. Could not checkout $GIT_COMMIT. Exiting..."
                ERROR="yes"
            fi
        popd

        if [ "xyes" ==  x"$ERROR" ] 
        then
            exit -1
        fi

        RUN_NEW="nest_""$GIT_COMMIT"".sh"
        echo "Setting up $RUN_NEW..."
        cp "$SOURCE_PATH""$RUN_SCRIPT" "$RUN_NEW" -v
        sed -i "s|nest_v_s|nest_$GIT_COMMIT|" "$RUN_NEW"
        sed -i "s|nodes=.*|nodes=$NUM_NODES|" "$RUN_NEW"
        sed -i "s|NUM_NODES=.*|NUM_NODES=$NUM_NODES|" "$RUN_NEW"
        sed -i "s|SIM_TIME=.*|SIM_TIME=$SIM_TIME|" "$RUN_NEW"

        mkdir -v result
        touch result/"00-GIT-COMMIT-""$GIT_COMMIT"
    popd
}

function usage
{
    echo "Usage: $0"
    echo "Run up a task for a particular git commit"
    echo "$0 <git_commit> <number_nodes>"
}

if [ "$#" -ne 2 ];
then
    echo "Error occurred. Exiting..."
    echo "Received $# arguments. Expected: 3"
    usage
    exit -1
fi

GIT_COMMIT="$1"
NUM_NODES="$2"
setup_env
run_task

exit 0
