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

SOURCE_PATH="/home/asinha/Documents/02_Code/00_repos/00_mine/Sinha2016/scripts/cluster/"
RUN_SCRIPT="nest-runsim.sh"

pushd $SOURCE_PATH
    GIT_SHORT_COMMIT=$(git lg | head -1 | cut -d" " -f2 | sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g")
    RUN_NEW="nest-""$GIT_SHORT_COMMIT"".sh"
    echo "Commit is: $GIT_SHORT_COMMIT. File is $RUN_NEW."
popd

cp "$SOURCE_PATH""$RUN_SCRIPT" "$RUN_NEW"
sed -i "s|nest_v_s|nest_$GIT_SHORT_COMMIT|" "$RUN_NEW"
qsub "$RUN_NEW"
