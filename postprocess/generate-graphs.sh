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
# File : generate-graphs.sh
#

echo "Generating graphs"
pushd consolidated_files

source activate python3

(
python3 ~/Documents/00_Code/00_repos/00_mine/Sinha2016/postprocess/nest-spike2hz.py spikes-E.gdf firing-rate-E.gdf
) &
(
python3 ~/Documents/00_Code/00_repos/00_mine/Sinha2016/postprocess/nest-spike2hz.py spikes-I.gdf firing-rate-I.gdf
) &
(
python3 ~/Documents/00_Code/00_repos/00_mine/Sinha2016/postprocess/nest-spike2hz.py spikes-pattern.gdf firing-rate-pattern.gdf
) &
(
python3 ~/Documents/00_Code/00_repos/00_mine/Sinha2016/postprocess/nest-spike2hz.py spikes-E.gdf firing-rate-E.gdf
) &

wait

gnuplot ~/Documents/00_Code/00_repos/00_mine/Sinha2016/postprocess/plot-firing-rates.plt

source deactivate
