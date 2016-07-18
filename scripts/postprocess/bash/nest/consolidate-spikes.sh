#!/bin/bash

# Copyright 2016 Ankur Sinha 
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
# File : consolidate-spikes.sh
#

SORTTMPDIR="/home/asinha/dump/sort-tmpdir"

echo "Combining files for NEST simulation"
mkdir consolidated_files

echo "Combining recall files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*recall*.gdf > spikes-recall.gdf
mv spikes-recall.gdf consolidated_files

echo "Combining pattern files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*pattern*.gdf > spikes-pattern.gdf
mv spikes-pattern.gdf consolidated_files

echo "Combining deaff files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*deaffed*.gdf > spikes-deaffed.gdf
mv spikes-deaffed.gdf consolidated_files

echo "Combining background files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*background*.gdf > spikes-background.gdf
mv spikes-background.gdf consolidated_files

echo "Combining E files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*E*.gdf > spikes-E.gdf
mv spikes-E.gdf consolidated_files

echo "Combining I files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*I*.gdf > spikes-I.gdf
mv spikes-I.gdf consolidated_files

echo "Combining Stim files"
LC_ALL=C sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*Stim*.gdf > spikes-Stim.gdf
mv spikes-Stim.gdf consolidated_files

echo "Copying over settings file"
#TODO copy over tasklist ini file

echo "All files combined."
exit 0
