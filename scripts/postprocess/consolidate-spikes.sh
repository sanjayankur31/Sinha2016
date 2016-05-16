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
# File : consolidate-spikes.sh
#

SORTTMPDIR="/home/asinha/dump/sort-tmpdir"

echo "Combining files"
mkdir consolidated_files

echo "Combining recall files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*recall*.gdf > spikes-recall.gdf
mv spikes-recall.gdf consolidated_files

echo "Combining pattern files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*pattern*.gdf > spikes-pattern.gdf
mv spikes-pattern.gdf consolidated_files


echo "Combining lesion files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*lesioned*.gdf > spikes-lesioned.gdf
mv spikes-lesioned.gdf consolidated_files

echo "Combining background files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*background*.gdf > spikes-background.gdf
mv spikes-background.gdf consolidated_files

echo "Combining E files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*E*.gdf > spikes-E.gdf
mv spikes-E.gdf consolidated_files

echo "Combining I files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*I*.gdf > spikes-I.gdf
mv spikes-I.gdf consolidated_files

echo "Combining ExtE files"
sort -k "2" -n --parallel=16 -T $SORTTMPDIR spikes-*ExtE*.gdf > spikes-ExtE.gdf
mv spikes-ExtE.gdf consolidated_files


echo "All files combined."
exit 0
