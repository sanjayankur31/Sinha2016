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

echo "Combining files for Auryn simulation"
mkdir consolidated_files

echo "Combining recall files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*recall*.gdf > spikes-auryn-recall.gdf
mv spikes-auryn-recall.gdf consolidated_files

echo "Combining pattern files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*pattern*.gdf > spikes-auryn-pattern.gdf
mv spikes-auryn-pattern.gdf consolidated_files


echo "Combining deaff files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*deaffed*.gdf > spikes-auryn-deaffed.gdf
mv spikes-auryn-deaffed.gdf consolidated_files

echo "Combining background files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*background*.gdf > spikes-auryn-background.gdf
mv spikes-auryn-background.gdf consolidated_files

echo "Combining E files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*E*.gdf > spikes-auryn-E.gdf
mv spikes-auryn-E.gdf consolidated_files

echo "Combining I files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*I*.gdf > spikes-auryn-I.gdf
mv spikes-auryn-I.gdf consolidated_files

echo "Combining Stim files"
sort -n --parallel=16 -T $SORTTMPDIR spikes-*Stim*.gdf > spikes-auryn-Stim.gdf
mv spikes-auryn-Stim.gdf consolidated_files

echo "Converting to nest spikes format.."
cd consolidated_files

# doesn't matter for neuron ID - it's an integer, so there's no loss of decimal places
cat spikes-auryn-E.gdf | awk '{printf "%s\t%f\n",$2, $1*1000}' > spikes-E.gdf
rm -f spikes-auryn-E.gdf

# add 8000 to the neuron ID for my rasters
cat spikes-auryn-I.gdf | awk '{printf "%d\t%f\n",$2+8000, $1*1000}' > spikes-I.gdf
rm -f spikes-auryn-I.gdf 

cat spikes-auryn-recall.gdf | awk '{printf "%s\t%f\n",$2, $1*1000}' > spikes-recall.gdf
rm -f spikes-auryn-recall.gdf 

cat spikes-auryn-pattern.gdf | awk '{printf "%s\t%f\n",$2, $1*1000}' > spikes-pattern.gdf
rm -f spikes-auryn-pattern.gdf 

cat spikes-auryn-deaffed.gdf | awk '{printf "%s\t%f\n",$2, $1*1000}' > spikes-deaffed.gdf
rm -f spikes-auryn-deaffed.gdf 

cat spikes-auryn-Stim.gdf | awk '{printf "%s\t%f\n",$2, $1*1000}' > spikes-Stim.gdf
rm -f spikes-auryn-Stim.gdf 

echo "All files combined."
exit 0
