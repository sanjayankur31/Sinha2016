set term pngcairo font "OpenSans, 28" size 1920,1028

set output "raster-I.png"
set title "Raster plot for I neurons"
plot "spikes-I.gdf" with points ps 1

set output "raster-E.png"
set title "Raster plot for E neurons"
plot "spikes-E.gdf" with points ps 1
