set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean firing rate of neurons (Hz)"
#set xtics 0,1,30
set yrange [0:200]

set output "firing-rate-E.png"
set title "Firing rate for E neurons"
plot "firing-rate-E.gdf" with lines lw 4 title "";


set output "firing-rate-I.png"
set title "Firing rate for I neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "";

set output "firing-rate-both.png"
set title "Firing rate for neurons"
plot "firing-rate-I.gdf" with lines lw 4 title "I", "firing-rate-E.gdf" with lines lw 4 title "E";
