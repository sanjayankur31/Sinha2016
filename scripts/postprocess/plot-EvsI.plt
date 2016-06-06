set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Mean I (Hz)"
set ylabel "Mean E (Hz)"
set yrange [0:200]
set xrange [0:200]
set title ""
set tics border
set size square

set output "EvsI.png"
plot x with lines lw 3 title "", "< join firing-rate-E.gdf firing-rate-I.gdf" every 100 using 2:3 with lines lw 6 title "";

