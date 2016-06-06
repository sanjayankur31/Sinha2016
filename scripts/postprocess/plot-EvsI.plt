set term pngcairo font "OpenSans, 28" size 1028,1028
set xlabel "Mean I (Hz)"
set ylabel "Mean E (Hz)"
set title "Mean E vs Mean I"
set yrange [0:200]
set xrange [0:200]
set ytics border nomirror 20
set xtics border nomirror

set output "EvsI.png"
plot x with lines lw 3, "< join firing-rate-E.gdf firing-rate-I.gdf" using 2:3 with lines lw 6;

