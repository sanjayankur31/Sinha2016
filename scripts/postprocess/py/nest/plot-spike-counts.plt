load 'settings.plt'

set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean firing rate of neurons per rank (Hz)"
set yrange [0:200]

set output "spike-counts-E.png"
set title "Firing rate for E neurons"
plot for [i=0:ranks-1] "00-spike-count-".i.".txt" using 1 with lines lw 6 title "";


set output "spike-counts-I.png"
set title "Firing rate for I neurons"
plot for [j=0:ranks-1] "00-spike-count-".j.".txt" using 2 with lines lw 6 title "";
