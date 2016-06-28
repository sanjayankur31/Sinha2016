load 'settings.plt'

set term pngcairo font "OpenSans, 28" size 1920,1028
set xlabel "Time (seconds)"
set ylabel "Mean firing rate of neurons per rank (Hz)"
set yrange [0:]

set output "firingrate-pattern-".pat.".png"
set title "Firing rate for pattern ".pat.".png"
plot for [i=0:ranks-1] "00-pattern-spike-count-".i."-pat-".pat.".txt" using 1 with lines lw 6 title "";
