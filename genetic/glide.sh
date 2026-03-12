"${SCHRODINGER}/glide" glide.in -OVERWRITE -HOST localhost:1 -TMPLAUNCHDIR -WAIT
awk -F',' '{print $2,$6,$1}' glide.csv | sed 's/"//g' | awk '{
    split($1, a, ":")
    key = a[2]
    if (!(key in min) || $2 < min[key]) {
        min[key] = $2
        val[key] = $3
    }
}
END {
    for (k in min)
        print val[k], min[k]
}' | grep -v score > out
