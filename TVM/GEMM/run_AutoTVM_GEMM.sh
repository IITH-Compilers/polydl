# set -x
export KMP_AFFINITY=granularity=fine,compact,1,0
OUT=AutoTVM_GEMM_perf.csv
rm ${OUT}

while IFS=, read -r bf1 bf2 factor M N K; do

    # echo "$bf1|$bf2|$factor|$M|$N|$K";
    config="${bf1}_${bf2}_${factor}_${M}_${N}_${K}"
    GFLOPS=`python3 AutoTVM_GEMM.py $bf1 $bf2 $factor $M $N $K &>run_output`;
    GFLOPS_OUT=`echo $GFLOPS | grep GFLOPS | cut -d" " -f4`;
    # echo $config;
    # echo $GFLOPS_OUT
    echo -n "${config},${GFLOPS_OUT}," >> ${OUT}
    echo >> ${OUT}

done < AutoTuningData.csv

