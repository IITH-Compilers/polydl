#!/bin/bash

set +x
M=2048
N=2048
K=2048
OUTPUT_FILE=perf.csv

echo M: $M N: $N K: $K

for (( M2_Tile=32; M2_Tile<=$M; M2_Tile=M2_Tile*2 ))
do  
   if [ `expr $M % ${M2_Tile}` -eq 0 ]
   then

        for (( N2_Tile=32; N2_Tile<=$N; N2_Tile=N2_Tile*2 ))
        do
          if [ `expr $N % ${N2_Tile}` -eq 0 ]
          then

        	for (( K2_Tile=32; K2_Tile<=$K; K2_Tile=K2_Tile*2 ))
        	do
          	 if [ `expr $K % ${K2_Tile}` -eq 0 ]
         	 then

../llvm-project/build/bin/mlir-opt --affine-loop-tile="tile-sizes=${M2_Tile} tile-sizes=${N2_Tile} tile-sizes=${K2_Tile}" -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm  large_sgemm_naive_codegen_perfectly_nested.mlir | ../llvm-project/build/bin/mlir-cpu-runner -O3 --loop-vectorize -e main -entry-point-result=void -shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so &> run_output
sleep 1
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
echo ${M2_Tile},${N2_Tile},${K2_Tile},$GFLOPS
echo ${M2_Tile},${N2_Tile},${K2_Tile},$GFLOPS >> ${OUTPUT_FILE}
          	 fi
        	done
          fi
        done
   fi
done
