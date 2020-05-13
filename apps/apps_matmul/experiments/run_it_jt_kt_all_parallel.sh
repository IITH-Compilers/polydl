#!/bin/bash

FILE=matmul3.c
ITERS=5
M=$1
N=$2
K=$3

echo M: $M N: $N K: $K

for (( M2_Tile=64; M2_Tile<=$M; M2_Tile=M2_Tile+64 ))
do  
   if [ `expr $M % ${M2_Tile}` -eq 0 ]
   then

        for (( N2_Tile=64; N2_Tile<=$N; N2_Tile=N2_Tile+64 ))
        do
          if [ `expr $N % ${N2_Tile}` -eq 0 ]
          then

        	for (( K2_Tile=64; K2_Tile<=$K; K2_Tile=K2_Tile+64 ))
        	do
          	 if [ `expr $K % ${K2_Tile}` -eq 0 ]
         	 then
			#echo $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64
			let n="( $M / ${M2_Tile} ) % 28"
			if [ $n -eq 0 ]
			then
				echo $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 it2
				sh ./run_matmul.sh $ITERS $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 $FILE it2
			fi

                        let n="( $N / ${N2_Tile} ) % 28"
                        if [ $n -eq 0 ]
                        then
                                echo $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 jt2
                                sh ./run_matmul.sh $ITERS $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 $FILE jt2
                        fi

                        let n="( ${M2_Tile} / 64 ) % 28"
                        if [ $n -eq 0 ]
                        then
                                echo $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 it1
                                sh ./run_matmul.sh $ITERS $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 $FILE it1
                        fi


                        let n="( ${N2_Tile} / 64 ) % 28"
                        if [ $n -eq 0 ]
                        then
                                echo $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 jt1
                                sh ./run_matmul.sh $ITERS $M $N $K ${M2_Tile} ${N2_Tile} ${K2_Tile} 64 64 64 $FILE jt1
                        fi

          	 fi
        	done
          fi
        done
   fi
done
