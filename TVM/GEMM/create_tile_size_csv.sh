#!/bin/bash
M=$1
N=$2
K=$3
FILE=$4

echo M: $M N: $N K: $K
MIN=4
for (( M1_Tile=${MIN}; M1_Tile<=$M; M1_Tile=M1_Tile*4 ))
do 
   if [ `expr $M % ${M1_Tile}` -eq 0 ]
   then

        for (( N1_Tile=${MIN}; N1_Tile<=$N; N1_Tile=N1_Tile*4 ))
        do
          if [ `expr $N % ${N1_Tile}` -eq 0 ]
          then

        	for (( K1_Tile=${MIN}; K1_Tile<=$K; K1_Tile=K1_Tile*4 ))
        	do
          	 if [ `expr $K % ${K1_Tile}` -eq 0 ]
         	 then

                        echo ${M1_Tile},${N1_Tile},${K1_Tile},$M,$N,$K >> $FILE
          	 fi
        	done
          fi
        done
   fi
done
