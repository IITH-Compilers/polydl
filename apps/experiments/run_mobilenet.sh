ITERS=1000
#mobilenet
# sh ./run_conv.sh ${ITERS} 224 224   3 32 3 3 1 1 2  mobilenet_1
 sh ./run_conv.sh ${ITERS} 112 112   32 64 1 1 0 0 1  mobilenet_2
 sh ./run_conv.sh ${ITERS} 56 56   64 128 1 1 0 0 1  mobilenet_3
 sh ./run_conv.sh ${ITERS} 56 56   128 128 1 1 0 0 1  mobilenet_4
 sh ./run_conv.sh ${ITERS} 28 28   128 256 1 1 0 0 1  mobilenet_5
 sh ./run_conv.sh ${ITERS} 28 28   256 256 1 1 0 0 1  mobilenet_6
 sh ./run_conv.sh ${ITERS} 14 14   256 512 1 1 0 0 1  mobilenet_7
 sh ./run_conv.sh ${ITERS} 14 14   512 512 1 1 0 0 1  mobilenet_8
 sh ./run_conv.sh ${ITERS} 7 7   512 1024 1 1 0 0 1  mobilenet_9
 sh ./run_conv.sh ${ITERS} 7 7   1024 1024 1 1 0 0 1  mobilenet_10
# sh ./run_conv.sh ${ITERS} 1 1   1024 5 1 1 0 0 1  mobilenet_11
