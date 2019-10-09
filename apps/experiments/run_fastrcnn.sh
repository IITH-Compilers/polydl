
ITERS=1000

#sh ./run_conv.sh ${ITERS} 606 756   3 64 7 7 0 0 2 fastrcnn_1 
sh ./run_conv.sh ${ITERS} 150 188   64 64 1 1 0 0 1  fastrcnn_2
sh ./run_conv.sh ${ITERS} 150 188   64 64 3 3 1 1 1  fastrcnn_3
sh ./run_conv.sh ${ITERS} 150 188   64 256 1 1 0 0 1  fastrcnn_4
sh ./run_conv.sh ${ITERS} 150 188   256 64 1 1 0 0 1  fastrcnn_5
sh ./run_conv.sh ${ITERS} 152 190   64 64 3 3 0 0 2  fastrcnn_6
sh ./run_conv.sh ${ITERS} 75 94   64 256 1 1 0 0 1  fastrcnn_7
sh ./run_conv.sh ${ITERS} 75 94   256 128 1 1 0 0 1  fastrcnn_8
sh ./run_conv.sh ${ITERS} 75 94   256 512 1 1 0 0 1  fastrcnn_9
sh ./run_conv.sh ${ITERS} 75 94   128 128 3 3 1 1 1  fastrcnn_10
sh ./run_conv.sh ${ITERS} 75 94   128 512 1 1 0 0 1  fastrcnn_11
sh ./run_conv.sh ${ITERS} 75 94   512 128 1 1 0 0 1  fastrcnn_12
sh ./run_conv.sh ${ITERS} 77 96   128 128 3 3 0 0 2  fastrcnn_13
sh ./run_conv.sh ${ITERS} 38 47   128 512 1 1 0 0 1  fastrcnn_14
sh ./run_conv.sh ${ITERS} 38 47   512 256 1 1 0 0 1  fastrcnn_15
sh ./run_conv.sh ${ITERS} 38 47   256 256 3 3 1 1 1  fastrcnn_16
sh ./run_conv.sh ${ITERS} 38 47   256 1024 1 1 0 0 1  fastrcnn_17
sh ./run_conv.sh ${ITERS} 38 47   512 1024 1 1 0 0 1  fastrcnn_18
sh ./run_conv.sh ${ITERS} 38 47   1024 256 1 1 0 0 1  fastrcnn_19
sh ./run_conv.sh ${ITERS} 38 47   1024 512 3 3 1 1 1  fastrcnn_20
sh ./run_conv.sh ${ITERS} 38 47   512 48 1 1 0 0 1  fastrcnn_21
sh ./run_conv.sh ${ITERS} 38 47   512 24 1 1 0 0 1  fastrcnn_22
sh ./run_conv.sh ${ITERS} 7 7   1024 512 1 1 0 0 1  fastrcnn_23
sh ./run_conv.sh ${ITERS} 7 7   512 512 3 3 1 1 1  fastrcnn_24
sh ./run_conv.sh ${ITERS} 7 7   512 2048 1 1 0 0 1  fastrcnn_25
sh ./run_conv.sh ${ITERS} 7 7   1024 2048 1 1 0 0 1  fastrcnn_26
sh ./run_conv.sh ${ITERS} 7 7   2048 512 1 1 0 0 1  fastrcnn_27
