
ITERS=1000
#xception
 #sh ./run_conv.sh ${ITERS} 299 299   3 32 3 3 0 0 2  xception_1
 sh ./run_conv.sh ${ITERS} 149 149   32 64 3 3 0 0 1  xception_2
 sh ./run_conv.sh ${ITERS} 147 147   64 128 1 1 0 0 1  xception_3
 sh ./run_conv.sh ${ITERS} 147 147   128 128 1 1 0 0 1  xception_4
 sh ./run_conv.sh ${ITERS} 74 74   128 256 1 1 0 0 1  xception_5
 sh ./run_conv.sh ${ITERS} 74 74   256 256 1 1 0 0 1  xception_6
 sh ./run_conv.sh ${ITERS} 37 37   256 728 1 1 0 0 1  xception_7
 sh ./run_conv.sh ${ITERS} 37 37   728 728 1 1 0 0 1  xception_8
 sh ./run_conv.sh ${ITERS} 19 19   728 728 1 1 0 0 1  xception_9
 sh ./run_conv.sh ${ITERS} 19 19   728 1024 1 1 0 0 1  xception_10
 sh ./run_conv.sh ${ITERS} 10 10   1024 1536 1 1 0 0 1  xception_11
 sh ./run_conv.sh ${ITERS} 10 10   1536 2048 1 1 0 0 1  xception_12
