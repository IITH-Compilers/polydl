ITERS=1000
#AlexNet
# sh ./run_conv.sh ${ITERS} 227 227       3   64 11 11 0 0 4 alexnet_1 
 sh ./run_conv.sh ${ITERS}  27  27      64  192  5  5 2 2 1  alexnet_2
 sh ./run_conv.sh ${ITERS}  13  13     192  384  3  3 1 1 1  alexnet_3
 sh ./run_conv.sh ${ITERS}  13  13     384  256  3  3 1 1 1  alexnet_4
 sh ./run_conv.sh ${ITERS}  13  13     256  256  3  3 1 1 1  alexnet_5
