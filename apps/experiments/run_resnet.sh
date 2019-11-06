
FILE=$1
ITERS=1000
#Resnet-50
sh ./$FILE $ITERS  224 224 3   64 7 7 3 3 2  resnet_0
sh ./$FILE $ITERS  56  56  64  256 1 1 0 0 1 resnet_1
sh ./$FILE $ITERS  56  56  64  64 1 1 0 0 1 resnet_2
sh ./$FILE $ITERS  56  56  64  64 3 3 1 1 1 resnet_3
sh ./$FILE $ITERS  56  56  256  64 1 1 0 0 1 resnet_4
sh ./$FILE $ITERS  56  56  256  512 1 1 0 0 2 resnet_5
sh ./$FILE $ITERS  56  56  256   128 1 1 0 0 2 resnet_6
sh ./$FILE $ITERS  28  28  128   128 3 3 1 1 1 resnet_7
sh ./$FILE $ITERS  28  28  128   512 1 1 0 0 1 resnet_8
sh ./$FILE $ITERS  28  28  512   128 1 1 0 0 1 resnet_9
sh ./$FILE $ITERS  28  28  512  1024 1 1 0 0 2 resnet_10
sh ./$FILE $ITERS  28  28  512   256 1 1 0 0 2 resnet_11
sh ./$FILE $ITERS  14  14  256   256 3 3 1 1 1 resnet_12
sh ./$FILE $ITERS  14  14  256  1024 1 1 0 0 1 resnet_13
sh ./$FILE $ITERS  14  14  1024   256 1 1 0 0 1 resnet_14
sh ./$FILE $ITERS  14   14   1024  2048 1 1 0 0 2 resnet_15
sh ./$FILE $ITERS  14   14   1024   512 1 1 0 0 2 resnet_16
sh ./$FILE $ITERS  7   7   512   512 3 3 1 1 1 resnet_17
sh ./$FILE $ITERS  7   7   512  2048 1 1 0 0 1 resnet_18
sh ./$FILE $ITERS  7   7   2048   512 1 1 0 0 1 resnet_19
