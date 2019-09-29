CC=gcc -std=c99
CFLAGS=-O3 -I . -I./libxsmm/include
LDFLAGS=-L./libxsmm/lib -lxsmm -lxsmmext -lpthread -lm
MACROFLAGS=

default: lstm

./libxsmm/include/libxsmm.h:
	rm -rf libxsmm/
	git clone https://github.com/hfp/libxsmm.git
	$(MAKE) realclean -C libxsmm
	$(MAKE) BLAS=0 -C libxsmm

lstm: lstm.c ./libxsmm/include/libxsmm.h
	$(CC) $(CFLAGS) $(MACROFLAGS) lstm.c $(LDFLAGS) -o lstm

clean: 
	rm -rf lstm

