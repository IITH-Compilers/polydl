#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>
#include <libxsmm.h>

#ifndef M1
#define M1 32
#endif // !M1

#ifndef N1
#define N1 32
#endif // !N1

#ifndef K1
#define K1 32
#endif // !K1

#ifndef M1_Tile
#define M1_Tile M1
#endif // !M1_Tile

#ifndef N1_Tile
#define N1_Tile N1
#endif // !N1_Tile

#ifndef K1_Tile
#define K1_Tile K1
#endif // !K1_Tile


#ifndef alpha
#define alpha 1
#endif // !alpha

#ifndef beta
#define beta 1
#endif // !beta

#ifndef NUM_ITERS
#define NUM_ITERS 1000
#endif // !NUM_ITERS



#define TIME
#ifdef TIME
#define IF_TIME(foo) foo;
#else
#define IF_TIME(foo)
#endif


/* function-pointer to LIBXSMM kernel */
libxsmm_smmfunction fwd_gemm;
extern double matmul_high_performance(float A[M1][K1], float B[K1][N1], float C[M1][N1], int iters);

void init_array(float A[M1][K1], float B[K1][N1], float C[M1][N1], float C_ref[M1][N1]) {
	int i, j;

	for (i = 0; i < M1; i++) {
		for (j = 0; j < K1; j++) {
			A[i][j] = ((float)i + (float)j) / (float)(M1 + K1);
		}
	}

	for (i = 0; i < K1; i++) {
		for (j = 0; j < N1; j++) {
			B[i][j] = ((float)i * (float)j) / (float)(K1 + N1);
		}
	}

	for (i = 0; i < M1; i++) {
		for (j = 0; j < N1; j++) {
			C[i][j] = 0.0;
			C_ref[i][j] = 0.0;
		}
	}
}


void matmul_ref(float A[M1][K1], float B[K1][N1], float C[M1][N1]) {
	int i, j, k;
	for (i = 0; i < M1; i++)
		for (j = 0; j < N1; j++)
			for (k = 0; k < K1; k++)
				C[i][j] = beta * C[i][j] + alpha * A[i][k] * B[k][j];

}

int AreEqual(float* ref, float *result, int num) {
	int i;
	int equal = 1;
	float THRESHOLD = 0.0000001;
	for (i = 0; i < num; i++) {
		if (abs(ref[i] - result[i]) >= THRESHOLD) {
			equal = 0;
			printf("ref[%d] = %f, result[%d] = %f\n", i, ref[i], i, result[i]);
		}
	}

	return equal;
}


void print_array(float C[M1][N1]) {
	int i, j;

	for (i = 0; i < M1; i++) {
		for (j = 0; j < N1; j++) {
			fprintf(stderr, "%lf ", C[i][j]);
			if (j % 80 == 79)
				fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
	}
}

double rtclock() {
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0)
		printf("Error return from gettimeofday: %d", stat);
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
double t_start, t_end;



int main() {
	int i, j, k, t;

	// C[M][N] = A[M][K] * B[K][N];
	float(*A)[K1] = (float*)libxsmm_aligned_malloc(M1*K1 * sizeof(float), 2097152);
	float(*B)[N1] = (float*)libxsmm_aligned_malloc(K1*N1 * sizeof(float), 2097152);
	float(*C)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);
	float(*C_ref)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);

	int N1_val = N1;
	int M1_val = M1;
	int K1_val = K1;

	printf("M1 = %d, N1 = %d, K1 = %d\n", M1, N1, K1);
	printf("M1_Tile = %d, N1_Tile = %d, K1_Tile = %d\n", M1_Tile, N1_Tile, K1_Tile);

	/*
	fwd_gemm = libxsmm_smmdispatch(N1_Tile, M1_Tile, K1_Tile,
		&N1_val, &M1_val, &K1_val, NULL, NULL, NULL, NULL);
		*/

	fwd_gemm = libxsmm_smmdispatch(N1_Tile, M1_Tile, K1_Tile,
		NULL, NULL, NULL, NULL, NULL, NULL, NULL);

	init_array(A, B, C, C_ref);
	matmul_ref(A, B, C);
	matmul_high_performance(A, B, C_ref, 1);

	if (AreEqual(C_ref, C, M1*N1) == 0) {
		printf("Correctness check failed. Exiting\n");
		exit(1);
	}
	else {
		printf("Correctness check passed.\n");
	}


	printf("A: %f, %f\n", A[0][0], A[M1 - 1][K1 - 1]);
	printf("B: %f, %f\n", B[0][0], B[K1 - 1][N1 - 1]);
	printf("C_ref: %f, %f\n", C_ref[0][0], C_ref[M1 - 1][N1 - 1]);
	printf("C: %f, %f\n", C[0][0], C[M1 - 1][N1 - 1]);

	init_array(A, B, C, C_ref);
	double l_total = matmul_high_performance(A, B, C, NUM_ITERS);
	printf("Total time in nano seconds: %f\n", l_total);

	double flops = NUM_ITERS * 2.0 * M1 * N1 * K1;
	printf("%0.2lf GFLOPS\n",
		(flops*1e-9) / l_total);

	libxsmm_free(A);
	libxsmm_free(B);
	libxsmm_free(C);
	libxsmm_free(C_ref);

	return 0;
}