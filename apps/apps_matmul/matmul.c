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

void init_array(float A[N1][N1], float B[N1][N1], float C[N1][N1], float C_ref[N1][N1]) {
	int i, j;

	for (i = 0; i < N1; i++) {
		for (j = 0; j < N1; j++) {
			A[i][j] = (i + j);
			B[i][j] = (float)(i * j);
			C[i][j] = 0.0;
			C_ref[i][j] = 0.0;
		}
	}
}


void matmul_ref(float A[N1][N1], float B[N1][N1], float C[N1][N1]) {
	int i, j, k;
	for (i = 0; i < M1; i++)
		for (j = 0; j < N1; j++)
			for (k = 0; k < K1; k++)
				C[i][j] = beta * C[i][j] + alpha * A[i][k] * B[k][j];

}

void compare(float* ref, float *result, int num) {
	int i;
	float THRESHOLD = 0.0001;
	for (i = 0; i < num; i++) {
		if (abs(ref[i] - result[i]) >= THRESHOLD) {
			printf("ref[%d] = %f, result[%d] = %f\n", i, ref[i], i, result[i]);
		}
	}
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

	float(*A)[K1] = (float*)libxsmm_aligned_malloc(M1*K1 * sizeof(float), 2097152);
	float(*B)[N1] = (float*)libxsmm_aligned_malloc(K1*N1 * sizeof(float), 2097152);
	float(*C)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);
	float(*C_ref)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);

	fwd_gemm = libxsmm_smmdispatch(M1, N1, K1, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	init_array(A, B, C, C_ref);
	matmul_ref(A, B, C);
	matmul_high_performance(A, B, C_ref);
	compare(C_ref, C, M1*N1);
	printf("Comparison DONE\n");

	init_array(A, B, C, C_ref);

	IF_TIME(t_start = rtclock());

	for (t = 0; t < NUM_ITERS; t++) {
#pragma scop
		matmul_high_performance(A, B, C);
#pragma endscop
	}

	IF_TIME(t_end = rtclock());
	IF_TIME(fprintf(stdout, "%0.6lfs\n", t_end - t_start));
	IF_TIME(fprintf(stdout, "%0.2lf GFLOPS\n",
		NUM_ITERS * 2.0 * M1 * N1 * K1 / (t_end - t_start) / 1E9));

	fprintf(stdout, "%0.6f, %0.6f\n", C[0][0], C[M1 - 1][N1 - 1]);

	libxsmm_free(A);
	libxsmm_free(B);
	libxsmm_free(C);
	libxsmm_free(C_ref);

	return 0;
}