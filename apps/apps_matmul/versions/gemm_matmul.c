#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>
#include <libxsmm.h>


#ifndef M1
#define M1 128
#endif // !M1

#ifndef N1
#define N1 128
#endif // !N1

#ifndef K1
#define K1 128
#endif // !K1

#ifndef NUM_ITERS
#define NUM_ITERS 100
#endif // !NUM_ITERS

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
#pragma omp parallel for private(j, k)
    for (i = 0; i < M1; i++)
        for (j = 0; j < N1; j++)
            for (k = 0; k < K1; k++)
                C[i][j] = C[i][j] + A[i][k] * B[k][j];

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

typedef struct {
    double max_rel_err;
    double max_abs_err;
    double l2_rel_err;
    double one_norm_ref;
    double one_norm_test;
} correctness_t;

void compare_buf(float* ref, float* test, long size, correctness_t* norms)
{
    int i;
    double diff, rel_err;

    norms->max_rel_err = 0.;
    norms->max_abs_err = 0.;
    norms->l2_rel_err = 0.;
    norms->one_norm_ref = 0.;
    norms->one_norm_test = 0.;

    for (i = 0; i < size; ++i) {
        norms->one_norm_ref += (double)ref[i];
        norms->one_norm_test += (double)test[i];
        diff = fabs((double)ref[i] - (double)test[i]);
        norms->l2_rel_err += (diff*diff);
        rel_err = 0.0;
        if (diff > 0.0) {
            rel_err = diff / fabs((double)ref[i]);
        }
        if (rel_err > norms->max_rel_err) {
            norms->max_rel_err = rel_err;
#if 0
            printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e) (R:%12.4e)\n", i, ref[i], test[i], diff, rel_err);
#endif
        }
        if (diff > norms->max_abs_err) {
            norms->max_abs_err = diff;
        }
#if 0
        if (diff > 1.0) {
            printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i, ref[i], test[i], diff);
        }
#endif

    }
    norms->l2_rel_err = sqrt(norms->l2_rel_err);
}

void matmul_gemm(float A[M1][K1], float B[K1][N1], float C[M1][N1]) {
    libxsmm_smmfunction kernel = libxsmm_smmdispatch(N1, M1, K1,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    kernel(B, A, C);
}

void tiled_matmul_gemm(float A[M1][K1], float B[K1][N1], float C[M1][N1]) {
    int SIZE = 32;
    int it, jt, kt, i, j, k;

    for (it = 0; it < M1; it += SIZE)
        for (jt = 0; jt < N1; jt += SIZE)
            for (kt = 0; kt < K1; kt += SIZE)
                for (i = it; i < it + SIZE; i++)
                    for (j = jt; j < jt + SIZE; j++)
                        for (k = kt; k < kt + SIZE; k++) {
                            C[i][j] = C[i][j] + A[i][k] * B[k][j];
                        }

}

double tiled_matmul_gemm_baseline(float A[M1][K1], float B[K1][N1], float C[M1][N1],
    int iters) {

    int SIZE = 16;
    int it, jt, kt, i, j, k;
    int lda = K1;
    int ldb = N1;
    int ldc = N1;

    libxsmm_smmfunction kernel = libxsmm_smmdispatch(SIZE, SIZE, SIZE,
        &ldb, &lda, &ldc, NULL, NULL, NULL, NULL);

    unsigned long long l_start, l_end;
    double l_total = 0.0;
    l_start = libxsmm_timer_tick();

    for (i = 0; i < iters; i++) {

#pragma omp parallel for private(jt, kt)
        for (it = 0; it < M1; it += SIZE)
            for (jt = 0; jt < N1; jt += SIZE)
                for (kt = 0; kt < K1; kt += SIZE) {
                    kernel(&B[kt][jt], &A[it][kt], &C[it][jt]);
                }

    }

    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    return l_total;
}

/*
void tiled_matmul_brgemm(float A[M1][K1], float B[K1][N1], float C[M1][N1]) {
    int SIZE = 32;
    int it, jt, kt, i, j, k;
    int lda = K1;
    int ldb = N1;
    int ldc = N1;

    int stride_b = N1 * SIZE * sizeof(float); // B[kt][jt] -- corresponding to jt loop. The distance
    // between B[kt][jt] and B[kt+1][jt], B[kt][0] and B[kt+1][0].
    // N1 * K_Tile_Size * sizeof(float)
    int stride_a = SIZE * sizeof(float); // A[it][kt]. The distance between A[it][kt] and A[it][kt+1].
    // K_Tile_Size * sizeof(float)

    unsigned long long l_br = K1 / SIZE;

    libxsmm_smmfunction_reducebatch_strd  kernel =
        libxsmm_smmdispatch_reducebatch_strd(SIZE, SIZE, SIZE, stride_b, stride_a,
            &ldb, &lda, &ldc, NULL, NULL, NULL, NULL);

    for (it = 0; it < M1; it += SIZE)
        for (jt = 0; jt < N1; jt += SIZE) {
            kt = 0;
            kernel(&B[kt][jt], &A[it][kt], &C[it][jt], &l_br);
        }

}
*/

int main() {
    int i, j, k, t;

    correctness_t norms_fwd;

    // C[M][N] = A[M][K] * B[K][N];
    float(*A)[K1] = (float*)libxsmm_aligned_malloc(M1*K1 * sizeof(float), 2097152);
    float(*B)[N1] = (float*)libxsmm_aligned_malloc(K1*N1 * sizeof(float), 2097152);
    float(*C)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);
    float(*C_ref)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);

    printf("M1 = %d, N1 = %d, K1 = %d, NUM_ITERS = %d\n", M1, N1, K1, NUM_ITERS);
    printf("SIZE A  (MB): %10.2f MB\n", (double)(M1*K1 * sizeof(float)) / (1024.0*1024.0));
    printf("SIZE B  (MB): %10.2f MB\n", (double)(K1*N1 * sizeof(float)) / (1024.0*1024.0));
    printf("SIZE C  (MB): %10.2f MB\n", (double)(M1*N1 * sizeof(float)) / (1024.0*1024.0));

    init_array(A, B, C, C_ref);
    matmul_ref(A, B, C_ref);
    // matmul_gemm(A, B, C);
    tiled_matmul_gemm_baseline(A, B, C, 1);

    /* compare */
    compare_buf(C_ref, C, M1*N1, &norms_fwd);
    printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
    printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
    printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);

    printf("A: %f, %f\n", A[0][0], A[M1 - 1][K1 - 1]);
    printf("B: %f, %f\n", B[0][0], B[K1 - 1][N1 - 1]);
    printf("C_ref: %f, %f, %f\n", C_ref[0][0], C_ref[M1 / 2][N1 / 2], C_ref[M1 - 1][N1 - 1]);
    printf("C: %f, %f, %f\n", C[0][0], C[M1 / 2][N1 / 2], C[M1 - 1][N1 - 1]);

    init_array(A, B, C, C_ref);
    double l_total = tiled_matmul_gemm_baseline(A, B, C, NUM_ITERS);

    double flops = NUM_ITERS * 2.0 * M1 * N1 * K1;
    printf("Real_GFLOPS =%0.2lf",
        (flops*1e-9) / l_total);


    libxsmm_free(A);
    libxsmm_free(B);
    libxsmm_free(C);
    libxsmm_free(C_ref);

    return 0;
}