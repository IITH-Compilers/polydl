#ifndef M1
#define M1 32
#endif // !M1

#ifndef N1
#define N1 32
#endif // !N1

#ifndef K1
#define K1 32
#endif // !K1


#ifndef M2_Tile
#define M2_Tile M1
#endif // !M2_Tile

#ifndef N2_Tile
#define N2_Tile N1
#endif // !N2_Tile

#ifndef K2_Tile
#define K2_Tile K1
#endif // !K2_Tile

#ifndef M1_Tile
#define M1_Tile M2_Tile
#endif // !M1_Tile

#ifndef N1_Tile
#define N1_Tile N2_Tile
#endif // !N1_Tile

#ifndef K1_Tile
#define K1_Tile K2_Tile
#endif // !K1_Tile


#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

void matmul_high_performance(float A[M1][K1], float B[K1][N1], float C[M1][N1])
{
	int it2, jt2, kt2, it1, jt1, kt1, i, j, k;
	// printf("In matmul2\n");
#pragma scop
	// First level of tiling
	for (it2 = 0; it2 < M1; it2 += M2_Tile) {
		for (jt2 = 0; jt2 < N1; jt2 += N2_Tile) {
			for (kt2 = 0; kt2 < K1; kt2 += K2_Tile) {

				// Second level of tiling
				for (it1 = it2; it1 < min(M1, it2 + M2_Tile); it1 += M1_Tile) {
					for (jt1 = jt2; jt1 < min(N1, jt2 + M2_Tile); jt1 += N1_Tile) {
						for (kt1 = kt2; kt1 < min(K1, kt2 + K2_Tile); kt1 += K1_Tile) {

							// Inner most intra-tile loops
							for (i = it1; i < min(M1, it1 + M1_Tile); i++) {
								for (j = jt1; j < min(N1, jt1 + N1_Tile); j++) {
									for (k = kt1; k < min(K1, kt1 + K1_Tile); k++) {
										C[i][j] = C[i][j] + A[i][k] * B[k][j];
									}
								}
							}
						}
					}
				}
			}
		}
	}
#pragma endscop

}

#ifdef USE_LIBXSMM
#include <libxsmm.h>
extern libxsmm_smmfunction fwd_gemm;

void matmul_high_performance_backup(float A[M1][K1], float B[K1][N1], float C[M1][N1])
{
	fwd_gemm(&B[0][0], &A[0][0], &C[0][0]);
}

#endif