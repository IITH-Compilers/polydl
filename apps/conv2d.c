// Some of the code here is borrowed from the LIBXSMM library: https://github.com/hfp/libxsmm/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "padded_conv_fp_stride_1_libxsmm_core_pluto.c"

#define USE_LIBXSMM

#if defined(USE_LIBXSMM)
#include <libxsmm.h>
/* function-pointer to LIBXSMM kernel */
libxsmm_smmfunction fwd_gemm;
#endif

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifndef T_ofm_tile
#define T_ofm_tile 4
#endif // !T_ofm_tile

#ifndef T_ifm_tile
#define T_ifm_tile 1
#endif // !T_ifm_tile

#ifndef T_oj
#define T_oj 4
#endif // !T_oj

#ifndef T_oi
#define T_oi 28
#endif // !T_oi

#define GEMM_BLOCK 64
#define NUM_TRIALS 3

typedef struct {
	double max_rel_err;
	double max_abs_err;
	double l2_rel_err;
	double one_norm_ref;
	double one_norm_test;
} correctness_t;

void copy_NCHW_to_GEMM(int N, int H, int W, int C, const float nchw[N][C][H][W],
	float gemm[N][C / GEMM_BLOCK][H][W][GEMM_BLOCK]);
void copy_GEMM_to_PADDED_GEMM(int N, int H, int W, int C, int pad_h, int pad_w,
	const float gemm[N][C / GEMM_BLOCK][H][W][GEMM_BLOCK], float pad_gemm[N][C / GEMM_BLOCK][H + 2 * pad_h][W + 2 * pad_w][GEMM_BLOCK]);
void compare_buf(float* ref, float* test, long size, correctness_t* norms);

void naive_conv_fp_stride_1(
	int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float input[nImg][nIfm][ifhp][ifwp], float output[nImg][nOfm][ofhp][ofwp], const float filter[nOfm][nIfm][kh][kw])
{
	/* loop counters */
	int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

	/*
	float input[nImg][nIfm][ifhp][ifwp];
	float output[nImg][nOfm][ofhp][ofwp];
	float filter[nOfm][nIfm][kh][kw];
	*/

	for (img = 0; img < nImg; ++img) {
		for (ofm = 0; ofm < nOfm; ++ofm) {
			for (ifm = 0; ifm < nIfm; ++ifm) {
				for (oj = 0; oj < ofh; ++oj) {
					ij = oj - pad_h;
					for (oi = 0; oi < ofw; ++oi) {
						ii = oi - pad_w;
						for (kj = 0; kj < kh; ++kj) {
							if (ij + kj < 0 || ij + kj >= ifh) continue;
							for (ki = 0; ki < kw; ++ki) {
								if (ii + ki < 0 || ii + ki >= ifw) continue;
								output[img][ofm][oj][oi] += input[img][ifm][ij + kj][ii + ki]
									* filter[ofm][ifm][kj][ki];
							}
						}
					}
				}
			}
		}
	}
}


/* With libxsmm parameters*/
void padded_conv_fp_stride_1_tiled_loop_order_1(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	// printf("LIBXMM version\n");
	/*
	FWD params...
Fwd_ofw_rb = 28
Fwd_ofh_rb = 1
Pack input = 0
Block oj = 4
Loop order = 1
Blocksifm_blocking = 1
Block fwd ofm = 4
Block fwd ifm = 1
Avoid rim fmas = 0
Ofm parallelization = 0
Shuffle filter accesses = 0
Avoid acc load = 1
Fwd GEMM flags = 640
	*/

	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;
	int t_ofm_tile, t_ifm_tile, t_oj, t_oi;

#pragma scop
	for (i = 0; i < iters; i++) {
#pragma omp parallel for private(img, t_ofm_tile, t_oj, oj, t_oi, ofm_tile, t_ifm_tile, ifm_tile, kj, ki)
		for (img = 0; img < nImg; ++img) {
			for (t_ofm_tile = 0; t_ofm_tile < nOfm / GEMM_BLOCK; t_ofm_tile += T_ofm_tile) {
				for (t_oj = 0; t_oj < ofh; t_oj += T_oj) {
					for (oj = t_oj; oj < min(ofh, t_oj + T_oj); ++oj) {
						for (t_oi = 0; t_oi < ofw; t_oi += T_oi) {
							for (ofm_tile = t_ofm_tile; ofm_tile < min(nOfm / GEMM_BLOCK, t_ofm_tile + T_ofm_tile); ++ofm_tile) {
								for (t_ifm_tile = 0; t_ifm_tile < nIfm / GEMM_BLOCK; t_ifm_tile += T_ifm_tile) {
									for (ifm_tile = t_ifm_tile; ifm_tile < min(nIfm / GEMM_BLOCK, t_ifm_tile + T_ifm_tile); ++ifm_tile) {
										for (kj = 0; kj < kh; ++kj) {
											for (ki = 0; ki < kw; ++ki) {

												/*
												// GEMM
												// min(ofw, t_oi + T_oi) is simplified to t_oi + T_oi because T_oi divides ofw.
												for (oi = t_oi; oi < t_oi + T_oi; ++oi) {
													for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
														for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
															output[img][ofm_tile][oj][oi][ofm] +=
																filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
														}
													}
												}
												*/

												fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
													&pad_gemm_input[img][ifm_tile][oj + kj][t_oi + ki][0],
													&output[img][ofm_tile][oj][t_oi][0]);

											}
										}
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

/* With libxsmm parameters*/
void padded_conv_fp_stride_1_tiled_loop_order_0(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	// printf("LIBXMM version: padded_conv_fp_stride_1_tiled_loop_order_0\n");
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;
	int t_ofm_tile, t_ifm_tile, t_oj, t_oi;

#pragma scop
	for (i = 0; i < iters; i++) {
#pragma omp parallel for private(img, t_ofm_tile, t_oj, oj, t_oi, ofm_tile, t_ifm_tile, ifm_tile, kj, ki)
		for (img = 0; img < nImg; ++img) {
			for (t_ofm_tile = 0; t_ofm_tile < nOfm / GEMM_BLOCK; t_ofm_tile += T_ofm_tile) {
				for (t_ifm_tile = 0; t_ifm_tile < nIfm / GEMM_BLOCK; t_ifm_tile += T_ifm_tile) {
					for (t_oj = 0; t_oj < ofh; t_oj += T_oj) {
						for (ofm_tile = t_ofm_tile; ofm_tile < min(nOfm / GEMM_BLOCK, t_ofm_tile + T_ofm_tile); ++ofm_tile) {
							for (ifm_tile = t_ifm_tile; ifm_tile < min(nIfm / GEMM_BLOCK, t_ifm_tile + T_ifm_tile); ++ifm_tile) {
								for (oj = t_oj; oj < min(ofh, t_oj + T_oj); ++oj) {
									for (t_oi = 0; t_oi < ofw; t_oi += T_oi) {
										for (kj = 0; kj < kh; ++kj) {
											for (ki = 0; ki < kw; ++ki) {


												// GEMM
												/*
												// min(ofw, t_oi + T_oi) is simplified to t_oi + T_oi because T_oi divides ofw.
												for (oi = t_oi; oi < t_oi + T_oi; ++oi) {
													for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
														for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
															output[img][ofm_tile][oj][oi][ofm] +=
																filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
														}
													}
												}
												*/

												fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
													&pad_gemm_input[img][ifm_tile][oj + kj][t_oi + ki][0],
													&output[img][ofm_tile][oj][t_oi][0]);

											}
										}
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

void padded_conv_fp_stride_1_libxsmm_core(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
	for (i = 0; i < iters; i++) {
#pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki)
		for (img = 0; img < nImg; ++img) {
			// printf("thread id = %d\n", omp_get_thread_num());
			// #pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki)
			for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
				for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
					for (oj = 0; oj < ofh; ++oj) {
						for (kj = 0; kj < kh; ++kj) {
							for (ki = 0; ki < kw; ++ki) {


								fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
									&pad_gemm_input[img][ifm_tile][oj + kj][ki][0],
									&output[img][ofm_tile][oj][0][0]);


								//GEMM
							/*
							for (oi = 0; oi < ofw; ++oi) {
								for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
									for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
										output[img][ofm_tile][oj][oi][ofm] +=
											filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
									}
								}
							}
							*/

							}
						}
					}
				}
			}
		}
	}
#pragma endscop
}

void padded_conv_fp_stride_1_libxsmm_core2(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
	for (i = 0; i < iters; i++) {
#pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki)
		for (img = 0; img < nImg; ++img) {
			for (oj = 0; oj < ofh; ++oj) {
				for (kj = 0; kj < kh; ++kj) {
					for (ki = 0; ki < kw; ++ki) {
						for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
							for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {

								fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0] /*A*/,
									&pad_gemm_input[img][ifm_tile][oj + kj][ki][0] /*B*/,
									&output[img][ofm_tile][oj][0][0] /*C*/);

								//GEMM
								/**
								for (oi = 0; oi < ofw; ++oi) {
									for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
										for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
											output[img][ofm_tile][oj][oi][ofm] +=
												filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
										}
									}
								}

								*/
							}
						}
					}
				}
			}
		}
	}
#pragma endscop
}


void padded_conv_fp_stride_1_libxsmm_core3(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
	for (i = 0; i < iters; i++) {
#pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki)
		for (img = 0; img < nImg; ++img) {
			for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
				for (kj = 0; kj < kh; ++kj) {
					for (ki = 0; ki < kw; ++ki) {
						for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
							for (oj = 0; oj < ofh; ++oj) {

								fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0] /*A*/,
									&pad_gemm_input[img][ifm_tile][oj + kj][ki][0] /*B*/,
									&output[img][ofm_tile][oj][0][0] /*C*/);

								//GEMM
								/**
								for (oi = 0; oi < ofw; ++oi) {
									for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
										for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
											output[img][ofm_tile][oj][oi][ofm] +=
												filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
										}
									}
								}

								*/
							}
						}
					}
				}
			}
		}
	}
#pragma endscop
}

void padded_conv_fp_stride_1_libxsmm_core4(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
	for (i = 0; i < iters; i++) {
#pragma omp parallel for private(ofm_tile, ifm_tile, oj, kj, ki)
		for (img = 0; img < nImg; ++img) {
			for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
				for (kj = 0; kj < kh; ++kj) {
					for (ki = 0; ki < kw; ++ki) {
						for (oj = 0; oj < ofh; ++oj) {
							for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {

								fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0] /*A*/,
									&pad_gemm_input[img][ifm_tile][oj + kj][ki][0] /*B*/,
									&output[img][ofm_tile][oj][0][0] /*C*/);

								//GEMM
								/**
								for (oi = 0; oi < ofw; ++oi) {
									for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
										for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
											output[img][ofm_tile][oj][oi][ofm] +=
												filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
										}
									}
								}

								*/
							}
						}
					}
				}
			}
		}
	}
#pragma endscop
}

void padded_conv_fp_stride_1_core(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int iters)
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki, i;

#pragma scop
	for (i = 0; i < iters; i++) {
		for (img = 0; img < nImg; ++img) {
			for (ofm_tile = 0; ofm_tile < nOfm / GEMM_BLOCK; ++ofm_tile) {
				for (ifm_tile = 0; ifm_tile < nIfm / GEMM_BLOCK; ++ifm_tile) {
					for (oj = 0; oj < ofh; ++oj) {
						for (kj = 0; kj < kh; ++kj) {
							for (ki = 0; ki < kw; ++ki) {
								//GEMM
								for (oi = 0; oi < ofw; ++oi) {
									for (ofm = 0; ofm < GEMM_BLOCK; ++ofm) {
										for (ifm = 0; ifm < GEMM_BLOCK; ++ifm) {
											output[img][ofm_tile][oj][oi][ofm] +=
												filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
										}
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



void init_buf(float* buf, long size)
{
	int i;
	for (i = 0; i < size; ++i) {
		buf[i] = drand48();
	}
}

void zero_buf(float* buf, long size) {
	int i;
	for (i = 0; i < size; ++i) {
		buf[i] = 0.0f;
	}
}

double padded_conv_fp_stride_1(
	int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float input[nImg][nIfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK], float output[nImg][nOfm / GEMM_BLOCK][ofhp][ofwp][GEMM_BLOCK], const float filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK], int version, int iters)
{
	unsigned long long l_start, l_end;
	double l_total = 0.0;
	/* declare a physcial padded buffer */

	/*
	float pad_gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK];
	*/
	float(*pad_gemm_input)[nIfm / GEMM_BLOCK][ifhp + 2 * pad_h][ifwp + 2 * pad_w][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nImg*nIfm*(ifhp + 2 * pad_h)*(ifwp + 2 * pad_w) * sizeof(float), 2097152);
	// printf("pad_gemm_input = %p\n", pad_gemm_input);
	zero_buf(&pad_gemm_input[0][0][0][0][0], (nImg)*(nIfm / GEMM_BLOCK)*(ifhp + 2 * pad_h)*(ifwp + 2 * pad_w) * GEMM_BLOCK);



	// printf("Calling copy_GEMM_to_PADDED_GEMM\n");
	copy_GEMM_to_PADDED_GEMM(nImg, ifhp, ifwp, nIfm, pad_h, pad_w, input, pad_gemm_input);

	if (version == 0) {
		// printf("padded_conv_fp_stride_1_core\n");
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_tiled_loop_order_0(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 1) {
		// printf("padded_conv_fp_stride_1_tiled_core\n");
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_tiled_loop_order_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 2) {
		// printf("padded_conv_fp_stride_1_libxsmm_core\n");
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_libxsmm_core(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 3) {
		// printf("padded_conv_fp_stride_1_libxsmm_core\n");
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_libxsmm_core2(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 4) {
		// printf("padded_conv_fp_stride_1_libxsmm_core\n");
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_libxsmm_core3(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 5) {
		// printf("padded_conv_fp_stride_1_libxsmm_core\n");
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_libxsmm_core4(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 6) {
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_core(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else if (version == 7) {
		l_start = libxsmm_timer_tick();
		padded_conv_fp_stride_1_libxsmm_core_pluto(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter, iters);
		l_end = libxsmm_timer_tick();
	}
	else {
		printf("Incorrect version\n");
		libxsmm_free(pad_gemm_input);
		exit(0);
	}

	libxsmm_free(pad_gemm_input);
	l_total = libxsmm_timer_duration(l_start, l_end);
	return l_total;
}


void copy_GEMM_to_PADDED_GEMM(int N, int H, int W, int C, int pad_h, int pad_w,
	const float gemm[N][C / GEMM_BLOCK][H][W][GEMM_BLOCK], float pad_gemm[N][C / GEMM_BLOCK][H + 2 * pad_h][W + 2 * pad_w][GEMM_BLOCK])
{
	int n, h, w, c1, c2;

	for (n = 0; n < N; n++) {
		for (c1 = 0; c1 < C / GEMM_BLOCK; c1++) {
			for (h = 0; h < H; h++) {
				for (w = 0; w < W; w++) {
					for (c2 = 0; c2 < GEMM_BLOCK; c2++) {
						pad_gemm[n][c1][h + pad_h][w + pad_w][c2] = gemm[n][c1][h][w][c2];
					}
				}
			}
		}
	}
}


void copy_GEMM_to_NCHW(int N, int H, int W, int C,
	const float input[N][C / GEMM_BLOCK][H][W][GEMM_BLOCK], float output[N][C][H][W])
{
	int n, h, w, c1, c2;

	for (n = 0; n < N; n++) {
		for (c1 = 0; c1 < C / GEMM_BLOCK; c1++) {
			for (h = 0; h < H; h++) {
				for (w = 0; w < W; w++) {
					for (c2 = 0; c2 < GEMM_BLOCK; c2++) {
						output[n][c1 * GEMM_BLOCK + c2][h][w] = input[n][c1][h][w][c2];
					}
				}
			}
		}
	}
}

void copy_NCHW_to_GEMM(int N, int H, int W, int C, const float nchw[N][C][H][W],
	float gemm[N][C / GEMM_BLOCK][H][W][GEMM_BLOCK])
{
	int n, h, w, c1, c2;

	for (n = 0; n < N; n++) {
		for (c1 = 0; c1 < C / GEMM_BLOCK; c1++) {
			for (h = 0; h < H; h++) {
				for (w = 0; w < W; w++) {
					for (c2 = 0; c2 < GEMM_BLOCK; c2++) {
						gemm[n][c1][h][w][c2] = nchw[n][c1 * GEMM_BLOCK + c2][h][w];
					}
				}
			}
		}
	}
}

void copy_KCRS_to_GEMM(int R, int S, int C, int K, const float input[K][C][R][S], float output[K / GEMM_BLOCK][C / GEMM_BLOCK][R][S][GEMM_BLOCK][GEMM_BLOCK])
{
	int r, s, c1, c2, k1, k2;

	for (k1 = 0; k1 < K / GEMM_BLOCK; k1++) {
		for (c1 = 0; c1 < C / GEMM_BLOCK; c1++) {
			for (r = 0; r < R; r++) {
				for (s = 0; s < S; s++) {
					for (c2 = 0; c2 < GEMM_BLOCK; c2++) {
						for (k2 = 0; k2 < GEMM_BLOCK; k2++) {
							output[k1][c1][r][s][c2][k2] =
								input[k1 * GEMM_BLOCK + k2][c1 * GEMM_BLOCK + c2][r][s];
						}
					}
				}
			}
		}
	}
}

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

int main(int argc, char **argv) {
	int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
	int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
	int version = 5;
	int check_correctness = 1;

	correctness_t norms_fwd;
	memset(&norms_fwd, 0, sizeof(norms_fwd));

	/* some parameters we can overwrite via cli,
	   default is some inner layer of overfeat */
	int iters = 1;         /* repetitions of benchmark */
	int ifw = 14;           /* input width, "W" */
	int ifh = 18;           /* input height, "H" */
	int nImg = 32;          /* mini-batch size, "N" */
	int nIfm = 256;         /* number of input feature maps, "C" */
	int nOfm = 512;         /* number of output feature maps, "K" */
	int kh = 3;             /* filter height, "R" */
	int kw = 3;             /* filter width, "S" */
	int pad = 2;            /* padding in output */
	int stride = 1;         /* stride when accessing inputs */

	pad_w = pad;
	pad_h = pad;

	unsigned long long l_start, l_end;
	double l_total = 0.0;
	double flops = 0.0;

	/* reading new values from cli */
	int i = 1;
	if (argc > i) iters = atoi(argv[i++]);
	if (argc > i) ifw = atoi(argv[i++]);
	if (argc > i) ifh = atoi(argv[i++]);
	if (argc > i) nIfm = atoi(argv[i++]);
	if (argc > i) nOfm = atoi(argv[i++]);
	if (argc > i) kw = atoi(argv[i++]);
	if (argc > i) kh = atoi(argv[i++]);
	if (argc > i) pad_w = atoi(argv[i++]);
	if (argc > i) pad_h = atoi(argv[i++]);
	if (argc > i) stride = atoi(argv[i++]);
	if (argc > i) nImg = atoi(argv[i++]);
	if (argc > i) version = atoi(argv[i++]);
	if (argc > i) check_correctness = atoi(argv[i++]);

	printf("version = %d\n", version);

	if (stride != 1) {
		printf("A non-unit stride is not supported yet\n");
		exit(0);
	}

	/* apply stride in both dimensions */
	stride_w = stride;
	stride_h = stride;

	pad_h_in = 0;
	pad_w_in = 0;
	pad_h_out = 0;
	pad_w_out = 0;

	/* deriving some values image size */
	ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
	ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
	ifhp = ifh + 2 * pad_h_in;
	ifwp = ifw + 2 * pad_w_in;
	ofhp = ofh + 2 * pad_h_out;
	ofwp = ofw + 2 * pad_w_out;

	/* some empty lines at the beginning */
	printf("\n\n\n");

	/* print some summary */
	printf("##########################################\n");
	printf("#                Setting Up              #\n");
	printf("##########################################\n");
	printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
	printf("PARAMS: ITERS:%d", iters);
	printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
	printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
	printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Input   (1): %10.2f MiB\n", (double)(1 * nIfm*ifhp*ifwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Output  (1): %10.2f MiB\n", (double)(1 * nOfm*ofhp*ofwp * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh * sizeof(float)) / (1024.0*1024.0));

	if ((nIfm % GEMM_BLOCK != 0) || (nOfm % GEMM_BLOCK != 0)) {
		printf("\nThis code only works for ofm/ifm %d!\n\n\n", GEMM_BLOCK);
		return -1;
	}


	/* apply stride in both dimensions */
/* JIT GEMM kernel */
#if defined(USE_LIBXSMM)
	int ldx;
	ldx = GEMM_BLOCK;

	if (version == 0 || version == 1) {
		// LIBXSMM tiled
		if (ofwp % T_oi != 0) {
			printf("The tiling factor %d for oi loop should divide ofwp = %d\n. Exiting\n", T_oi, ofwp);
			return -1;
		}

		fwd_gemm = libxsmm_smmdispatch(GEMM_BLOCK, T_oi, GEMM_BLOCK, NULL, NULL /* &ldx */, NULL, NULL, NULL, NULL, NULL);
	}
	else {
		fwd_gemm = libxsmm_smmdispatch(GEMM_BLOCK, ofwp, GEMM_BLOCK, NULL, /* &ldx */ NULL, NULL, NULL, NULL, NULL, NULL);
	}

#endif

	printf("Allocating data\n");
	/* allocate data */
	float naive_input[nImg][nIfm][ifhp][ifwp];
	float naive_output[nImg][nOfm][ofhp][ofwp];
	float naive_filter[nOfm][nIfm][kh][kw];

	/*
	float gemm_input[nImg][nIfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK] __attribute__((aligned(2097152)));
	float gemm_output[nImg][nOfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK] __attribute__((aligned(2097152)));
	float gemm_filter[nOfm / GEMM_BLOCK][nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK] __attribute__((aligned(2097152)));
	float check_output[nImg][nOfm][ofhp][ofwp] __attribute__((aligned(2097152)));
	*/

	float(*gemm_input)[nIfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nImg*nIfm*ifhp*ifwp * sizeof(float), 2097152);
	float(*gemm_output)[nOfm / GEMM_BLOCK][ifhp][ifwp][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nImg*nOfm*ifhp*ifwp * sizeof(float), 2097152);
	float(*gemm_filter)[nIfm / GEMM_BLOCK][kh][kw][GEMM_BLOCK][GEMM_BLOCK] = (float*)libxsmm_aligned_malloc(nOfm*nIfm*kh*kw * sizeof(float), 2097152);
	float(*check_output)[nImg][nOfm][ofhp][ofwp] = (float*)libxsmm_aligned_malloc(nImg*nOfm*ofhp*ofwp * sizeof(float), 2097152);

	/*
	printf("gemm_input = %p\n", gemm_input);
	printf("gemm_output = %p\n", gemm_output);
	printf("gemm_filter = %p\n", gemm_filter);
	printf("check_output = %p\n", check_output);
	*/
	printf("Initializing data\n");
	/* initialize data */
	srand48(100);
	init_buf(&naive_input[0][0][0][0], nImg*nIfm*ifhp*ifwp);
	init_buf(&gemm_input[0][0][0][0], nImg*nIfm*ifhp*ifwp);
	init_buf(&naive_filter[0][0][0][0], nOfm*nIfm*kh*kw);
	init_buf(&gemm_filter[0][0][0][0], nOfm*nIfm*kh*kw);
	zero_buf(&naive_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);
	zero_buf(&gemm_output[0][0][0][0][0], nImg*nOfm*ofhp*ofwp);
	zero_buf(&check_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);

	copy_NCHW_to_GEMM(nImg, ifhp, ifwp, nIfm, naive_input, gemm_input);
	copy_KCRS_to_GEMM(kh, kw, nIfm, nOfm, naive_filter, gemm_filter);

	clock_t start, end;
	double exec_time;
	flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

	if (check_correctness) {
		printf("##########################################\n");
		printf("#   Correctness - FWD (custom-Storage)   #\n");
		printf("##########################################\n");
		printf("Calling naive_conv_fp_stride_1\n");
		start = clock();

		naive_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, naive_input, naive_output, naive_filter);

		end = clock();
		exec_time = (double)(end - start) / CLOCKS_PER_SEC;
		printf("Total time of naive_conv_fp_stride_1 = %f seconds\n", exec_time);

		printf("Calling padded_conv_fp_stride_1\n");
		padded_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, gemm_input, gemm_output, gemm_filter, version, 1 /*iters*/);

		printf("Calling copy_GEMM_to_NCHW\n");
		copy_GEMM_to_NCHW(nImg, ofhp, ofwp, nOfm, gemm_output, check_output);

		printf("Printing input values\n");
		printf("%f %f %f\n", naive_input[0][0][0][0], naive_input[nImg / 2][nIfm / 2][ifhp / 2][ifwp / 2], naive_input[nImg - 1][nIfm - 1][ifhp - 1][ifwp - 1]);
		printf("%f %f %f\n", gemm_input[0][0][0][0][0], gemm_input[nImg / 2][(nIfm / 2) / GEMM_BLOCK][ifhp / 2][ifwp / 2][(nIfm / 2) % GEMM_BLOCK], gemm_input[nImg - 1][(nIfm - 1) / GEMM_BLOCK][ifhp - 1][ifwp - 1][(nIfm - 1) % GEMM_BLOCK]);
		printf("Printing weight values\n");
		printf("%f %f %f\n", naive_filter[0][0][0][0], naive_filter[nOfm / 2][nIfm / 2][kh / 2][kw / 2], naive_filter[nOfm - 1][nIfm - 1][kh - 1][kw - 1]);
		printf("%f %f %f\n", gemm_filter[0][0][0][0][0][0], gemm_filter[(nOfm / 2) / GEMM_BLOCK][(nIfm / 2) / GEMM_BLOCK][kh / 2][kw / 2][(nOfm / 2) % GEMM_BLOCK][(nIfm / 2) % GEMM_BLOCK], gemm_filter[(nOfm - 1) / GEMM_BLOCK][(nIfm - 1) / GEMM_BLOCK][kh - 1][kw - 1][(nOfm - 1) % GEMM_BLOCK][(nIfm - 1) % GEMM_BLOCK]);
		printf("Printing output values\n");
		printf("%f %f %f\n", naive_output[0][0][0][0], naive_output[nImg / 2][nOfm / 2][ofhp / 2][ofwp / 2], naive_output[nImg - 1][nOfm - 1][ofhp - 1][ofwp - 1]);
		printf("Printing check_output values\n");
		printf("%f %f %f\n", check_output[0][0][0][0], check_output[nImg / 2][nOfm / 2][ofhp / 2][ofwp / 2], check_output[nImg - 1][nOfm - 1][ofhp - 1][ofwp - 1]);
		printf("Printing gemm_output values\n");
		printf("%f %f %f\n", gemm_output[0][0][0][0][0], gemm_output[nImg / 2][(nOfm / 2) / GEMM_BLOCK][ofhp / 2][ofwp / 2][(nOfm / 2) % GEMM_BLOCK], gemm_output[nImg - 1][(nOfm - 1) / GEMM_BLOCK][ofhp - 1][ofwp - 1][(nOfm - 1) % GEMM_BLOCK]);

		/* compare */
		compare_buf(naive_output, check_output, nImg*nOfm*ofhp*ofwp, &norms_fwd);
		printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
		printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
		printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
		printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
		printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);

	}
	else {
		/* Warm up */
		padded_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, gemm_input, gemm_output, gemm_filter, version, 100 /*iters*/);

	}

	printf("##########################################\n");
	printf("#   Performance - FWD (custom-Storage)   #\n");
	printf("##########################################\n");

        int trial;
	double min_l_total = 0.0;
        for (trial = 0; trial < NUM_TRIALS; trial++) {
		l_total = padded_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
		ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
		pad_w_out, kh, kw, stride_h, stride_w, gemm_input, gemm_output, gemm_filter, version, iters);

		if (trial == 0) {
			min_l_total = l_total;
		} else {
			min_l_total = min(min_l_total, l_total);
		}
	}

	l_total = min_l_total;

	printf("Elapsed time of padded_conv_fp_stride_1 = %f seconds\n", l_total);
	printf("GFLOP  = %.5g\n", flops*1e-9 / (double)iters);
	printf("fp time = %.5g\n", ((double)(l_total / iters)));
	printf("GFLOPS =%.5g\n", (flops*1e-9) / l_total);

	libxsmm_free(gemm_input);
	libxsmm_free(gemm_output);
	libxsmm_free(gemm_filter);
	libxsmm_free(check_output);
	return 0;
}

