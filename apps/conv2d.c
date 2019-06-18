// Some of the code here is borrowed from the LIBXSMM library: https://github.com/hfp/libxsmm/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

typedef struct {
	double max_rel_err;
	double max_abs_err;
	double l2_rel_err;
	double one_norm_ref;
	double one_norm_test;
} correctness_t;

void copy_NCHW_to_GEMM(int N, int H, int W, int C, const float nchw[N][C][H][W],
	float gemm[N][C / 16][H][W][16]);
void copy_GEMM_to_PADDED_GEMM(int N, int H, int W, int C, int pad_h, int pad_w,
	const float gemm[N][C / 16][H][W][16], float pad_gemm[N][C / 16][H + 2 * pad_h][W + 2 * pad_w][16]);
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


void padded_conv_fp_stride_1_core(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / 16][ifhp + 2 * pad_h][ifwp + 2 * pad_w][16], float output[nImg][nOfm / 16][ofhp][ofwp][16], const float filter[nOfm / 16][nIfm / 16][kh][kw][16][16])
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki;

#pragma scop
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / 16; ++ofm_tile) {
			for (ifm_tile = 0; ifm_tile < nIfm / 16; ++ifm_tile) {
				for (oj = 0; oj < ofh; ++oj) {
					for (oi = 0; oi < ofw; ++oi) {
						for (kj = 0; kj < kh; ++kj) {
							for (ki = 0; ki < kw; ++ki) {
								for (ofm = 0; ofm < 16; ++ofm) {
									for (ifm = 0; ifm < 16; ++ifm) {
										output[img][ofm_tile][oj][oi][ofm] +=
											filter[ofm_tile][ifm_tile][kj][ki][ofm][ifm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
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

void padded_conv_fp_stride_1_tiled_core(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / 16][ifhp + 2 * pad_h][ifwp + 2 * pad_w][16], float output[nImg][nOfm][ofhp][ofwp], const float filter[nOfm][nIfm][kh][kw])
{
	/* loop counters */
	int img, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki;
	int T_ofm = 16, T_ifm_tile = 16, T_oj = 16, T_oi = 16;
	int t_ofm, t_ifm_tile, t_oj, t_oi;

#pragma scop
	for (img = 0; img < nImg; ++img) {
		// Loops to optimize
		for (t_ofm = 0; t_ofm < nOfm; t_ofm += T_ofm) {
			for (t_ifm_tile = 0; t_ifm_tile < nIfm / 16; t_ifm_tile += T_ifm_tile) {
				for (t_oj = 0; t_oj < ofh; t_oj += T_oj) {
					for (t_oi = 0; t_oi < ofw; t_oi += T_oi) {
						for (oj = t_oj; oj < min(ofh, t_oj + T_oj); ++oj) {
							for (kj = 0; kj < kh; ++kj) {
								for (ki = 0; ki < kw; ++ki) {

									// Batch-reduce-GEMM
									for (ifm_tile = t_ifm_tile;
										ifm_tile < min(nIfm / 16, t_ifm_tile + T_ifm_tile); ++ifm_tile) {
										// GEMM
										for (ofm = t_ofm; ofm < min(nOfm, t_ofm + T_ofm); ++ofm) {
											for (oi = t_oi; oi < min(ofw, t_oi + T_oi); ++oi) {
												for (ifm = 0; ifm < 16; ++ifm) {
													output[img][ofm][oj][oi] +=
														filter[ofm][ifm_tile * 16 + ifm][kj][ki]
														* pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
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
	}
#pragma endscop
}

void padded_conv_fp_stride_1(
	int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float input[nImg][nIfm / 16][ifhp][ifwp][16], float output[nImg][nOfm / 16][ofhp][ofwp][16], const float filter[nOfm / 16][nIfm / 16][kh][kw][16][16], int tiled)
{
	/* declare a physcial padded buffer */
	float pad_gemm_input[nImg][nIfm / 16][ifhp + 2 * pad_h][ifwp + 2 * pad_w][16];
	zero_buf(pad_gemm_input, (nImg)*(nIfm / 16)*(ifhp + 2 * pad_h)*(ifwp + 2 * pad_w) * 16);

	printf("Calling copy_GEMM_to_PADDED_GEMM\n");
	copy_GEMM_to_PADDED_GEMM(nImg, ifhp, ifwp, nIfm, pad_h, pad_w, input, pad_gemm_input);

	printf("padded_conv_fp_stride_1_core\n");
	padded_conv_fp_stride_1_core(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
		ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
		pad_w_out, kh, kw, stride_h, stride_w, pad_gemm_input, output, filter);
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

void copy_GEMM_to_PADDED_GEMM(int N, int H, int W, int C, int pad_h, int pad_w,
	const float gemm[N][C / 16][H][W][16], float pad_gemm[N][C / 16][H + 2 * pad_h][W + 2 * pad_w][16])
{
	int n, h, w, c1, c2;

	for (n = 0; n < N; n++) {
		for (c1 = 0; c1 < C / 16; c1++) {
			for (h = 0; h < H; h++) {
				for (w = 0; w < W; w++) {
					for (c2 = 0; c2 < 16; c2++) {
						pad_gemm[n][c1][h + pad_h][w + pad_w][c2] = gemm[n][c1][h][w][c2];
					}
				}
			}
		}
	}
}


void copy_GEMM_to_NCHW(int N, int H, int W, int C,
	const float input[N][C / 16][H][W][16], float output[N][C][H][W])
{
	int n, h, w, c1, c2;

	for (n = 0; n < N; n++) {
		for (c1 = 0; c1 < C / 16; c1++) {
			for (h = 0; h < H; h++) {
				for (w = 0; w < W; w++) {
					for (c2 = 0; c2 < 16; c2++) {
						output[n][c1 * 16 + c2][h][w] = input[n][c1][h][w][c2];
					}
				}
			}
		}
	}
}

void copy_NCHW_to_GEMM(int N, int H, int W, int C, const float nchw[N][C][H][W],
	float gemm[N][C / 16][H][W][16])
{
	int n, h, w, c1, c2;

	for (n = 0; n < N; n++) {
		for (c1 = 0; c1 < C / 16; c1++) {
			for (h = 0; h < H; h++) {
				for (w = 0; w < W; w++) {
					for (c2 = 0; c2 < 16; c2++) {
						gemm[n][c1][h][w][c2] = nchw[n][c1 * 16 + c2][h][w];
					}
				}
			}
		}
	}
}

void copy_KCRS_to_GEMM(int R, int S, int C, int K, const float input[K][C][R][S], float output[K / 16][C / 16][R][S][16][16])
{
	int r, s, c1, c2, k1, k2;

	for (k1 = 0; k1 < K / 16; k1++) {
		for (c1 = 0; c1 < C / 16; c1++) {
			for (r = 0; r < R; r++) {
				for (s = 0; s < S; s++) {
					for (c2 = 0; c2 < 16; c2++) {
						for (k2 = 0; k2 < 16; k2++) {
							output[k1][c1][r][s][c2][k2] =
								input[k1 * 16 + k2][c1 * 16 + c2][r][s];
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
	int use_tiled_conv2d = 0;

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
	if (argc > i) nImg = atoi(argv[i++]);
	if (argc > i) nIfm = atoi(argv[i++]);
	if (argc > i) nOfm = atoi(argv[i++]);
	if (argc > i) kw = atoi(argv[i++]);
	if (argc > i) kh = atoi(argv[i++]);
	if (argc > i) pad_w = atoi(argv[i++]);
	if (argc > i) pad_h = atoi(argv[i++]);
	if (argc > i) stride = atoi(argv[i++]);

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

	if ((nIfm % 16 != 0) || (nOfm % 16 != 0)) {
		printf("\nThis code only works for ofm/ifm % 16!\n\n\n");
		return -1;
	}

	printf("Allocating data\n");
	/* allocate data */
	float naive_input[nImg][nIfm][ifhp][ifwp];
	float naive_output[nImg][nOfm][ofhp][ofwp];
	float naive_filter[nOfm][nIfm][kh][kw];

	float gemm_input[nImg][nIfm / 16][ifhp][ifwp][16];
	float gemm_output[nImg][nOfm / 16][ifhp][ifwp][16];
	float gemm_filter[nOfm / 16][nIfm / 16][kh][kw][16][16];
	float check_output[nImg][nOfm][ofhp][ofwp];

	printf("Initializing data\n");
	/* initialize data */
	init_buf(&naive_input[0][0][0][0], nImg*nIfm*ifhp*ifwp);
	init_buf(&gemm_input[0][0][0][0], nImg*nIfm*ifhp*ifwp);
	init_buf(&naive_filter[0][0][0][0], nOfm*nIfm*kh*kw);
	init_buf(&gemm_filter[0][0][0][0], nOfm*nIfm*kh*kw);
	zero_buf(&naive_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);
	zero_buf(&gemm_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);
	zero_buf(&check_output[0][0][0][0], nImg*nOfm*ofhp*ofwp);

	printf("##########################################\n");
	printf("#   Correctness - FWD (custom-Storage)   #\n");
	printf("##########################################\n");
	printf("Calling naive_conv_fp_stride_1\n");
	clock_t start = clock();

	naive_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
		ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
		pad_w_out, kh, kw, stride_h, stride_w, naive_input, naive_output, naive_filter);

	clock_t end = clock();
	double exec_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time of naive_conv_fp_stride_1 = %f seconds\n", exec_time);
	flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

	printf("Calling copy_NCHW_to_GEMM\n");
	copy_NCHW_to_GEMM(nImg, ifhp, ifwp, nIfm, naive_input, gemm_input);

	printf("Calling copy_KCRS_to_GEMM\n");
	copy_KCRS_to_GEMM(kh, kw, nIfm, nOfm, naive_filter, gemm_filter);
	printf("Calling padded_conv_fp_stride_1\n");
	padded_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
		ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
		pad_w_out, kh, kw, stride_h, stride_w, gemm_input, gemm_output, gemm_filter, use_tiled_conv2d);

	printf("Calling copy_GEMM_to_NCHW\n");
	copy_GEMM_to_NCHW(nImg, ofhp, ofwp, nOfm, gemm_output, check_output);

	printf("Printing input values\n");
	printf("%f %f %f\n", naive_input[0][0][0][0], naive_input[nImg / 2][nIfm / 2][ifhp / 2][ifwp / 2], naive_input[nImg - 1][nIfm - 1][ifhp - 1][ifwp - 1]);
	printf("%f %f %f\n", gemm_input[0][0][0][0][0], gemm_input[nImg / 2][(nIfm / 2) / 16][ifhp / 2][ifwp / 2][(nIfm / 2) % 16], gemm_input[nImg - 1][(nIfm - 1) / 16][ifhp - 1][ifwp - 1][(nIfm - 1) % 16]);
	printf("Printing weight values\n");
	printf("%f %f %f\n", naive_filter[0][0][0][0], naive_filter[nOfm / 2][nIfm / 2][kh / 2][kw / 2], naive_filter[nOfm - 1][nIfm - 1][kh - 1][kw - 1]);
	printf("%f %f %f\n", gemm_filter[0][0][0][0][0][0], gemm_filter[(nOfm / 2) / 16][(nIfm / 2) / 16][kh / 2][kw / 2][(nOfm / 2) % 16][(nIfm / 2) % 16], gemm_filter[(nOfm - 1) / 16][(nIfm - 1) / 16][kh - 1][kw - 1][(nOfm - 1) % 16][(nIfm - 1) % 16]);
	printf("Printing output values\n");
	printf("%f %f %f\n", naive_output[0][0][0][0], naive_output[nImg / 2][nOfm / 2][ofhp / 2][ofwp / 2], naive_output[nImg - 1][nOfm - 1][ofhp - 1][ofwp - 1]);
	printf("Printing check_output values\n");
	printf("%f %f %f\n", check_output[0][0][0][0], check_output[nImg / 2][nOfm / 2][ofhp / 2][ofwp / 2], check_output[nImg - 1][nOfm - 1][ofhp - 1][ofwp - 1]);
	printf("Printing gemm_output values\n");
	printf("%f %f %f\n", gemm_output[0][0][0][0][0], gemm_output[nImg / 2][(nOfm / 2) / 16][ofhp / 2][ofwp / 2][(nOfm / 2) % 16], gemm_output[nImg - 1][(nOfm - 1) / 16][ofhp - 1][ofwp - 1][(nOfm - 1) % 16]);

	/* compare */
	compare_buf(naive_output, check_output, nImg*nOfm*ofhp*ofwp, &norms_fwd);
	printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
	printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
	printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
	printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
	printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);

	printf("##########################################\n");
	printf("#   Performance - FWD (custom-Storage)   #\n");
	printf("##########################################\n");

	start = clock();
	for (i = 0; i < iters; i++) {
		padded_conv_fp_stride_1(nImg, nIfm, nOfm, ifhp, ifwp, ofhp, ofwp, ifh, ifw,
			ofh, ofw, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out,
			pad_w_out, kh, kw, stride_h, stride_w, gemm_input, gemm_output, gemm_filter, use_tiled_conv2d);
	}

	end = clock();
	exec_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time of padded_conv_fp_stride_1 = %f seconds\n", exec_time);
	printf("GFLOP  = %.5g\n", flops*1e-9 / (double)iters);
	printf("fp time = %.5g\n", ((double)(exec_time / iters)));
	printf("GFLOPS  = %.5g\n", (flops*1e-9) / exec_time);

	return 0;
}
