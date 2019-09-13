#define T_ofm_tile 4
#define T_ifm_tile 1
#define T_oj 4
#define T_oi 28
/* With libxsmm parameters*/
void padded_conv_fp_stride_1_tiled(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / 16][ifhp + 2 * pad_h][ifwp + 2 * pad_w][16], float output[nImg][nOfm / 16][ofhp][ofwp][16], const float filter[nOfm / 16][nIfm / 16][kh][kw][16][16])
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
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki;
	int t_ofm_tile, t_ifm_tile, t_oj, t_oi;

#pragma scop
	for (img = 0; img < nImg; ++img) {
		for (t_ofm_tile = 0; t_ofm_tile < nOfm / 16; t_ofm_tile += T_ofm_tile) {
			for (t_oj = 0; t_oj < ofh; t_oj += T_oj) {
				for (oj = t_oj; oj < min(ofh, t_oj + T_oj); ++oj) {
					for (t_oi = 0; t_oi < ofw; t_oi += T_oi) {
						for (ofm_tile = t_ofm_tile; ofm_tile < min(nOfm / 16, t_ofm_tile + T_ofm_tile); ++ofm_tile) {
							for (t_ifm_tile = 0; t_ifm_tile < nIfm / 16; t_ifm_tile += T_ifm_tile) {
								for (ifm_tile = t_ifm_tile; ifm_tile < min(nIfm / 16, t_ifm_tile + T_ifm_tile); ++ifm_tile) {
									for (kj = 0; kj < kh; ++kj) {
										for (ki = 0; ki < kw; ++ki) {

											
											// GEMM
											// min(ofw, t_oi + T_oi) is simplified to t_oi + T_oi because T_oi divides ofw.
											for (oi = t_oi; oi < t_oi + T_oi; ++oi) {
												for (ofm = 0; ofm < 16; ++ofm) {
													for (ifm = 0; ifm < 16; ++ifm) {
														output[img][ofm_tile][oj][oi][ofm] +=
															filter[ofm_tile][ifm_tile][kj][ki][ifm][ofm] * pad_gemm_input[img][ifm_tile][oj + kj][oi + ki][ifm];
													}
												}
											}
											/*

											fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
												&pad_gemm_input[img][ifm_tile][oj + kj][t_oi + ki][0],
												&output[img][ofm_tile][oj][t_oi][0]);
*/
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
