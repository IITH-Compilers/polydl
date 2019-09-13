void padded_conv_fp_stride_1_libxsmm_core3(int nImg, int nIfm, int nOfm, int ifhp, int ifwp, int ofhp, int ofwp, int ifh, int ifw,
	int ofh, int ofw, int pad_h, int pad_w, int pad_h_in, int pad_w_in, int pad_h_out,
	int pad_w_out, int kh, int kw, int stride_h, int stride_w,
	const float pad_gemm_input[nImg][nIfm / 16][ifhp + 2 * pad_h][ifwp + 2 * pad_w][16], float output[nImg][nOfm / 16][ofhp][ofwp][16], const float filter[nOfm / 16][nIfm / 16][kh][kw][16][16])
{
	/* loop counters */
	int img, ofm_tile, ofm, ifm_tile, ifm, oj, oi, ij, ii, kj, ki;

#pragma scop
	for (img = 0; img < nImg; ++img) {
		for (ofm_tile = 0; ofm_tile < nOfm / 16; ++ofm_tile) {
			for (kj = 0; kj < kh; ++kj) {
				for (ki = 0; ki < kw; ++ki) {
					for (ifm_tile = 0; ifm_tile < nIfm / 16; ++ifm_tile) {
						for (oj = 0; oj < ofh; ++oj) {

/*
							fwd_gemm(&filter[ofm_tile][ifm_tile][kj][ki][0][0],
								&pad_gemm_input[img][ifm_tile][oj + kj][ki][0],
								&output[img][ofm_tile][oj][0][0]);
*/
							//GEMM
							for (oi = 0; oi < ofw; ++oi) {
								for (ofm = 0; ofm < 16; ++ofm) {
									for (ifm = 0; ifm < 16; ++ifm) {
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
#pragma endscop
}
