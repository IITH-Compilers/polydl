To Collect Gflops Number -> sh ../files/collectGFlops.sh;

Run polyscientist to generate 4 CSV variant files like `padded_conv_fp_stride_1_tiled_loop_order_0.cconv_config.txt_ws_stats.csv`;

Then To combine stats of GFlops and polyscientist -> sh combine.sh;

The above step will generate different files like -> `out_N_28.csv`;

Pass these above files to Polyrank script;

Then examine the generated files like `out_N_28.csv_ranks.csv`;
