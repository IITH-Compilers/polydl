//command -
// ../mlir-opt cnn.mlir --convert-linalg-to-affine-loops --lower-affine  --convert-std-to-llvm  | ../mlir-translate --mlir-to-llvmir > clang.ll
#map1 = affine_map<(d0,d1) -> (d0 + 2*d1)>
#map2 = affine_map<(d0,d1,d2) -> ((d0 + 2*d1 - d2))>
#map3 = affine_map<(d0)[d1] -> ((d0 floordiv d1))>
#map4 = affine_map<(d0,d1) -> (d0 + d1)>
#map5 = affine_map<(d0)[d1] -> (d0 * d1)>

func @cnn(%nImg: index,%nIfm: index,%nOfm: index,%ifhp: index,%ifwp: index,%ofhp: index,%ofwp: index,%ifh: index,%ifw: index,%ofh: index,%ofw: index,%pad_h: index,%pad_w: index,%pad_h_in: index,
  %pad_w_in: index,%pad_h_out: index,%pad_w_out: index,%kh: index,%kw: index,%stride_h: index,%stride_w: index,
  %input: memref<1x1x56x56x64xf32>, %output: memref<1x4x27x27x64xf32>, %filter: memref<4x1x1x1x64x64xf32>) {

  %GEMM_BLOCK = constant 64 : index
  %STRIDE_H = constant 1 : index
  %STRIDE_W = constant 1 : index

  // Applying convolution

  affine.for %img = 0 to %nImg {
    %nofm_GEMM = affine.apply #map3 (%nOfm )[%GEMM_BLOCK]
    affine.for %ofm_tile = 0 to %nofm_GEMM {
      %nIfm_GEMM = affine.apply #map3 (%nIfm )[%GEMM_BLOCK]
      affine.for %ifm_tile = 0 to %nIfm_GEMM {
        affine.for %oj = 0 to %ofh {
          %ij =  affine.apply #map5 (%oj )[%STRIDE_H]
        affine.for %kj = 0 to %kh {
          affine.for %ki = 0 to %kw {
            affine.for %oi = 0 to %ofw {
              %ii =  affine.apply #map5 (%oi )[%STRIDE_H]
              affine.for %ofm = 0 to %GEMM_BLOCK {
                affine.for %ifm = 0 to %GEMM_BLOCK {
                    %ij_kj =  affine.apply #map4 (%ij ,%kj)
                    %ii_ki =  affine.apply #map4 (%ii ,%ki) 

                    %temp_input = affine.load %input[%img, %ifm_tile, %ij_kj, %ii_ki,%ifm] : memref<1x1x56x56x64xf32>
                    %temp_filter = affine.load %filter[%ofm_tile, %ifm_tile, %kj, %ki,%ifm,%ofm] : memref<4x1x1x1x64x64xf32>
                    %temp_output = affine.load %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<1x4x27x27x64xf32>
                    
                    %temp_mul = mulf %temp_input, %temp_filter : f32
                    %temp_add = addf %temp_output, %temp_mul : f32

                    //call @print_f32(%temp_add): (f32) -> ()

                    affine.store %temp_add, %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<1x4x27x27x64xf32>
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return
}


func @main() {

  //Declaraing constants with 0 and 1
  
  %cf1 = constant 1.00000e+00 : f32

  //Declaraing index variables
  

  %ifw = constant 56 : index
  %ifh = constant 56 : index 
  %nIfm = constant 64 : index
  %nOfm = constant 256 : index
  

  %kw = constant 1 : index
  %kh = constant 1 : index
  %pad_w = constant 0 : index
  %pad_h = constant 0 : index
  %nImg = constant 1 : index

  %stride_w = constant 1 : index
  %stride_h = constant 1 : index
  
  %pad_h_in = constant 0 : index
  %pad_w_in = constant 0 : index
  %pad_h_out = constant 0 : index
  %pad_w_out = constant 0 : index

  %ofh1 = affine.apply #map2 (%ifh ,%pad_h , %kh)
  %ofh = affine.apply #map3 (%ofh1)[%stride_h]
  %ofw1 = affine.apply #map2 (%ifw ,%pad_w , %kw)
  %ofw = affine.apply #map3 (%ofw1)[%stride_w]

  %ifhp =  affine.apply #map1 (%ifh ,%pad_h_in)
  %ifwp =  affine.apply #map1 (%ifw ,%pad_w_in)
  %ofhp =  affine.apply #map1 (%ofh ,%pad_h_out)
  %ofwp =  affine.apply #map1 (%ofw ,%pad_w_out)
  

  // Test code to cast index values to integers and back to index.
  // index2index, index2float , float2index are all invalid operations.

  //%gagan = index_cast %ofw : index to i32
  //%temp_add = addf %cf1, %gagan : f32
  //call @print_f32(%gagan): (f32) -> ()


  //Declaraing input/output images and filter for CNN-Convolution

  %input = alloc() : memref<1x1x56x56x64xf32>
  %output = alloc() : memref<1x4x27x27x64xf32>
  %filter = alloc() : memref<4x1x1x1x64x64xf32>
  %filter1 = alloc() : memref<4x4x4xf32>

  // Using Linear algebra Dialect to fill these matrices.

  linalg.fill(%input, %cf1) : memref<1x1x56x56x64xf32>, f32
  linalg.fill(%output, %cf1) : memref<1x4x27x27x64xf32>, f32
  linalg.fill(%filter, %cf1) : memref<4x1x1x1x64x64xf32>, f32
  linalg.fill(%filter1, %cf1) : memref<4x4x4xf32>, f32


  call @cnn(%nImg,%nIfm,%nOfm,%ifhp,%ifwp,%ofhp,%ofwp,%ifh,%ifw,%ofh,%ofw,%pad_h,%pad_w,%pad_h_in,%pad_w_in,%pad_h_out,%pad_w_out,%kh,%kw,%stride_h,%stride_w,
  %input, %output, %filter) : (index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,
  memref<1x1x56x56x64xf32>, memref<1x4x27x27x64xf32>, memref<4x1x1x1x64x64xf32>) -> ()
  //call @print_memref_3d_f32(%filter1): (memref<4x4x4xf32>) -> ()

  return
}

func @print_memref_3d_f32(memref<4x4x4xf32>)
func @print_f32(f32)
