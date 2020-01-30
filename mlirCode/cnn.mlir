
func @main() {

  //Declaraing input/output images and filter for CNN-Convolution

  %input = alloc() : memref<4x4x4x4xf32>
  %output = alloc() : memref<4x4x4x4xf32>
  %filter = alloc() : memref<4x4x4x4xf32>
  %filter1 = alloc() : memref<4x4x4xf32>

  //Declaraing constants with 0 and 1
  
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 0.00000e+00 : f32

  //Declaraing index variables

  %nImg = constant 4 : index
  %nOfm = constant 4 : index
  %nIfm = constant 4 : index
  %ofh = constant 4 : index
  %ofw = constant 4 : index
  %kh = constant 4 : index
  %kw = constant 4 : index

  // Using Linear algebra Dialect to fill these matrices.

  linalg.fill(%input, %cf2) : memref<4x4x4x4xf32>, f32
  linalg.fill(%output, %cf2) : memref<4x4x4x4xf32>, f32
  linalg.fill(%filter, %cf2) : memref<4x4x4x4xf32>, f32
  linalg.fill(%filter1, %cf1) : memref<4x4x4xf32>, f32

  // Applying convolution

  affine.for %img = 0 to %nImg {
    affine.for %ofm = 0 to %nOfm {
      affine.for %ifm = 0 to %nIfm {
        affine.for %oj = 0 to %ofh {
          affine.for %oi = 0 to %ofw {
            affine.for %kj = 0 to %kh {
              affine.for %ki = 0 to %kw {
                %temp_input = affine.load %input[%img, %ifm, %oj, %oi] : memref<4x4x4x4xf32>
                %temp_filter = affine.load %filter[%ofm, %ifm, %kj, %ki] : memref<4x4x4x4xf32>
                %temp_output = affine.load %output[%img, %ofm, %oj, %oi] : memref<4x4x4x4xf32>

                %temp_mul = mulf %temp_input, %temp_filter : f32
                %temp_add = addf %temp_output, %temp_mul : f32

                affine.store %temp_add, %output[%img, %ofm, %oj, %oi] : memref<4x4x4x4xf32>
              }
            }
          }
        }
      }
    }
  }
  call @print_memref_3d_f32(%filter1): (memref<4x4x4xf32>) -> ()

  return
}
func @print_memref_3d_f32(memref<4x4x4xf32>)
