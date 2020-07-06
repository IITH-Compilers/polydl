// C += A * B.
func @matmul(%A: memref<2048x2048xf32>, %B: memref<2048x2048xf32>, %C: memref<2048x2048xf32>) {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf32>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf32>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf32>
        %p = mulf %a, %b : f32
        %co = addf %ci, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
  return
}

func @main() {
  %A = alloc() : memref<2048x2048xf32>
  %B = alloc() : memref<2048x2048xf32>
  %C = alloc() : memref<2048x2048xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<2048x2048xf32>, f32
  linalg.fill(%B, %cf1) : memref<2048x2048xf32>, f32
  linalg.fill(%C, %cf1) : memref<2048x2048xf32>, f32

  call @matmul(%A, %B, %C) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()
// call @print_memref_f32(%C): (memref<2048x2048xf32>) -> ()
  call @print_memref_2d_f32(%C): (memref<2048x2048xf32>) -> ()
//  call @print_gagn(): ()->()
  call @print_open_gagan(%cf1): (f32)->()
// call @print_open(): ()->()
// call @print_open_array(%C):  (memref<2048x2048xf32>) -> ()
  return
}

 func @print_memref_2d_f32(memref<2048x2048xf32>)
//func @print_open()
//func @print_memref_f32()
//func @print_flops(f32)
//func @print_gagn()
func @print_open_gagan(f32)
//func @print_open_array(memref<2048x2048xf32>)
