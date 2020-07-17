
//../llvm-project/build/bin/mlir-opt  -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm  test2.mlir | ../llvm-project/build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=../llvm-project/build/lib/libmlir_runner_utils.so,../llvm-project/build/lib/libmlir_c_runner_utils.so


// RUN: mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm %s | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | FileCheck %s

func @main() {
  %A = alloc() : memref<2048x2048xf32>
  %B = alloc() : memref<2048x2048xf32>
  %C = alloc() : memref<2048x2048xf32>
  call @print_open(): () -> ()
  call @print_close(): () -> ()
  %cf1 = constant 1.00000e+00 : f32

  %ci0 = constant 0 : index
  %ci1 = constant 1 : index
  %ci2 = constant 2 : index

  linalg.fill(%A, %cf1) : memref<2048x2048xf32>, f32
  linalg.fill(%B, %cf1) : memref<2048x2048xf32>, f32
  linalg.fill(%C, %cf1) : memref<2048x2048xf32>, f32

  %pA = memref_cast %A : memref<2048x2048xf32> to memref<*xf32>
  %pB = memref_cast %B : memref<2048x2048xf32> to memref<*xf32>
  %pC = memref_cast %C : memref<2048x2048xf32> to memref<*xf32>
  %reps = constant 1 : index

  %t_start = call @rtclock() : () -> f64
  // affine.for %arg0 = 0 to 1 {
   // call @sgemm_naive(%A, %B, %C) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()
  call @polydl_matmul_f32(%pA, %pB, %pC) : (memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
 // }
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64

  // call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
  call @print_memref_f32_polydl(%pC) : (memref<*xf32>) -> ()

  %M = dim %C, %ci0 : memref<2048x2048xf32>
  %N = dim %C, %ci1 : memref<2048x2048xf32>
  %K = dim %A, %ci1 : memref<2048x2048xf32>

  %f1 = muli %M, %N : index
  %f2 = muli %f1, %K : index

  // 2*M*N*K.
  %c2 = constant 2 : index
  %f3 = muli %c2, %f2 : index
  %num_flops = muli %reps, %f3 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()

  return
}
// CHECK: 2049, 2049, 2049,

func @sgemm_naive(%arg0: memref<2048x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2048x2048xf32>) {
  %c0 = constant 0 : index
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %3 = affine.load %arg0[%arg3, %arg5] : memref<2048x2048xf32>
        %4 = affine.load %arg1[%arg5, %arg4] : memref<2048x2048xf32>
        %5 = affine.load %arg2[%arg3, %arg4] : memref<2048x2048xf32>
        %6 = mulf %3, %4 : f32
        %7 = addf %6, %5 : f32
        affine.store %7, %arg2[%arg3, %arg4] : memref<2048x2048xf32>
      }
    }
  }
  return
}

func @print_flops(f64)
func @rtclock() -> f64
func @print_memref_f32_polydl(memref<*xf32>)
func @print_memref_f32(memref<*xf32>)
func @print_open()
func @print_close() 
func @polydl_matmul_f32(memref<*xf32>, memref<*xf32>, memref<*xf32>)
