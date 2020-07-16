func @main() {
  %c1 = constant 10 : index
  %c2 = constant 20 : index
  %A = alloc() : memref<10x20xf32>
  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<10x20xf32>, f32
  affine.for %i = 0 to %c1 {
    affine.for %j = 0 to %c2 { 
      %v1 = muli %i, %c2 : index
      %v2 = addi %v1, %j : index
      %v3 = index_cast %v2 : index to i32
      %v4 = sitofp %v3 : i32 to f32
      affine.store %v4, %A[%i, %j] : memref<10x20xf32>
   }
  }

  %pA = memref_cast %A : memref<10x20xf32> to memref<*xf32>
  call @print_memref_f32_polydl(%pA) : (memref<*xf32>) -> ()
  return
}

func @print_memref_f32_polydl(memref<*xf32>)

