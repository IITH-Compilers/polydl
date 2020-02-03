func @main() {
  
  // Declaring Indexes.
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 4 : index
  %M = constant 5 : index

  %c5 = constant 5 : index

  // Declaring constants
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32
  
  // Declaring operator `range` from LinAlg Dialect
  %r1 = linalg.range %c0:%c2:%c1 : !linalg.range
  %r2 = linalg.range %c0:%c2:%c1 : !linalg.range


  // We can multiply 2 Indexesas well.
  %0 = muli %M, %M : index
  %2 = muli %0, %c2 : index

  //Need to define the total space for buffer which will be converted to
  //multi-dimensional array in view form of arbitrary type.

  %1 = alloc (%2) : memref<?xi8>
  %12 = alloc (%2) : memref<?xi8>

  %3 = view %1[%c0][%M, %M] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>

  //s1 is a slice which is same as view2d with both dimensions as ranges.
  //%s1 = linalg.slice %3[%r1, %r1] : memref<?x?xf32, offset: ?, strides: [?, 1]>, !linalg.range, !linalg.range, memref<?x?xf32, offset: ?, strides: [?, 1]>

  //s2 is a slice which is one rank less than view2d with one dimension as range & other as constant index.
  // Fixing an index is like fixing that corresponding row or coloumn of that matrix.
  %s2 = linalg.slice %3[%r1, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>, !linalg.range, index, memref<?xf32, offset: ?, strides: [1]>
  //%s3 = linalg.slice %3[%c1, %r1] : memref<?x?xf32, offset: ?, strides: [?, 1]>, index, !linalg.range,  memref<?xf32, offset: ?, strides: [1]>

  // this %4 is just to showcase different views on same memref changes the same values in that buffer only.
//  %4 = view %12[%c0][%M, %M] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>

//  linalg.fill(%3, %cf1) : memref<?x?xf32, offset: ?, strides: [?,1]>, f32

  // Instead of filling the whole with one constant we are filling it with the sequence of natural numbers.
  
  %f1 = alloc(%c1) : memref<?xf32>
  affine.store %cf2, %f1[%c0] : memref<?xf32>

 // call @print_f32(%temp_input): (f32) -> ()

  loop.for %arg0 = %c0 to %c5 step %c1 {
        loop.for %arg1 = %c0 to %c5 step %c1 {
          %temp_input = affine.load %f1[%c0] : memref<?xf32>

          store %temp_input, %3[%arg0, %arg1] : memref<?x?xf32, offset: ?, strides: [?,1]>

          %temp_input1 = addf %temp_input ,%cf1 : f32
          affine.store %temp_input1, %f1[%c0] : memref<?xf32>
        }
      }



//  linalg.fill(%4, %cf2) : memref<?x?xf32, offset: ?, strides: [?,1]>, f32

  //call @print_memref_2d_f32(%s1): (memref<?x?xf32, offset: ?, strides: [?,1]>) -> ()
  call @print_memref_1d_f32(%s2): (memref<?xf32, offset: ?, strides: [1]>) -> ()
//  call @print_memref_2d_f32(%4): (memref<?x?xf32, offset: ?, strides: [?,1]>) -> ()

  return
}
func @print_memref_2d_f32(memref<?x?xf32, offset: ?, strides: [?,1]>  )
func @print_memref_1d_f32(memref<?xf32, offset: ?, strides: [1]>  )
func @print_f32(f32)

