//===- RunnerUtils.cpp - Utils for MLIR exec on targets with a C++ runtime ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to debug structured MLIR types at
// runtime. Entities in this file may not be compatible with targets without a
// C++ runtime. These may be progressively migrated to CRunnerUtils.cpp over
// time.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RunnerUtils.h"

#ifndef _WIN32
#include <sys/time.h>
#endif // _WIN32

extern "C" void _mlir_ciface_print_memref_vector_4x4xf32(
	StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
	impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_i8(UnrankedMemRefType<int8_t> *M) {
	impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_print_memref_i32(UnrankedMemRefType<int32_t> *M) {
	impl::printMemRef(*M);
}

extern "C" void print_f32_polydl(
	long long int rank, long long int offset,
	long long int size1, long long int size2,
	long long int stride1, long long int stride2,
	void *base);


extern "C" void _mlir_ciface_print_memref_f32_polydl(UnrankedMemRefType<float> *M) {
	std::cout << "Hello from polydl_print_memref_f32()" << std::endl;

	if (M == NULL) {
		std::cout << "The pointer is NULL" << std::endl;
	}
	else {
		std::cout << "The pointer is NOT NULL" << std::endl;
	}

	std::cout << "Rank: " << M->rank << std::endl;

	DynamicMemRefType<float> V = DynamicMemRefType<float>(*M);
	std::cout << " rank = " << V.rank << std::endl;
	std::cout << "base@ = " << reinterpret_cast<void *>(V.data) << " rank = " << V.rank
		<< " offset = " << V.offset;

	if (V.rank != 2) {
		std::cout << "NOT a 2 dimensional array. Returning." << std::endl;
		return;
	}

	auto print = [&](const int64_t *ptr) {
		if (V.rank == 0)
			return;
		std::cout << ptr[0];
		for (int64_t i = 1; i < V.rank; ++i)
			std::cout << ", " << ptr[i];
	};
	std::cout << " sizes = [";
	print(V.sizes);
	std::cout << "] strides = [";
	print(V.strides);
	std::cout << "]";

	print_f32_polydl(V.rank, V.offset, V.sizes[0], V.sizes[1], V.strides[0], V.strides[1],
		reinterpret_cast<void *>(V.data));
}

extern "C" void polydl_lib_matmul_f32(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C);

extern "C" void polydl_matmul_f32(
	int64_t A_rank, void *A_ptr,
	int64_t B_rank, void *B_ptr,
	int64_t C_rank, void *C_ptr,
	int64_t M, int64_t N, int64_t K) {
	// A[M][K], B[K][N], C[M][N]
	UnrankedMemRefType<float> A_descriptor = { A_rank, A_ptr };
	UnrankedMemRefType<float> B_descriptor = { B_rank, B_ptr };
	UnrankedMemRefType<float> C_descriptor = { C_rank, C_ptr };

	DynamicMemRefType<float> A = DynamicMemRefType<float>(A_descriptor);
	DynamicMemRefType<float> B = DynamicMemRefType<float>(B_descriptor);
	DynamicMemRefType<float> C = DynamicMemRefType<float>(C_descriptor);

	if (A.rank != 2 || B.rank != 2 || C.rank != 2) {
		fprintf(stderr, "The three matrices for matrix multiplication are not 2-dimensional.\n");
		return;
	}

	polydl_lib_matmul_f32(M, N, K, A.strides[0], B.strides[0], C.strides[0],
		&A.data[A.offset], &B.data[B.offset], &C.data[C.offset]);

	/*
	if (A.sizes[0] == C.sizes[0] && A.sizes[1] == B.sizes[0] && B.sizes[1] && C.sizes[1]) {
		polydl_lib_matmul_f32(&A.data[A.offset], A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
			&B.data[B.offset], B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
			&C.data[C.offset], C.sizes[0], C.sizes[1], C.strides[0], C.strides[1]);
	}
	else {
		fprintf(stderr, "The matrix sizes are not compatible for multiplication.\n");
		return;
	}
	*/
}


extern "C" void _mlir_ciface_print_memref_f32(UnrankedMemRefType<float> *M) {
	impl::printMemRef(*M);
}

extern "C" void print_memref_i32(int64_t rank, void *ptr) {
	UnrankedMemRefType<int32_t> descriptor = { rank, ptr };
	_mlir_ciface_print_memref_i32(&descriptor);
}

extern "C" void print_memref_f32(int64_t rank, void *ptr) {
	UnrankedMemRefType<float> descriptor = { rank, ptr };
	_mlir_ciface_print_memref_f32(&descriptor);
}

extern "C" void print_memref_f32_polydl(int64_t rank, void *ptr) {
	UnrankedMemRefType<float> descriptor = { rank, ptr };
	_mlir_ciface_print_memref_f32_polydl(&descriptor);
}


extern "C" void
_mlir_ciface_print_memref_0d_f32(StridedMemRefType<float, 0> *M) {
	impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_1d_f32(StridedMemRefType<float, 1> *M) {
	impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_2d_f32(StridedMemRefType<float, 2> *M) {
	impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_3d_f32(StridedMemRefType<float, 3> *M) {
	impl::printMemRef(*M);
}
extern "C" void
_mlir_ciface_print_memref_4d_f32(StridedMemRefType<float, 4> *M) {
	impl::printMemRef(*M);
}

/// Prints GFLOPS rating.
extern "C" void print_flops(double flops) {
	fprintf(stderr, "%lf GFLOPS\n", flops / 1.0E9);
}

/// Returns the number of seconds since Epoch 1970-01-01 00:00:00 +0000 (UTC).
extern "C" double rtclock() {
#ifndef _WIN32
	struct timeval tp;
	int stat = gettimeofday(&tp, NULL);
	if (stat != 0)
		fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
	return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
	fprintf(stderr, "Timing utility not implemented on Windows\n");
	return 0.0;
#endif // _WIN32
}
