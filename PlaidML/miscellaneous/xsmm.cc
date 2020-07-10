// Copyright 2020 Intel Corporation

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "libxsmm.h" // NOLINT [build/include_subdir]

#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

extern "C" void plaidml_rt_xsmm_gemm_invoke_f32(     //
  size_t aRank, StridedMemRefType<float, 2> *aRef, //
  size_t bRank, StridedMemRefType<float, 2> *bRef, //
  size_t cRank, StridedMemRefType<float, 2> *cRef, //
  uint64_t func_addr) {
  auto aPtr = aRef->data + aRef->offset;
  auto bPtr = bRef->data + bRef->offset;
  auto cPtr = cRef->data + cRef->offset;
  using FunctionPtr = void(*)(const void *, const void *, void *, ...);
  auto *func_ptr = reinterpret_cast<FunctionPtr>(func_addr);
  libxsmm_xmmfunction sgemm;
  sgemm.xmm = func_ptr;

  sgemm.smm(bPtr, aPtr, cPtr);
}

extern "C" void plaidml_rt_xsmm_brgemm_invoke_f32(     //
  size_t aRank, StridedMemRefType<float, 2> *aRef, //
  size_t bRank, StridedMemRefType<float, 2> *bRef, //
  size_t cRank, StridedMemRefType<float, 2> *cRef, //
  uint64_t func_addr) {
  auto aPtr = aRef->data + aRef->offset;
  auto bPtr = bRef->data + bRef->offset;
  auto cPtr = cRef->data + cRef->offset;
  using FunctionPtr = void(*)(const void *, const void *, void *, ...);
  auto *func_ptr = reinterpret_cast<FunctionPtr>(func_addr);
  libxsmm_xmmfunction sbrgemm;
  sbrgemm.xmm = func_ptr;

  unsigned long long l_br = (unsigned long long)1;
  const float *A_ptrs[l_br], *B_ptrs[l_br];
  A_ptrs[0] = aPtr;
  B_ptrs[0] = bPtr;
  sbrgemm.smra(B_ptrs, A_ptrs, cPtr, &l_br);
}

extern "C" void plaidml_rt_xsmm_brgemm_unroll4_invoke_f32(     //
  size_t aRank1, StridedMemRefType<float, 2> *aRef1, //
  size_t aRank2, StridedMemRefType<float, 2> *aRef2, //
  size_t aRank3, StridedMemRefType<float, 2> *aRef3, //
  size_t aRank4, StridedMemRefType<float, 2> *aRef4, //
  size_t bRank1, StridedMemRefType<float, 2> *bRef1, //
  size_t bRank2, StridedMemRefType<float, 2> *bRef2, //
  size_t bRank3, StridedMemRefType<float, 2> *bRef3, //
  size_t bRank4, StridedMemRefType<float, 2> *bRef4, //
  size_t cRank, StridedMemRefType<float, 2> *cRef, //
  uint64_t func_addr) {
  auto aPtr1 = aRef1->data + aRef1->offset;
  auto aPtr2 = aRef2->data + aRef2->offset;
  auto aPtr3 = aRef3->data + aRef3->offset;
  auto aPtr4 = aRef4->data + aRef4->offset;

  auto bPtr1 = bRef1->data + bRef1->offset;
  auto bPtr2 = bRef2->data + bRef2->offset;
  auto bPtr3 = bRef3->data + bRef3->offset;
  auto bPtr4 = bRef4->data + bRef4->offset;

  auto cPtr = cRef->data + cRef->offset;
  using FunctionPtr = void(*)(const void *, const void *, void *, ...);
  auto *func_ptr = reinterpret_cast<FunctionPtr>(func_addr);
  libxsmm_xmmfunction sbrgemm;
  sbrgemm.xmm = func_ptr;

  unsigned long long l_br = (unsigned long long)4;
  const float *A_ptrs[l_br], *B_ptrs[l_br];
  A_ptrs[0] = aPtr1;
  A_ptrs[1] = aPtr2;
  A_ptrs[2] = aPtr3;
  A_ptrs[3] = aPtr4;

  B_ptrs[0] = bPtr1;
  B_ptrs[1] = bPtr2;
  B_ptrs[2] = bPtr3;
  B_ptrs[3] = bPtr4;

  sbrgemm.smra(B_ptrs, A_ptrs, cPtr, &l_br);
}

extern "C" uint64_t plaidml_rt_xsmm_gemm_dispatch_f32(int32_t lda, int32_t ldb,
  int32_t ldc, int32_t m,
  int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  auto sgemm =
    libxsmm_smmdispatch(n_int, m_int, k_int, &ldb_int, &lda_int, &ldc_int,
      /*alpha=*/nullptr, /*beta=*/nullptr,
      /*flags=*/nullptr, /*prefetch=*/nullptr);

  return reinterpret_cast<uint64_t>(sgemm);
}

extern "C" uint64_t plaidml_rt_xsmm_brgemm_dispatch_f32(int32_t lda, int32_t ldb,
  int32_t ldc, int32_t m,
  int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  auto sbrgemm =
    libxsmm_smmdispatch_reducebatch_addr(n_int, m_int, k_int, &ldb_int, &lda_int, &ldc_int,
      /*alpha=*/nullptr, /*beta=*/nullptr,
      /*flags=*/nullptr, /*prefetch=*/nullptr);

  return reinterpret_cast<uint64_t>(sbrgemm);
}

namespace {
  struct Registration {
    Registration() {
      libxsmm_init();

      using pmlc::compiler::registerSymbol;
      registerSymbol("plaidml_rt_xsmm_gemm_invoke_f32",
        reinterpret_cast<void *>(plaidml_rt_xsmm_gemm_invoke_f32));
      registerSymbol("plaidml_rt_xsmm_gemm_dispatch_f32",
        reinterpret_cast<void *>(plaidml_rt_xsmm_gemm_dispatch_f32));

      registerSymbol("plaidml_rt_xsmm_brgemm_invoke_f32",
        reinterpret_cast<void *>(plaidml_rt_xsmm_brgemm_invoke_f32));
      registerSymbol("plaidml_rt_xsmm_brgemm_dispatch_f32",
        reinterpret_cast<void *>(plaidml_rt_xsmm_brgemm_dispatch_f32));
      registerSymbol("plaidml_rt_xsmm_brgemm_unroll4_invoke_f32",
        reinterpret_cast<void *>(plaidml_rt_xsmm_brgemm_unroll4_invoke_f32));
    }
  };
  static Registration reg;
} // namespace
