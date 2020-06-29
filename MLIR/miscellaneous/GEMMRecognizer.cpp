// mlir/lib/Dialect/Affine/Transforms/GEMMRecognizer.cpp
// mlir-opt  -affine-gemm-recognizer <file> -debug-only="affine-gemm-recognizer"
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile loop nests.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace mlir;

using llvm::dbgs;
#define DEBUG_TYPE "affine-gemm-recognizer"

namespace {

	struct GEMMRecognizer : public GEMMRecognizerBase<GEMMRecognizer> {
		GEMMRecognizer() = default;
		void runOnFunction() override;
	};

} // end anonymous namespace


std::unique_ptr<OperationPass<FuncOp>> mlir::createGEMMRecognizerPass() {
	return std::make_unique<GEMMRecognizer>();
}


void GEMMRecognizer::runOnFunction() {
	LLVM_DEBUG(dbgs() << "Running the GEMM recognizer pass \n");
}
