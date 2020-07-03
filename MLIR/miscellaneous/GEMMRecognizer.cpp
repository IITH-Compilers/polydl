//===- LoopTiling.cpp --- Loop tiling pass ------------------------------*-===//
//
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

bool isAGEMMLoopNest(AffineForOp forOp1) {
	Block &body1 = forOp1.region().front();
	if (auto forOp2 = dyn_cast<AffineForOp>(body1.front())) {
		Block &body2 = forOp2.region().front();
		if (auto forOp3 = dyn_cast<AffineForOp>(body2.front())) {
			LLVM_DEBUG(forOp1.getOperation()->print(dbgs() << "The triple nested loop is\n"));

			if (forOp1.getStep() == 1 && forOp2.getStep() == 1 &&
				forOp3.getStep() == 1) {
				LLVM_DEBUG(dbgs() << "All 3 loops have the stride of 1.\n");

				// The last Op will be affine.terminator. Therefore, skipping that.
				LLVM_DEBUG(dbgs() << " num_ops: " <<
					forOp3.getOperation()->getBlock()->getOperations().size() << "\n");

				if (forOp3.getOperation()->getBlock()->getOperations().size() == 2) {
					Block &body3 = forOp3.region().front();
					auto range = llvm::make_range(
						body3.getOperations().begin(),
						std::prev(body3.getOperations().end()));

					int numLoads = 0, numStores = 0, numAdds = 0, numMuls = 0, numOthers = 0;
					for (Operation &op : range) {
						LLVM_DEBUG(op.print(dbgs() << "\nOp:\n"));

						OperationName name = op.getName();
						StringRef nameString = name.getStringRef();
						LLVM_DEBUG(dbgs() << "\n Operation Name: " << nameString);

						if (nameString.contains(".load")) {
							numLoads++;
						}
						else if (nameString.contains(".store")) {
							numStores++;
						}
						else if (nameString.contains(".add")) {
							numAdds++;
						}
						else if (nameString.contains(".mul")) {
							numMuls++;
						}
						else {
							numOthers++;
						}
					}

					if (numLoads == 3 && numStores == 1 && numAdds == 1
						&& numMuls == 1 && numOthers == 0) {
						// Matrix multiplication pattern has been found.
						return true;
					}
				}
			}
		}
	}

	return false;
}

void GEMMRecognizer::runOnFunction() {
	LLVM_DEBUG(dbgs() << "Running the GEMM recognizer pass \n");

	FuncOp f = getFunction();


	f.walk([&](AffineForOp forOp) {
		if (isAGEMMLoopNest(forOp)) {
			LLVM_DEBUG(dbgs() << "GEMM pattern has been FOUND\n");
		}
		else {
			LLVM_DEBUG(dbgs() << "NOT a GEMM pattern \n");
		}
	});


	/*
	for (auto &block : f) {
		for (auto &op : block) {
			if (auto forOp = dyn_cast<AffineForOp>(op)) {
				if (isAGEMMLoopNest(forOp)) {
					LLVM_DEBUG(dbgs() << "GEMM pattern has been FOUND\n");
				}
				else {
					LLVM_DEBUG(dbgs() << "NOT a GEMM pattern \n");
				}
			}
		}
	}
	*/
}