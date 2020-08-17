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

struct GEMMOperand {
	bool isGEMM;
	int64_t M, N, K;
	Value CMemRef;
	Value AMemRef;
	Value BMemRef;
};
typedef struct GEMMOperand GEMMOperand;

int64_t extractBound(AffineMapAttr boundMap,
	Operation::operand_range boundOperands) {
	AffineMap map = boundMap.getValue();

	if (map.getNumResults() == 1) {
		AffineExpr expr = map.getResult(0);

		// Print constant bound.
		if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
			if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
				return constExpr.getValue();;
			}
		}
	}
	else {
		return -1;
	}
}

bool doArrayIndexesMatch(AffineForOp forOp1, AffineForOp forOp2, AffineLoadOp loadOp) {
	ValueRange mapOperands = loadOp.getMapOperands();
	if (forOp1.getBody()->getArgument(0) == mapOperands[0] &&
		forOp2.getBody()->getArgument(0) == mapOperands[1]) {
		return true;
	}

	return false;
}


GEMMOperand isAGEMMLoopNest(AffineForOp forOp1) {
	GEMMOperand gemmOperand;
	gemmOperand.isGEMM = false;

	Block &body1 = forOp1.region().front();
	if (auto forOp2 = dyn_cast<AffineForOp>(body1.front())) {
		Block &body2 = forOp2.region().front();
		if (auto forOp3 = dyn_cast<AffineForOp>(body2.front())) {
			LLVM_DEBUG(forOp1.getOperation()->print(dbgs() << "The triple nested loop is\n"));

			if (forOp1.getStep() == 1 && forOp2.getStep() == 1 &&
				forOp3.getStep() == 1 &&
				forOp1.hasConstantUpperBound() &&
				forOp2.hasConstantUpperBound() &&
				forOp3.hasConstantUpperBound()) {
				LLVM_DEBUG(dbgs() << "All 3 loops have the stride of 1.\n");

				// The last Op will be affine.terminator. Therefore, skipping that.
				LLVM_DEBUG(dbgs() << " num_ops: " <<
					forOp3.getOperation()->getBlock()->getOperations().size() << "\n");

				if (forOp3.getOperation()->getBlock()->getOperations().size() == 2) {
					Block &body3 = forOp3.region().front();
					AffineLoadOp loadOp1, loadOp2, loadOp3;
					AffineForOp MForOp, NForOp, KForOp;
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
							AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(op);

							if (numLoads == 0) {
								loadOp1 = loadOp;
							}
							else if (numLoads == 1) {
								loadOp2 = loadOp;
							}
							else if (numLoads == 2) {
								loadOp3 = loadOp;
							}

							numLoads++;
						}
						else if (nameString.contains(".store")) {
							AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(op);
							LLVM_DEBUG(storeOp.getMemRef().print(dbgs() << "\ngetMemRef():\n"));
							LLVM_DEBUG(dbgs() << "MemRef\n: " << storeOp.getMemRef());
							Value memRef = storeOp.getMemRef();
							gemmOperand.CMemRef = memRef;
							numStores++;
							ValueRange mapOperands = storeOp.getMapOperands();

							LLVM_DEBUG(dbgs() << "The map operands are: \n");
							for (Value val : mapOperands) {
								LLVM_DEBUG(val.print(dbgs() << "\nValue:\n"));
							}

							// Check if the array index variable is the same as the loop variable
							// C[i][j] : C[M][N]
							bool forOp1Taken = false, forOp2Taken = false, forOp3Taken = false;

							if (forOp1.getBody()->getArgument(0) == mapOperands[0]) {
								gemmOperand.M = forOp1.getConstantUpperBound();
								forOp1Taken = true;
								MForOp = forOp1;
								LLVM_DEBUG(dbgs() << "forOp1 loop variable = store's first index \n");
							}
							else if (forOp2.getBody()->getArgument(0) == mapOperands[0]) {
								gemmOperand.M = forOp2.getConstantUpperBound();
								forOp2Taken = true;
								MForOp = forOp2;
								LLVM_DEBUG(dbgs() << "forOp2 loop variable = store's first index \n");
							}
							else if (forOp3.getBody()->getArgument(0) == mapOperands[0]) {
								gemmOperand.M = forOp1.getConstantUpperBound();
								forOp3Taken = true;
								MForOp = forOp3;
								LLVM_DEBUG(dbgs() << "forOp3 loop variable = store's first index \n");
							}

							// Check if the array index variable is the same as the loop variable
							if (forOp1.getBody()->getArgument(0) == mapOperands[1]) {
								gemmOperand.N = forOp1.getConstantUpperBound();
								forOp1Taken = true;
								NForOp = forOp1;
								LLVM_DEBUG(dbgs() << "forOp1 loop variable = store's second index \n");
							}
							else if (forOp2.getBody()->getArgument(0) == mapOperands[1]) {
								gemmOperand.N = forOp2.getConstantUpperBound();
								forOp2Taken = true;
								NForOp = forOp2;
								LLVM_DEBUG(dbgs() << "forOp2 loop variable = store's second index \n");
							}
							else if (forOp3.getBody()->getArgument(0) == mapOperands[1]) {
								gemmOperand.N = forOp3.getConstantUpperBound();
								forOp3Taken = true;
								NForOp = forOp3;
								LLVM_DEBUG(dbgs() << "forOp3 loop variable = store's second index \n");
							}

							if (forOp1Taken == false) {
								gemmOperand.K = forOp1.getConstantUpperBound();
								KForOp = forOp1;
							}

							if (forOp2Taken == false) {
								gemmOperand.K = forOp2.getConstantUpperBound();
								KForOp = forOp2;
							}

							if (forOp3Taken == false) {
								gemmOperand.K = forOp3.getConstantUpperBound();
								KForOp = forOp3;
							}

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
						gemmOperand.isGEMM = true;
						/*
						gemmOperand.K = extractBound(forOp3.getUpperBoundMapAttr(), forOp3.getUpperBoundOperands());
						*/
						gemmOperand.K = forOp3.getConstantUpperBound();
						LLVM_DEBUG(dbgs() << "K: gemmOperand.K: " << gemmOperand.K);


						// C[M][N] += A[M][K] * B[K][N];
						// We need to figure out which of the three loads is A and which one is B.

						// Let's find A[M][K]
						if (doArrayIndexesMatch(MForOp, KForOp, loadOp1)) {
							gemmOperand.AMemRef = loadOp1.getMemRef();
						}
						else if (doArrayIndexesMatch(MForOp, KForOp, loadOp2)) {
							gemmOperand.AMemRef = loadOp2.getMemRef();
						}
						else if (doArrayIndexesMatch(MForOp, KForOp, loadOp3)) {
							gemmOperand.AMemRef = loadOp3.getMemRef();
						}

						// Let's find B[K][N]
						if (doArrayIndexesMatch(KForOp, NForOp, loadOp1)) {
							gemmOperand.BMemRef = loadOp1.getMemRef();
						}
						else if (doArrayIndexesMatch(KForOp, NForOp, loadOp2)) {
							gemmOperand.BMemRef = loadOp2.getMemRef();
						}
						else if (doArrayIndexesMatch(KForOp, NForOp, loadOp3)) {
							gemmOperand.BMemRef = loadOp3.getMemRef();
						}

						return gemmOperand;
					}
				}
			}
		}
	}

	return gemmOperand;
}


void GEMMRecognizer::runOnFunction() {
	LLVM_DEBUG(dbgs() << "Running the GEMM recognizer pass \n");

	FuncOp f = getFunction();


	f.walk([&](AffineForOp forOp) {
		GEMMOperand gemmOperand = isAGEMMLoopNest(forOp);
		if (gemmOperand.isGEMM) {
			LLVM_DEBUG(dbgs() << "GEMM pattern has been FOUND\n");
			// Now we want to call a matrix multiplication routine here.
			OpBuilder b(forOp);
			SmallVector<Value, 6> ops;
			ops.push_back(gemmOperand.CMemRef);
			ops.push_back(gemmOperand.CMemRef);
			ops.push_back(gemmOperand.CMemRef);

			auto AMemRef = b.create<MemRefCastOp>(forOp.getLoc(),
				gemmOperand.AMemRef, UnrankedMemRefType);
			auto op = b.create<PolyDLGEMMOp>(forOp.getLoc(),
				AMemRef,
				gemmOperand.BMemRef,
				gemmOperand.CMemRef,
				gemmOperand.M,
				gemmOperand.N,
				gemmOperand.K);

			LLVM_DEBUG(dbgs() << "CallOp: " << op);
			forOp.erase();
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