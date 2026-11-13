/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SSCP_METAL_MIXED_ADDRESS_SPACE_SELECT_PASS_HPP
#define HIPSYCL_SSCP_METAL_MIXED_ADDRESS_SPACE_SELECT_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// MSL has no generic address space, so `select` of pointers that point into
// different physical address spaces (e.g. threadgroup vs. device scratch) cannot
// be expressed. This pass versions the code dominated by such a select: it
// branches on the select condition and duplicates the dominated region, so that
// in each copy the select is replaced by the corresponding operand.
class MixedAddressSpaceSelectPass
  : public llvm::PassInfoMixin<MixedAddressSpaceSelectPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace compiler
} // namespace hipsycl

#endif
