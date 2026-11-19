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
#include "hipSYCL/compiler/llvm-to-backend/metal/MixedAddressSpaceSelectPass.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

namespace {

constexpr unsigned UnknownAS = ~0u - 1;
constexpr unsigned WildAS = ~0u; // null: compatible with everything

// Cheap, local deduction of the physical address space of a pointer value.
unsigned deduceAS(const Value *V, unsigned Depth = 0) {
  if (Depth > 16 || !V->getType()->isPointerTy())
    return UnknownAS;
  if (isa<ConstantPointerNull>(V) || isa<UndefValue>(V))
    return WildAS;
  if (unsigned AS = V->getType()->getPointerAddressSpace())
    return AS;
  if (isa<AllocaInst>(V))
    return 5;
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V))
    return deduceAS(ASC->getPointerOperand(), Depth + 1);
  if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
    return deduceAS(GEP->getPointerOperand(), Depth + 1);
  if (auto *BC = dyn_cast<BitCastInst>(V))
    return deduceAS(BC->getOperand(0), Depth + 1);
  if (auto *Sel = dyn_cast<SelectInst>(V)) {
    unsigned T = deduceAS(Sel->getTrueValue(), Depth + 1);
    unsigned F = deduceAS(Sel->getFalseValue(), Depth + 1);
    if (T == WildAS) return F;
    if (F == WildAS) return T;
    return T == F ? T : UnknownAS;
  }
  return UnknownAS;
}

SelectInst *findMixedSelect(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Sel = dyn_cast<SelectInst>(&I);
      if (!Sel || !Sel->getType()->isPointerTy())
        continue;
      unsigned T = deduceAS(Sel->getTrueValue());
      unsigned F2 = deduceAS(Sel->getFalseValue());
      if (T == UnknownAS || F2 == UnknownAS || T == WildAS || F2 == WildAS)
        continue;
      if (T != F2)
        return Sel;
    }
  }
  return nullptr;
}

bool versionOnSelect(Function &F, SelectInst *Sel) {
  Value *Cond = Sel->getCondition();
  Value *TrueV = Sel->getTrueValue();
  Value *FalseV = Sel->getFalseValue();

  BasicBlock *Head = Sel->getParent();
  BasicBlock *Tail = Head->splitBasicBlock(Sel->getIterator(), "acpp.as.tail");
  // The splitBasicBlock above created `br Tail` at the end of Head.

  DominatorTree DT(F);
  SmallVector<BasicBlock *, 16> Region;
  SmallPtrSet<BasicBlock *, 16> RegionSet;
  for (BasicBlock &BB : F) {
    if (DT.dominates(Tail, &BB)) {
      Region.push_back(&BB);
      RegionSet.insert(&BB);
    }
  }

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 16> Clones;
  for (BasicBlock *BB : Region) {
    BasicBlock *C = CloneBasicBlock(BB, VMap, ".acpp.as", &F);
    VMap[BB] = C;
    Clones.push_back(C);
  }
  for (BasicBlock *C : Clones)
    for (Instruction &I : *C)
      RemapInstruction(&I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

  // Frontier phis get an incoming edge from the clone as well.
  for (BasicBlock *BB : Region) {
    BasicBlock *CBB = cast<BasicBlock>(VMap[BB]);
    for (BasicBlock *Succ : successors(BB)) {
      if (RegionSet.count(Succ))
        continue;
      for (PHINode &Phi : Succ->phis()) {
        Value *V = Phi.getIncomingValueForBlock(BB);
        auto It = VMap.find(V);
        Phi.addIncoming(It != VMap.end() ? static_cast<Value *>(It->second) : V, CBB);
      }
    }
  }

  auto *SelClone = cast<SelectInst>(VMap[Sel]);
  BasicBlock *CTail = cast<BasicBlock>(VMap[Tail]);

  SelClone->replaceAllUsesWith(FalseV);
  SelClone->eraseFromParent();
  Sel->replaceAllUsesWith(TrueV);
  Sel->eraseFromParent();

  Head->getTerminator()->eraseFromParent();
  BranchInst::Create(Tail, CTail, Cond, Head);
  return true;
}

} // namespace

PreservedAnalyses MixedAddressSpaceSelectPass::run(Function &F, FunctionAnalysisManager &FAM) {
  bool Changed = false;
  // Each transformation duplicates code, so bound the number of iterations.
  for (unsigned Iter = 0; Iter < 8; ++Iter) {
    SelectInst *Sel = findMixedSelect(F);
    if (!Sel)
      break;
    Changed |= versionOnSelect(F, Sel);
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
