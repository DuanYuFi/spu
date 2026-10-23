// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include "libspu/mpc/kernel.h"

namespace spu::mpc::spdzwise::ring {
// Value-only split-3, bit-sliced CSA + ripple-carry addition modulo 2^64.
// Uses the current ABFL session; final verification remains deferred.
class A2B final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "a2b"; }
  void evaluate(KernelEvalContext* ctx) const override;
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};
// Split-3 CSA followed by a logarithmic carry-only tree; no B2A or opening.
class MsbA2B final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "msb_a2b"; }
  void evaluate(KernelEvalContext* ctx) const override;
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};
// One bit uses checked daBit Bit2A. Wider words use an authenticated random
// mask -> split-3 A2B -> Boolean add -> checked open.
class B2A final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "b2a"; }
  void evaluate(KernelEvalContext* ctx) const override;
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};
}  // namespace spu::mpc::spdzwise::ring
