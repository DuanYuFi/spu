// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include "libspu/mpc/kernel.h"

namespace spu::mpc::spdzwise::ring {

// MP-SPDZ TruncPrRing(signed=True, k=63) over the application ring Z/2^64.
// Caller guarantees -2^62 <= x < 2^62; this is not a full-width truncation.
class TruncPrRandomBit final : public TruncAKernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }
  void evaluate(KernelEvalContext* ctx) const override;
  Kind kind() const override { return Kind::Dynamic; }
  bool hasMsbError() const override { return false; }
  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const override;
};

// Exact signed floor(x / 2^bits), for full-width int64 inputs and bits in
// [0,63].
class TruncNaive final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }
  void evaluate(KernelEvalContext* ctx) const override;
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const;
};

// Public trunc_a entry point. Add future algorithm selection in proc(), using
// public parameters only. The default computes exact signed floor; no generic
// rounding metadata is needed by the runtime dispatcher.
class TruncASelector final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "trunc_a"; }
  void evaluate(KernelEvalContext* ctx) const override;
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t bits,
                  SignType sign) const;
};

}  // namespace spu::mpc::spdzwise::ring
