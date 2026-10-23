// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <chrono>

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/correlated_randomness.h"
#include "libspu/mpc/spdzwise/ring_arithmetic.h"
#include "libspu/mpc/spdzwise/state.h"

namespace spu::mpc::spdzwise {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
}  // namespace

NdArrayRef SpdzWiseState::finishBit2A(ring::DaBit mask, const NdArrayRef& bit) {
  auto start = Clock::now();
  // Checking only arithmetic would permit a faulty upstream Boolean circuit
  // to influence this opening. Flush both dependencies before opening.
  verify();
  mula1b_ms_[1] = Ms(start);
  start = Clock::now();
  const NdArrayView<RepShare> b(bit), r(mask.boolean);
  const int64_t blocks = bit.numel() / 64 + (bit.numel() % 64 != 0);
  std::vector<RepShare> packed(blocks);
  // Each block processes 64 bits; express the grain in blocks, not bits.
  yacl::parallel_for(
      0, blocks, (kMinTaskSize + 63) / 64, [&](int64_t begin, int64_t end) {
        for (int64_t block = begin; block < end; ++block) {
          for (int64_t i = 64 * block;
               i < std::min(bit.numel(), 64 * (block + 1)); ++i)
            for (size_t j = 0; j < 2; ++j)
              packed[block][j] |= (b[i][j] ^ r[i][j]) << (i % 64);
        }
      });
  const auto opened = boolean_->open(packed, "bit2a-open");
  // The single-use mask has been verified and its MAC records are independent
  // snapshots. Consume its arithmetic storage for the linear output.
  auto result = std::move(mask.arithmetic);
  auto* out = result.data<ring::AuthShare>();
  // Derive authenticated one using the existing public-constant operation.
  // Work in the full 128-bit ring: FM64 UINT64_MAX is not internal -1.
  NdArrayRef one_plain(makeType<Pub2kTy>(FM64), {});
  one_plain.data<uint64_t>()[0] = 1;
  const auto one = ring_->publicValue(one_plain).data<ring::AuthShare>()[0];
  pforeach(0, bit.numel(), [&](int64_t i) {
    const bool flip = (opened[i / 64] >> (i % 64)) & 1;
    for (size_t j = 0; j < 4; ++j)
      out[i][j] = flip ? one[j] - out[i][j] : out[i][j];
  });
  mula1b_ms_[2] = Ms(start);
  return result;
}

NdArrayRef SpdzWiseState::bit2a(const NdArrayRef& bit) {
  const auto start = Clock::now();
  auto mask = ring_->randomDaBit(bit.shape());
  mula1b_ms_[0] = Ms(start);
  return finishBit2A(std::move(mask), bit);
}

NdArrayRef SpdzWiseState::mulA1B(const NdArrayRef& x, const NdArrayRef& bit) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing(), "MulA1B requires RING");
    validateInternal(x);
    validateBoolean(bit);
    SPU_ENFORCE(bit.eltype().as<BShrTy>()->nbits() == 1,
                "MulA1B requires a one-bit Boolean share");
    SPU_ENFORCE(x.shape() == bit.shape(), "MulA1B shape mismatch");
    mula1b_ms_ = {};
    if (!x.numel()) return NdArrayRef(x.eltype(), x.shape());
    maybeCheck();
    const auto b = bit2a(bit);
    const auto start = Clock::now();
    auto out = multiply(x, b);
    mula1b_ms_[3] = Ms(start);
    return out;
  } catch (...) {
    abort();
    throw;
  }
}

void ring::MulA1B::evaluate(KernelEvalContext* ctx) const {
  const auto& x = ctx->getParam<Value>(0);
  const auto& b = ctx->getParam<Value>(1);
  // Validation lives inside the session's abort guard, including shape errors.
  ctx->pushOutput(Value(proc(ctx, x.data(), b.data()), x.dtype()));
}
NdArrayRef ring::MulA1B::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& bit) const {
  return ctx->getState<SpdzWiseState>()->mulA1B(x, bit);
}
}  // namespace spu::mpc::spdzwise
