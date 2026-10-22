// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Adapted from MP-SPDZ Compiler/floatingpoint.py::TruncPrRing
// (see MP-SPDZ-LICENSE.txt).

#include "libspu/mpc/spdzwise/ring_truncation.h"

#include <chrono>
#include <limits>

#include "yacl/crypto/hash/blake3.h"
#include "yacl/link/algorithm/allgather.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/spdzwise/correlated_randomness.h"
#include "libspu/mpc/spdzwise/state.h"

namespace spu::mpc::spdzwise::ring {
namespace {
using Clock = std::chrono::steady_clock;

void RequireAll(const std::shared_ptr<yacl::link::Context>& link, bool ok,
                const std::string& context) {
  for (size_t round = 0; round < 2; ++round) {
    const std::string status(1, ok ? '1' : '0');
    const auto all =
        yacl::link::AllGather(link, yacl::ByteContainerView(status),
                              context + "/status/" + std::to_string(round));
    for (const auto& item : all) ok &= std::string_view(item) == "1";
  }
  SPU_ENFORCE(ok, "SPDZWISE ring check failed (truncation)");
}

std::string Hash(std::string_view context, std::string_view bytes) {
  yacl::crypto::Blake3Hash hash;
  hash.Update("SPDZWISE-RING-TRUNC-v1/" + std::to_string(context.size()) + ":" +
              std::string(context));
  hash.Update(bytes);
  const auto digest = hash.CumulativeHash();
  return {reinterpret_cast<const char*>(digest.data()), digest.size()};
}

AuthShare Public(Word x, size_t rank, const RepShare& key) {
  return {rank == 0 ? x : 0, rank == 2 ? x : 0, key[0] * x, key[1] * x};
}
}  // namespace

std::vector<uint64_t> ArithmeticState::openTruncMask(
    const std::vector<RepShare>& values) {
  const auto context = tag_("trunc-open");
  std::vector<RepShare> low(values.size());
  std::string sent(values.size() * 8, '\0');
  for (size_t i = 0; i < values.size(); ++i) {
    low[i] = {uint64_t(values[i][0]), uint64_t(values[i][1])};
    for (size_t byte = 0; byte < 8; ++byte)
      sent[8 * i + byte] = char(low[i][0] >> (8 * byte));
  }
  CheckCopies(link_, low, context + "/copies");
  // Pi already has components i and i+1; receive the missing i+2 from i-1.
  link_->SendAsync(link_->NextRank(), yacl::ByteContainerView(sent), context);
  const auto received = link_->Recv(link_->PrevRank(), context);
  RequireAll(link_, size_t(received.size()) == sent.size(), context + "/wire");
  std::vector<uint64_t> opened(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t missing = 0;
    for (size_t byte = 0; byte < 8; ++byte)
      missing |= uint64_t(uint8_t(received.data<char>()[8 * i + byte]))
                 << (8 * byte);
    opened[i] = uint64_t(low[i][0]) + uint64_t(low[i][1]) + missing;
    for (size_t byte = 0; byte < 8; ++byte)
      sent[8 * i + byte] = char(opened[i] >> (8 * byte));
  }
  const auto digest = Hash(session_ + "/" + context, sent);
  const auto all = yacl::link::AllGather(link_, yacl::ByteContainerView(digest),
                                         context + "/agree");
  bool ok = true;
  for (const auto& item : all) ok &= std::string_view(item) == digest;
  RequireAll(link_, ok, context + "/agree");
  return opened;
}

NdArrayRef ArithmeticState::truncPrRandomBit(const NdArrayRef& x, size_t bits) {
  validateInternal(x);
  SPU_ENFORCE(bits <= 62, "signed 63-bit truncation requires bits in [0,62]");
  trunc_material_ms_ = 0;
  if (bits == 0 || x.numel() == 0) return x;
  SPU_ENFORCE(
      size_t(x.numel()) <= size_t(std::numeric_limits<int64_t>::max()) / 64,
      "truncation batch is too large");
  const auto material_start = Clock::now();
  // Bound transient memory while preserving one fresh mask per element.
  // Deferred records remain owned by the session until finalization.
  constexpr int64_t kTile = 4096;
  const size_t tiles = (size_t(x.numel()) + kTile - 1) / kTile;
  const size_t available = pending_.max_size() - pending_.size();
  SPU_ENFORCE(
      tiles <= available && size_t(x.numel()) <= (available - tiles) / (3 * 64),
      "truncation authentication queue is too large");
  const size_t records = 3 * 64 * size_t(x.numel()) + tiles;
  pending_.reserve(pending_.size() + records);
  const NdArrayView<AuthShare> input(x);
  NdArrayRef out(x.eltype(), x.shape());
  auto* result = out.data<AuthShare>();
  std::vector<AuthShare> masks(x.numel()), upper(x.numel()), msb(x.numel());
  {
    // Include scratch destruction in the material phase as well as in the
    // primary interval; online timing starts with only the assembled masks.
    TruncationBuffers buffers(x.eltype(), 64 * std::min(kTile, x.numel()));
    for (int64_t offset = 0; offset < x.numel(); offset += kTile) {
      const auto count = std::min(kTile, x.numel() - offset);
      const auto random = randomTruncBits(64 * count, buffers);
      const auto* rb = random.data<AuthShare>();
      // One input combines 64 authenticated bits: use a grain in input units,
      // not the generic 50K-element grain used for cheap scalar operations.
      yacl::parallel_for(0, count, 256, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; ++i) {
          AuthShare mask{}, high{};
          for (size_t j = 0; j < 64; ++j) {
            for (size_t c = 0; c < 4; ++c)
              mask[c] += rb[64 * i + j][c] << j;
          }
          for (size_t j = bits; j < 63; ++j) {
            for (size_t c = 0; c < 4; ++c)
              high[c] += rb[64 * i + j][c] << (j - bits);
          }
          masks[offset + i] = mask;
          upper[offset + i] = high;
          msb[offset + i] = rb[64 * i + 63];
        }
      });
    }
  }
  trunc_material_ms_ =
      std::chrono::duration<double, std::milli>(Clock::now() - material_start)
          .count();
  const auto offset = Public(Word(1) << 62, link_->Rank(), key_);
  std::vector<RepShare> masked(x.numel());
  pforeach(0, x.numel(), [&](int64_t i) {
    masked[i] = {input[i][0] + offset[0] + masks[i][0],
                 input[i][1] + offset[1] + masks[i][1]};
  });
  const auto opened = openTruncMask(masked);
  pforeach(0, x.numel(), [&](int64_t i) {
    const Word cb = opened[i] >> 63;
    const Word q = (opened[i] & ((uint64_t(1) << 63) - 1)) >> bits;
    const Word factor = Word(1) << (63 - bits);
    // Compute constants in the internal ring, including a genuine 128-bit
    // additive inverse for the signed correction; never shift a MAC/share.
    const auto constant =
        Public(q + cb * factor - (Word(1) << (62 - bits)), link_->Rank(), key_);
    for (size_t c = 0; c < 4; ++c)
      result[i][c] =
          constant[c] - upper[i][c] + (Word(1) - 2 * cb) * factor * msb[i][c];
  });
  return out;
}

NdArrayRef TruncPrRandomBit::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         size_t bits, SignType sign) const {
  return ctx->getState<SpdzWiseState>()->truncPrRandomBit(in, bits, sign);
}

void TruncPrRandomBit::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto bits = ctx->getParam<size_t>(1);
  const auto sign = ctx->getParam<SignType>(2);
  ctx->pushOutput(Value(proc(ctx, in.data(), bits, sign), in.dtype()));
}
NdArrayRef TruncASelector::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                                size_t bits, SignType sign) const {
  return TruncPrRandomBit().proc(ctx, in, bits, sign);
}

void TruncASelector::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  const auto bits = ctx->getParam<size_t>(1);
  const auto sign = ctx->getParam<SignType>(2);
  ctx->pushOutput(Value(proc(ctx, in.data(), bits, sign), in.dtype()));
}
}  // namespace spu::mpc::spdzwise::ring

namespace spu::mpc::spdzwise {
NdArrayRef SpdzWiseState::truncPrRandomBit(const NdArrayRef& x, size_t bits,
                                  SignType sign) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing() && ring_, "probabilistic truncation requires RING");
    // All hints use the same signed protocol; no secret sign is computed.
    SPU_ENFORCE(sign == SignType::Unknown || sign == SignType::Positive ||
                    sign == SignType::Negative,
                "invalid truncation sign hint");
    return ring_->truncPrRandomBit(x, bits);
  } catch (...) {
    abort();
    throw;
  }
}
}  // namespace spu::mpc::spdzwise
