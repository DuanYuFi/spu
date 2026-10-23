// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

// Local split-3 follows MP-SPDZ SpdzWiseRingShare::split and
// Rep3Share2::split(case 3); see MP-SPDZ-LICENSE.txt. Only value limbs enter
// the circuit. The application ring is Z/2^64, irrespective of MAC extension.
#include "libspu/mpc/spdzwise/ring_conversion.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/state.h"

namespace spu::mpc::spdzwise {
namespace {
using Planes = std::vector<NdArrayRef>;
NdArrayRef Zero(const Type& type, int64_t n) {
  NdArrayRef a(type, {n});
  std::fill_n(a.data<RepShare>(), n, RepShare{});
  return a;
}
NdArrayRef Xor(const NdArrayRef& a, const NdArrayRef& b) {
  NdArrayRef r(a.eltype(), a.shape());
  const auto* x = a.data<RepShare>();
  const auto* y = b.data<RepShare>();
  auto* z = r.data<RepShare>();
  for (int64_t i = 0; i < a.numel(); ++i)
    z[i] = {x[i][0] ^ y[i][0], x[i][1] ^ y[i][1]};
  return r;
}
// Batch independent gates into one ABFL call/round. All lanes, including the
// last input block's padding, stay in the verifier's immutable snapshots.
Planes AndLayer(SpdzWiseState* state, const Planes& a, const Planes& b) {
  SPU_ENFORCE(!a.empty() && a.size() == b.size());
  const int64_t n = a[0].numel();
  NdArrayRef x(a[0].eltype(), {int64_t(a.size()) * n});
  NdArrayRef y(x.eltype(), x.shape());
  for (size_t j = 0; j < a.size(); ++j) {
    std::copy_n(a[j].data<RepShare>(), n, x.data<RepShare>() + j * n);
    std::copy_n(b[j].data<RepShare>(), n, y.data<RepShare>() + j * n);
  }
  auto z = state->booleanBinary(x, y, true, false);
  Planes out;
  for (size_t j = 0; j < a.size(); ++j) {
    NdArrayRef v(z.eltype(), {n});
    std::copy_n(z.data<RepShare>() + j * n, n, v.data<RepShare>());
    out.push_back(std::move(v));
  }
  return out;
}
// In-place 64x64 transpose: row i, bit j maps to row j, bit i.
void Transpose(std::array<uint64_t, 64>& a) {
  uint64_t mask = 0x00000000ffffffffULL;
  for (unsigned d = 32; d; d >>= 1, mask ^= mask << d) {
    for (unsigned i = 0; i < 64; i = (i + d + 1) & ~d) {
      const auto t = ((a[i] >> d) ^ a[i + d]) & mask;
      a[i] ^= t << d;
      a[i + d] ^= t;
    }
  }
}
// Ripple carry: one AND per retained carry. A2B skips bit zero because
// its second CSA output has a public zero there; general B2A addition does not.
Planes Add(SpdzWiseState* state, const Planes& a, const Planes& b,
           bool zero_lsb = false) {
  Planes result;
  if (zero_lsb) result.push_back(a[0]);
  auto carry = Zero(a[0].eltype(), a[0].numel());
  for (size_t bit = zero_lsb ? 1 : 0; bit < 64; ++bit) {
    result.push_back(Xor(Xor(a[bit], b[bit]), carry));
    if (bit < 63) {
      auto next = state->booleanBinary(Xor(a[bit], carry), Xor(b[bit], carry),
                                       true, false);
      carry = Xor(next, carry);
    }
  }
  return result;
}
Planes SliceBoolean(const NdArrayRef& in, const Type& type) {
  const auto blocks = in.numel() / 64 + (in.numel() % 64 != 0);
  Planes out;
  for (size_t bit = 0; bit < 64; ++bit) out.push_back(Zero(type, blocks));
  const NdArrayView<RepShare> input(in);
  yacl::parallel_for(0, blocks, 64, [&](int64_t begin, int64_t end) {
    for (auto block = begin; block < end; ++block) {
      for (size_t c = 0; c < 2; ++c) {
        std::array<uint64_t, 64> rows{};
        const auto lanes = std::min(int64_t(64), in.numel() - 64 * block);
        for (int64_t lane = 0; lane < lanes; ++lane)
          rows[lane] = input[64 * block + lane][c];
        Transpose(rows);
        for (size_t bit = 0; bit < 64; ++bit)
          out[bit].data<RepShare>()[block][c] = rows[bit];
      }
    }
  });
  return out;
}
NdArrayRef Unslice(const Planes& planes, const Shape& shape) {
  NdArrayRef out(planes[0].eltype(), shape);
  const auto blocks = planes[0].numel();
  yacl::parallel_for(0, blocks, 64, [&](int64_t begin, int64_t end) {
    for (auto block = begin; block < end; ++block)
      for (size_t c = 0; c < 2; ++c) {
        std::array<uint64_t, 64> rows;
        for (size_t bit = 0; bit < 64; ++bit)
          rows[bit] = planes[bit].data<RepShare>()[block][c];
        Transpose(rows);
        const auto lanes = std::min(int64_t(64), out.numel() - 64 * block);
        for (int64_t lane = 0; lane < lanes; ++lane)
          out.data<RepShare>()[64 * block + lane][c] = rows[lane];
      }
  });
  return out;
}
// 63 CSA ANDs per input. The carry above bit 63 is discarded modulo 2^64.
std::array<Planes, 2> SplitSummands(SpdzWiseState* state, const NdArrayRef& in) {
  const auto type = makeType<BShrTy>(state->session(), 64);
  const auto blocks = in.numel() / 64 + (in.numel() % 64 != 0);
  std::array<Planes, 3> x;
  for (auto& summand : x)
    for (size_t bit = 0; bit < 64; ++bit)
      summand.push_back(Zero(type, blocks));
  const NdArrayView<ring::AuthShare> input(in);
  yacl::parallel_for(0, blocks, 64, [&](int64_t begin, int64_t end) {
    for (auto block = begin; block < end; ++block) {
      for (size_t c = 0; c < 2; ++c) {
        std::array<uint64_t, 64> rows{};
        const auto lanes = std::min(int64_t(64), in.numel() - 64 * block);
        for (int64_t lane = 0; lane < lanes; ++lane) {
          rows[lane] = uint64_t(input[64 * block + lane][c]);
        }
        Transpose(rows);
        for (size_t bit = 0; bit < 64; ++bit)
          x[(state->rank() + c) % 3][bit].data<RepShare>()[block][c] = rows[bit];
      }
    }
  });
  Planes s, lhs, rhs;
  for (size_t bit = 0; bit < 64; ++bit) {
    auto a = Xor(x[0][bit], x[2][bit]);
    s.push_back(Xor(a, x[1][bit]));
    if (bit < 63) {
      lhs.push_back(std::move(a));
      rhs.push_back(Xor(x[1][bit], x[2][bit]));
    }
  }
  const auto products = AndLayer(state, lhs, rhs);
  Planes b{Zero(type, blocks)};
  for (size_t bit = 0; bit < 63; ++bit)
    b.push_back(Xor(products[bit], x[2][bit]));
  return {std::move(s), std::move(b)};
}
Planes Sliced(SpdzWiseState* state, const NdArrayRef& in) {
  const auto summands = SplitSummands(state, in);
  return Add(state, summands[0], summands[1], true);
}
// MP-SPDZ CarryOutRawLE/CarryOutAux: adjacent little-endian intervals combine
// as G = G_hi XOR (P_hi AND G_lo), P = P_hi AND P_lo. The XOR equals OR
// because generate and propagate are disjoint. The lowest interval never
// needs P (carry-in is zero). Split-3 also makes b[0] public zero, so bit zero
// can be omitted altogether. Keep only the carry into bit 63, not all sums.
NdArrayRef MsbCarry(SpdzWiseState* state, const Planes& a, const Planes& b) {
  Planes lhs(a.begin() + 1, a.begin() + 63);
  Planes rhs(b.begin() + 1, b.begin() + 63);
  auto g = AndLayer(state, lhs, rhs);
  Planes p(g.size());
  for (size_t i = 1; i < p.size(); ++i) p[i] = Xor(lhs[i], rhs[i]);
  while (g.size() > 1) {
    lhs.clear();
    rhs.clear();
    for (size_t i = 0; i + 1 < g.size(); i += 2) {
      lhs.push_back(p[i + 1]);
      rhs.push_back(g[i]);
      if (i != 0) {
        lhs.push_back(p[i + 1]);
        rhs.push_back(p[i]);
      }
    }
    const auto products = AndLayer(state, lhs, rhs);
    Planes next_g, next_p;
    size_t gate = 0;
    for (size_t i = 0; i + 1 < g.size(); i += 2) {
      next_g.push_back(Xor(g[i + 1], products[gate++]));
      next_p.push_back(i ? products[gate++] : NdArrayRef{});
    }
    if (g.size() % 2) {
      next_g.push_back(g.back());
      next_p.push_back(p.back());
    }
    g = std::move(next_g);
    p = std::move(next_p);
  }
  return g[0];
}
NdArrayRef UnsliceBit(const NdArrayRef& plane, const Type& type,
                     const Shape& shape) {
  NdArrayRef out(type, shape);
  yacl::parallel_for(0, plane.numel(), 64, [&](int64_t begin, int64_t end) {
    for (auto block = begin; block < end; ++block) {
      const auto& packed = plane.data<RepShare>()[block];
      const auto lanes = std::min(int64_t(64), out.numel() - 64 * block);
      for (int64_t lane = 0; lane < lanes; ++lane)
        out.data<RepShare>()[64 * block + lane] = {
            (packed[0] >> lane) & 1, (packed[1] >> lane) & 1};
    }
  });
  return out;
}
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

}  // namespace

NdArrayRef SpdzWiseState::a2b(const NdArrayRef& in) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing(), "split-3 A2B requires RING");
    validateInternal(in);
    if (in.numel() == 0)
      return NdArrayRef(makeType<BShrTy>(session_, 64), in.shape());
    return Unslice(Sliced(this, in), in.shape());
  } catch (...) {
    abort();
    throw;
  }
}
NdArrayRef SpdzWiseState::msbA2B(const NdArrayRef& in) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing(), "split-3 MSB A2B requires RING");
    validateInternal(in);
    const auto type = makeType<BShrTy>(session_, 1);
    if (in.numel() == 0) return NdArrayRef(type, in.shape());
    const auto summands = SplitSummands(this, in);
    const auto carry = MsbCarry(this, summands[0], summands[1]);
    const auto msb = Xor(Xor(summands[0][63], summands[1][63]), carry);
    return UnsliceBit(msb, type, in.shape());
  } catch (...) {
    abort();
    throw;
  }
}
// This checkpoint is shared by the real B2A path and fault-injection tests.
// It is private: callers cannot open arbitrary unchecked Boolean values.
NdArrayRef SpdzWiseState::finishB2A(const NdArrayRef& mask,
                                    const NdArrayRef& masked) {
  auto start = Clock::now();
  auto bytes = link_->GetStats()->sent_bytes.load();
  verify();
  b2a_ms_[2] = Ms(start);
  b2a_bytes_[2] = link_->GetStats()->sent_bytes.load() - bytes;
  start = Clock::now();
  bytes = link_->GetStats()->sent_bytes.load();
  const auto opened = boolean_->open(
      {masked.data<RepShare>(), size_t(masked.numel())}, "b2a-open");
  NdArrayRef c(makeType<Pub2kTy>(FM64), mask.shape());
  std::copy(opened.begin(), opened.end(), c.data<uint64_t>());
  // Keep c-r in the extended ring. Its low 64 bits equal x, and the MAC
  // authenticates this exact representative, including any high limbs.
  auto out = add(publicValue(c), negate(mask));
  b2a_ms_[3] = Ms(start);
  b2a_bytes_[3] = link_->GetStats()->sent_bytes.load() - bytes;
  return out;
}
NdArrayRef SpdzWiseState::b2a(const NdArrayRef& in) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing(), "split-3 B2A requires RING");
    validateBoolean(in);
    const auto width = in.eltype().as<BShrTy>()->nbits();
    SPU_ENFORCE(width >= 1 && width <= 64, "B2A requires 1..64 bits");
    b2a_ms_ = {};
    b2a_bytes_ = {};
    b2a_and_bits_ = 0;
    if (!in.numel())
      return NdArrayRef(makeType<ring::AShrTy>(session_), in.shape());
    maybeCheck();
    // Reuse the checked daBit conversion for a single Boolean bit. Wider
    // words, including signed exact-truncation results, retain split-3 B2A.
    if (width == 1) return bit2a(in);
    auto start = Clock::now();
    auto bytes = link_->GetStats()->sent_bytes.load();
    const auto mask = ring_->randomAuthenticated(in.shape());
    b2a_ms_[0] = Ms(start);
    b2a_bytes_[0] = link_->GetStats()->sent_bytes.load() - bytes;
    start = Clock::now();
    bytes = link_->GetStats()->sent_bytes.load();
    const auto before = boolean_->totalBits();
    const auto r = Sliced(this, mask);
    const auto x = SliceBoolean(in, makeType<BShrTy>(session_, 64));
    const auto masked = Unslice(Add(this, x, r), in.shape());
    b2a_and_bits_ = boolean_->totalBits() - before;
    b2a_ms_[1] = Ms(start);
    b2a_bytes_[1] = link_->GetStats()->sent_bytes.load() - bytes;
    return finishB2A(mask, masked);
  } catch (...) {
    abort();
    throw;
  }
}
void ring::B2A::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  ctx->pushOutput(Value(proc(ctx, in.data()), in.dtype()));
}
NdArrayRef ring::B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->b2a(in);
}
void ring::A2B::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  ctx->pushOutput(Value(proc(ctx, in.data()), in.dtype()));
}
NdArrayRef ring::A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->a2b(in);
}
void ring::MsbA2B::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  ctx->pushOutput(Value(proc(ctx, in.data()), in.dtype()));
}
NdArrayRef ring::MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->msbA2B(in);
}
}  // namespace spu::mpc::spdzwise
