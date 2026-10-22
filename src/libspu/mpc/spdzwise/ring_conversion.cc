// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

// Local split-3 follows MP-SPDZ SpdzWiseRingShare::split and
// Rep3Share2::split(case 3); see MP-SPDZ-LICENSE.txt. Only value limbs enter
// the circuit. The application ring is Z/2^64, irrespective of MAC extension.
#include "libspu/mpc/spdzwise/ring_conversion.h"

#include <algorithm>
#include <array>
#include <vector>

#include "libspu/core/parallel_utils.h"
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
// 63 CSA ANDs plus 62 ripple-carry ANDs per input; neither the CSA
// carry above bit 63 nor the final addition carry is needed modulo 2^64.
Planes Sliced(SpdzWiseState* state, const NdArrayRef& in) {
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
  Planes result{s[0]};
  auto carry = Zero(type, blocks);
  for (size_t bit = 1; bit < 64; ++bit) {
    result.push_back(Xor(Xor(s[bit], b[bit]), carry));
    if (bit < 63) {
      auto next = state->booleanBinary(Xor(s[bit], carry),
                                       Xor(b[bit], carry), true, false);
      carry = Xor(next, carry);
    }
  }
  return result;
}
}  // namespace

NdArrayRef SpdzWiseState::a2b(const NdArrayRef& in) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing(), "split-3 A2B requires RING");
    validateInternal(in);
    NdArrayRef out(makeType<BShrTy>(session_, 64), in.shape());
    if (in.numel() == 0) return out;
    const auto planes = Sliced(this, in);
    const auto blocks = planes[0].numel();
    yacl::parallel_for(0, blocks, 64, [&](int64_t begin, int64_t end) {
      for (auto block = begin; block < end; ++block)
        for (size_t c = 0; c < 2; ++c) {
          std::array<uint64_t, 64> rows;
          for (size_t bit = 0; bit < 64; ++bit)
            rows[bit] = planes[bit].data<RepShare>()[block][c];
          Transpose(rows);
          const auto lanes = std::min(int64_t(64), in.numel() - 64 * block);
          for (int64_t lane = 0; lane < lanes; ++lane) {
            out.data<RepShare>()[64 * block + lane][c] = rows[lane];
          }
        }
    });
    return out;
  } catch (...) {
    abort();
    throw;
  }
}
void ring::A2B::evaluate(KernelEvalContext* ctx) const {
  const auto& in = ctx->getParam<Value>(0);
  ctx->pushOutput(Value(proc(ctx, in.data()), in.dtype()));
}
NdArrayRef ring::A2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->a2b(in);
}
}  // namespace spu::mpc::spdzwise
