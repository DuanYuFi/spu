// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/beaver/conversion.h"

#include <functional>

#include "libspu/core/parallel_utils.h"
#include "libspu/core/platform_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/beaver/state.h"
#include "libspu/mpc/beaver/type.h"
#include "libspu/mpc/beaver/value.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"

namespace spu::mpc::beaver {
// Referrence:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 1 rotate and 1 ppa.
ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct
  //   M = [((x0+x1)^z0, z1) (z1, z2), (z2, (x0+x1)^z0)]
  //   N = [(0, 0), (0, x2), (x2, 0)]
  // Then
  //   Y = PPA(M, N) as the output.
  const PtType out_btype = calcBShareBacktype(SizeOf(field) * 8);
  const auto out_ty = makeType<BShrTy>(out_btype, SizeOf(out_btype) * 8);
  ArrayRef m(out_ty, in.numel());
  ArrayRef n(out_ty, in.numel());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    const auto _in = ArrayView<std::array<ring2k_t, 2>>(in);

    DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
      using BShrT = ScalarT;

      std::vector<BShrT> r0(in.numel());
      std::vector<BShrT> r1(in.numel());
      prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

      pforeach(0, in.numel(), [&](int64_t idx) {
        r0[idx] ^= r1[idx];
        if (comm->getRank() == 0) {
          r0[idx] ^= _in[idx][0] + _in[idx][1];
        }
      });

      r1 = comm->rotate<BShrT>(r0, "a2b");  // comm => 1, k

      auto _m = ArrayView<std::array<BShrT, 2>>(m);
      auto _n = ArrayView<std::array<BShrT, 2>>(n);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _m[idx][0] = r0[idx];
        _m[idx][1] = r1[idx];

        if (comm->getRank() == 0) {
          _n[idx][0] = 0;
          _n[idx][1] = 0;
        } else if (comm->getRank() == 1) {
          _n[idx][0] = 0;
          _n[idx][1] = _in[idx][1];
        } else if (comm->getRank() == 2) {
          _n[idx][0] = _in[idx][0];
          _n[idx][1] = 0;
        }
      });
    });
  });

  return add_bb(ctx->caller(), m, n);  // comm => log(k) + 1, 2k(logk) + k
}

// Referrence:
// 5.3 Share Conversions
// https://eprint.iacr.org/2018/403.pdf
//
// In the semi-honest setting, this can be further optimized by having party 2
// provide (−x2−x3) as private input and compute
//   [x1]B = [x]B + [-x2-x3]B
// using a parallel prefix adder. Regardless, x1 is revealed to parties
// 1,3 and the final sharing is defined as
//   [x]A := (x1, x2, x3)
// Overall, the conversion requires 1 + log k rounds and k + k log k gates.
//
// TODO: convert to single share, will reduce number of rotate.
ArrayRef B2AByPPA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  SPU_ENFORCE(in_nbits <= SizeOf(field) * 8, "invalid nbits={}", in_nbits);
  const auto out_ty = makeType<AShrTy>(field);
  ArrayRef out(out_ty, in.numel());
  if (in_nbits == 0) {
    // special case, it's known to be zero.
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;
      auto _out = ArrayView<std::array<AShrT, 2>>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = 0;
        _out[idx][1] = 0;
      });
    });
    return out;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using BShrT = ScalarT;
    auto _in = ArrayView<std::array<BShrT, 2>>(in);
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;

      // first expand b share to a share length.
      const auto expanded_ty = makeType<BShrTy>(
          calcBShareBacktype(SizeOf(field) * 8), SizeOf(field) * 8);
      ArrayRef x(expanded_ty, in.numel());
      auto _x = ArrayView<std::array<AShrT, 2>>(x);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _x[idx][0] = _in[idx][0];
        _x[idx][1] = _in[idx][1];
      });

      // P1 & P2 local samples ra, note P0's ra is not used.
      std::vector<AShrT> ra0(in.numel());
      std::vector<AShrT> ra1(in.numel());
      std::vector<AShrT> rb0(in.numel());
      std::vector<AShrT> rb1(in.numel());

      prg_state->fillPrssPair(absl::MakeSpan(ra0), absl::MakeSpan(ra1));
      prg_state->fillPrssPair(absl::MakeSpan(rb0), absl::MakeSpan(rb1));

      pforeach(0, in.numel(), [&](int64_t idx) {
        const auto zb = rb0[idx] ^ rb1[idx];
        if (comm->getRank() == 1) {
          rb0[idx] = zb ^ (ra0[idx] + ra1[idx]);
        } else {
          rb0[idx] = zb;
        }
      });
      rb1 = comm->rotate<AShrT>(rb0, "b2a.rand");  // comm => 1, k

      // compute [x+r]B
      ArrayRef r(expanded_ty, in.numel());
      auto _r = ArrayView<std::array<AShrT, 2>>(r);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _r[idx][0] = rb0[idx];
        _r[idx][1] = rb1[idx];
      });

      // comm => log(k) + 1, 2k(logk) + k
      auto x_plus_r = add_bb(ctx->caller(), x, r);
      auto _x_plus_r = ArrayView<std::array<AShrT, 2>>(x_plus_r);

      // reveal
      std::vector<AShrT> x_plus_r_2(in.numel());
      if (comm->getRank() == 0) {
        x_plus_r_2 = comm->recv<AShrT>(2, "reveal.x_plus_r.to.P0");
      } else if (comm->getRank() == 2) {
        std::vector<AShrT> x_plus_r_0(in.numel());
        pforeach(0, in.numel(),
                 [&](int64_t idx) { x_plus_r_0[idx] = _x_plus_r[idx][0]; });
        comm->sendAsync<AShrT>(0, x_plus_r_0, "reveal.x_plus_r.to.P0");
      }

      // P0 hold x+r, P1 & P2 hold -r, reuse ra0 and ra1 as output
      pforeach(0, in.numel(), [&](int64_t idx) {
        if (comm->getRank() == 0) {
          ra0[idx] = _x_plus_r[idx][0] ^ _x_plus_r[idx][1] ^ x_plus_r_2[idx];
        } else {
          ra0[idx] = -ra0[idx];
        }
      });

      ra1 = comm->rotate<AShrT>(ra0, "b2a.rotate");

      auto _out = ArrayView<std::array<AShrT, 2>>(out);
      pforeach(0, in.numel(), [&](int64_t idx) {
        _out[idx][0] = ra0[idx];
        _out[idx][1] = ra1[idx];
      });
    });
  });
  return out;
}

namespace {
// split even and odd bits. e.g.
//   xAyBzCwD -> (xyzw, ABCD)
std::pair<ArrayRef, ArrayRef> bit_split(const ArrayRef& in) {
  constexpr std::array<uint128_t, 6> kSwapMasks = {{
      yacl::MakeUint128(0x2222222222222222, 0x2222222222222222),  // 4bit
      yacl::MakeUint128(0x0C0C0C0C0C0C0C0C, 0x0C0C0C0C0C0C0C0C),  // 8bit
      yacl::MakeUint128(0x00F000F000F000F0, 0x00F000F000F000F0),  // 16bit
      yacl::MakeUint128(0x0000FF000000FF00, 0x0000FF000000FF00),  // 32bit
      yacl::MakeUint128(0x00000000FFFF0000, 0x00000000FFFF0000),  // 64bit
      yacl::MakeUint128(0x0000000000000000, 0xFFFFFFFF00000000),  // 128bit
  }};
  constexpr std::array<uint128_t, 6> kKeepMasks = {{
      yacl::MakeUint128(0x9999999999999999, 0x9999999999999999),  // 4bit
      yacl::MakeUint128(0xC3C3C3C3C3C3C3C3, 0xC3C3C3C3C3C3C3C3),  // 8bit
      yacl::MakeUint128(0xF00FF00FF00FF00F, 0xF00FF00FF00FF00F),  // 16bit
      yacl::MakeUint128(0xFF0000FFFF0000FF, 0xFF0000FFFF0000FF),  // 32bit
      yacl::MakeUint128(0xFFFF00000000FFFF, 0xFFFF00000000FFFF),  // 64bit
      yacl::MakeUint128(0xFFFFFFFF00000000, 0x00000000FFFFFFFF),  // 128bit
  }};

  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();
  SPU_ENFORCE(in_nbits != 0 && in_nbits % 2 == 0, "in_nbits={}", in_nbits);
  const size_t out_nbits = in_nbits / 2;
  const auto out_backtype = calcBShareBacktype(out_nbits);
  const auto out_type = makeType<BShrTy>(out_backtype, out_nbits);

  ArrayRef lo(out_type, in.numel());
  ArrayRef hi(out_type, in.numel());
  using InT = ScalarT;
  auto _in = ArrayView<std::array<InT, 2>>(in);
  DISPATCH_UINT_PT_TYPES(out_backtype, "_", [&]() {
    using OutT = ScalarT;
    auto _lo = ArrayView<std::array<OutT, 2>>(lo);
    auto _hi = ArrayView<std::array<OutT, 2>>(hi);

    if constexpr (sizeof(InT) <= 8) {
      pforeach(0, in.numel(), [&](int64_t idx) {
        constexpr uint64_t S = 0x5555555555555555;  // 01010101
        const InT M = (InT(1) << (in_nbits / 2)) - 1;

        uint64_t r0 = _in[idx][0];
        uint64_t r1 = _in[idx][1];
        _lo[idx][0] = pext_u64(r0, S) & M;
        _hi[idx][0] = pext_u64(r0, ~S) & M;
        _lo[idx][1] = pext_u64(r1, S) & M;
        _hi[idx][1] = pext_u64(r1, ~S) & M;
      });
    } else {
      pforeach(0, in.numel(), [&](int64_t idx) {
        InT r0 = _in[idx][0];
        InT r1 = _in[idx][1];
        // algorithm:
        //      0101010101010101
        // swap  ^^  ^^  ^^  ^^
        //      0011001100110011
        // swap   ^^^^    ^^^^
        //      0000111100001111
        // swap     ^^^^^^^^
        //      0000000011111111
        for (int k = 0; k + 1 < Log2Ceil(in_nbits); k++) {
          InT keep = static_cast<InT>(kKeepMasks[k]);
          InT move = static_cast<InT>(kSwapMasks[k]);
          int shift = 1 << k;

          r0 = (r0 & keep) ^ ((r0 >> shift) & move) ^ ((r0 & move) << shift);
          r1 = (r1 & keep) ^ ((r1 >> shift) & move) ^ ((r1 & move) << shift);
        }
        InT mask = (InT(1) << (in_nbits / 2)) - 1;
        _lo[idx][0] = static_cast<OutT>(r0) & mask;
        _hi[idx][0] = static_cast<OutT>(r0 >> (in_nbits / 2)) & mask;
        _lo[idx][1] = static_cast<OutT>(r1) & mask;
        _hi[idx][1] = static_cast<OutT>(r1 >> (in_nbits / 2)) & mask;
      });
    }
  });

  return std::make_pair(hi, lo);
}

// compute the k'th bit of x + y
ArrayRef carry_out(Object* ctx, const ArrayRef& x, const ArrayRef& y,
                   size_t k) {
  // init P & G
  auto P = xor_bb(ctx, x, y);
  auto G = and_bb(ctx, x, y);

  // Use kogge stone layout.
  while (k > 1) {
    if (k % 2 != 0) {
      k += 1;
      P = lshift_b(ctx, P, 1);
      G = lshift_b(ctx, G, 1);
    }
    auto [P1, P0] = bit_split(P);
    auto [G1, G0] = bit_split(G);

    // Calculate next-level of P, G
    //   P = P1 & P0
    //   G = G1 | (P1 & G0)
    //     = G1 ^ (P1 & G0)
    std::vector<ArrayRef> v = vectorize(
        {P0, G0}, {P1, P1}, [&](const ArrayRef& xx, const ArrayRef& yy) {
          return and_bb(ctx, xx, yy);
        });
    P = std::move(v[0]);
    G = xor_bb(ctx, G1, v[1]);
    k >>= 1;
  }

  return G;
}

}  // namespace

ArrayRef MsbA2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // First construct 2 boolean shares.
  // Let
  //   X = [(x0, x1), (x1, x2), (x2, x0)] as input.
  //   Z = (z0, z1, z2) as boolean zero share.
  //
  // Construct M, N as boolean shares,
  //   M = [((x0+x1)^z0, z1), (z1, z2), (z2, (x0+x1)^z0)]
  //   N = [(0,          0),  (0,  x2), (x2, 0         )]
  //
  // That
  //  M + N = (x0+x1)^z0^z1^z2 + x2
  //        = x0 + x1 + x2 = X
  const Type bshr_type =
      makeType<BShrTy>(GetStorageType(field), SizeOf(field) * 8);
  ArrayRef m(bshr_type, in.numel());
  ArrayRef n(bshr_type, in.numel());
  DISPATCH_ALL_FIELDS(field, "aby3.msb.split", [&]() {
    using U = ring2k_t;

    auto _in = ArrayView<std::array<U, 2>>(in);
    auto _m = ArrayView<std::array<U, 2>>(m);
    auto _n = ArrayView<std::array<U, 2>>(n);

    std::vector<U> r0(numel);
    std::vector<U> r1(numel);
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    pforeach(0, in.numel(), [&](int64_t idx) {
      r0[idx] = r0[idx] ^ r1[idx];
      if (comm->getRank() == 0) {
        r0[idx] ^= (_in[idx][0] + _in[idx][1]);
      }
    });

    r1 = comm->rotate<U>(r0, "m");

    pforeach(0, in.numel(), [&](int64_t idx) {
      _m[idx][0] = r0[idx];
      _m[idx][1] = r1[idx];
      _n[idx][0] = comm->getRank() == 2 ? _in[idx][0] : 0;
      _n[idx][1] = comm->getRank() == 1 ? _in[idx][1] : 0;
    });
  });

  // Compute the k-1'th carry bit.
  size_t nbits = SizeOf(field) * 8 - 1;
  auto carry = carry_out(ctx->caller(), m, n, nbits);

  // Compute the k'th bit.
  //   (m^n)[k] ^ carry
  auto* obj = ctx->caller();
  return xor_bb(obj, rshift_b(obj, xor_bb(obj, m, n), nbits), carry);
}
}  // namespace spu::mpc::beaver