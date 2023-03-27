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

#include "libspu/mpc/beaver/boolean.h"

#include <algorithm>

#include "libspu/core/bit_utils.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/platform_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/beaver/state.h"
#include "libspu/mpc/beaver/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"

namespace spu::mpc::beaver {

ArrayRef B2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const PtType btype = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  using BShrT = ScalarT;
  using PShrT = ring2k_t;

  ArrayRef out(makeType<Pub2kTy>(field), in.numel());

  auto _in = ArrayView<std::array<BShrT, 2>>(in);
  auto _out = ArrayView<PShrT>(out);

  // using spu::mpc::aby3::getShareAs;

  std::vector<BShrT> x2(in.numel());

  pforeach(0, in.numel(), [&](int64_t idx) {  //
    x2[idx] = _in[idx][1];
  });
  auto x3 = comm->rotate<BShrT>(x2, "b2p");  // comm => 1, k

  pforeach(0, in.numel(), [&](int64_t idx) {
    _out[idx] = static_cast<PShrT>(_in[idx][0] ^ _in[idx][1] ^ x3[idx]);
  });

  return out;
}

ArrayRef P2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<Pub2kTy>();
  const auto field = in_ty->field();

  auto _in = ArrayView<ring2k_t>(in);
  const size_t nbits = _in.maxBitWidth();

  const PtType btype = PT_U128;

  using BShrT = ScalarT;
  ArrayRef out(makeType<BShrTy>(btype, nbits), in.numel());
  auto _out = ArrayView<std::array<BShrT, 2>>(out);

  pforeach(0, in.numel(), [&](int64_t idx) {
    if (comm->getRank() == 0) {
      _out[idx][0] = static_cast<BShrT>(_in[idx]);
      _out[idx][1] = 0U;
    } else if (comm->getRank() == 1) {
      _out[idx][0] = 0U;
      _out[idx][1] = 0U;
    } else {
      _out[idx][0] = 0U;
      _out[idx][1] = static_cast<BShrT>(_in[idx]);
    }
  });
  return out;
}

ArrayRef AndBP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  using RhsT = ring2k_t;
  auto _rhs = ArrayView<RhsT>(rhs);
  const size_t rhs_nbits = _rhs.maxBitWidth();
  const size_t out_nbits = std::min(lhs_ty->nbits(), rhs_nbits);
  const PtType out_btype = PT_U128;

  using LhsT = ScalarT;
  auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

  using OutT = ScalarT;

  ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());
  auto _out = ArrayView<std::array<OutT, 2>>(out);
  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][0] = _lhs[idx][0] & _rhs[idx];
    _out[idx][1] = _lhs[idx][1] & _rhs[idx];
  });

  return out;
}

ArrayRef AndBB::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto* prg_state = ctx->getState<PrgState>();
  auto* beaver_state = ctx->getState<BeaverState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* lhs_ty = lhs.eltype().as<BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = PT_U128;
  ArrayRef out(makeType<BShrTy>(out_btype, out_nbits), lhs.numel());

  using RhsT = ScalarT;
  auto _rhs = ArrayView<std::array<RhsT, 2>>(rhs);

  using LhsT = ScalarT;
  auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

  using OutT = ScalarT;

  std::vector<beaver::BinaryTriple> trusted_triples =
      beaver_state->get_bin_triples(ctx->caller(), lhs.numel());

  std::vector<std::array<LhsT, 2>> to_be_open(lhs.numel() * 2);
  pforeach(0, lhs.numel(), [&](uint64_t idx) {
    to_be_open[idx * 2][0] = _lhs[idx][0] ^ trusted_triples[idx][0][0];
    to_be_open[idx * 2][1] = _lhs[idx][1] ^ trusted_triples[idx][0][1];

    to_be_open[idx * 2 + 1][0] = _rhs[idx][0] ^ trusted_triples[idx][1][0];
    to_be_open[idx * 2 + 1][1] = _rhs[idx][1] ^ trusted_triples[idx][1][1];
  });

  auto opened = ctx->caller()->call("B2P", to_be_open);
  auto _opened = ArrayView<OutT>(opened);

  auto _out = ArrayView<std::array<OutT, 2>>(out);
  auto rank = comm->getRank();

  pforeach(0, lhs.numel(), [&](uint64_t idx) {
    auto d = _opened[idx * 2];
    auto e = _opened[idx * 2 + 1];
    _out[idx][0] =
        d & _rhs[idx][0] ^ e & _lhs[idx][0] ^ trusted_triples[idx][2][0];
    _out[idx][1] =
        d & _rhs[idx][1] ^ e & _lhs[idx][1] ^ trusted_triples[idx][2][1];

    if (rank == 0) {
      _out[idx][0] ^= (d & e);
    }
    if (rank == 2) {
      _out[idx][1] ^= (d & e);
    }
  });
  return out;
}

}  // namespace spu::mpc::beaver