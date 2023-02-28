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

#include "libspu/mpc/spdzwisefield/arithmetic.h"

#include <functional>
#include <future>

#include "spdlog/spdlog.h"

#include "libspu/core/array_ref.h"
#include "libspu/core/trace.h"
// #include "libspu/mpc/aby3/ot.h"
#include "libspu/mpc/spdzwisefield/type.h"
// #include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdzwisefield {

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  auto* honest_state = ctx->getState<SpdzWiseFieldState>();
  auto* prg_state = ctx->getState<PrgState>();
  const auto key = honest_state->key();
  const auto field = in.eltype().as<Ring2k>()->field();

  auto rank = comm->getRank();

  using U = uint128_t;
  using Field = MersennePrimeField;

  // get the input share from semi-honest protocol
  ArrayRef honest_out = honest_state->call("p2a", in);

  ArrayRef out(makeType<AShrTy>(field), in.numel());
  ArrayView _in = ArrayView<U>(in);
  ArrayView _out = ArrayView<std::array<U, 4>>(out);
  ArrayView _honest_out = ArrayView<std::array<U, 2>>(honest_out);

  // get the share of mac. [[mac]] = [[x]] * [[key]]
  std::vector<U> r0(in.numel());
  std::vector<U> r1(in.numel());
  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  pforeach(0, in.numel(), [&](int64_t idx) {
    // r0[idx] = (_in[idx][0] * key[0]) +  //
    //           (_in[idx][0] * key[1]) +  //
    //           (_in[idx][1] * key[0]) +  //
    //           (r0[idx] - r1[idx]);
    r0[idx] = Field::add(Field::mul(_in[idx][0], key[0]),  //
                         Field::mul(_in[idx][0], key[1]),  //
                         Field::mul(_in[idx][1], key[0]),  //
                         Field::sub(r0[idx], r1[idx]));    //
  });

  r1 = comm->rotate<U>(r0, "p2a");  // comm => 1, k

  pforeach(0, in.numel(), [&](int64_t idx) {  //
    _out[idx][0] = _honest_out[idx][0];       //
    _out[idx][1] = _honest_out[idx][1];       //
    _out[idx][2] = r0[idx];                   //
    _out[idx][3] = r1[idx];
  });

  return out;
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  using ring2k_t = uint128_t;
  using Field = MersennePrimeField;

  ArrayRef out(makeType<Pub2kTy>(field), in.numel());
  auto _in = ArrayView<std::array<ring2k_t, 4>>(in);
  auto _out = ArrayView<ring2k_t>(out);

  std::vector<AShrT> x2(in.numel());

  pforeach(0, in.numel(), [&](int64_t idx) {  //
    x2[idx] = _in[idx][1];
  });

  auto x3 = comm->rotate<AShrT>(x2, "a2p");

  pforeach(0, in.numel(), [&](int64_t idx) {
    // _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    _out[idx] = Field::add(_in[idx][0], _in[idx][1], x3[idx]);
  });

  return out;
}

}  // namespace spu::mpc::spdzwisefield