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
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/spdzwisefield/state.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/spdzwisefield/utils.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdzwisefield {

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  auto* sy_ring_state = ctx->getState<SpdzWiseFieldState>();
  auto* prg_state = ctx->getState<PrgState>();
  const auto key = sy_ring_state->key();
  const auto field = in.eltype().as<Ring2k>()->field();

  auto rank = comm->getRank();

  using U = uint64_t;
  using Field = SpdzWiseFieldState::Field;
  using HAShrTy = spu::mpc::aby3::AShrTy;

  // get the input share from semi-honest protocol
  ArrayRef honest_out(makeType<HAShrTy>(field), in.numel());
  auto _honest_out = ArrayView<std::array<U, 2>>(honest_out);
  auto _in = ArrayView<U>(in);

  pforeach(0, in.numel(), [&](int64_t idx) {
    _honest_out[idx][0] = rank == 0 ? _in[idx] : 0;
    _honest_out[idx][1] = rank == 2 ? _in[idx] : 0;
  });

  // for debug purpose, randomize the inputs to avoid corner cases.

  std::vector<U> r0(in.numel());
  std::vector<U> r1(in.numel());

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  pforeach(0, in.numel(), [&](uint64_t idx) {
    r0[idx] = Field::modp(r0[idx]);
    r1[idx] = Field::modp(r1[idx]);
    r0[idx] = Field::sub(r0[idx], r1[idx]);
  });

  r1 = comm->rotate<U>(r0, "p2a");

  pforeach(0, in.numel(), [&](uint64_t idx) {
    _honest_out[idx][0] = Field::add(_honest_out[idx][0], r0[idx]);
    _honest_out[idx][1] = Field::add(_honest_out[idx][1], r1[idx]);
  });

  ArrayRef out(makeType<AShrTy>(field), in.numel());
  auto _out = ArrayView<std::array<U, 4>>(out);

  // get the share of mac. [[mac]] = [[x]] * [[key]]
  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  pforeach(0, in.numel(), [&](int64_t idx) {
    r0[idx] = Field::add(Field::mul(_honest_out[idx][0], key[0]),  //
                         Field::mul(_honest_out[idx][0], key[1]),  //
                         Field::mul(_honest_out[idx][1], key[0]),  //
                         Field::sub(r0[idx], r1[idx]));            //
  });

  r1 = comm->rotate<U>(r0, "p2a");  // comm => 1, k

  pforeach(0, in.numel(), [&](int64_t idx) {  //
    _out[idx][0] = _honest_out[idx][0];
    _out[idx][1] = _honest_out[idx][1];
    _out[idx][2] = r0[idx];
    _out[idx][3] = r1[idx];
  });

  return out;
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  using ring2k_t = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  ArrayRef out(makeType<Pub2kTy>(field), in.numel());
  auto _in = ArrayView<std::array<ring2k_t, 4>>(in);
  auto _out = ArrayView<ring2k_t>(out);

  std::vector<ring2k_t> x2(in.numel());

  pforeach(0, in.numel(), [&](int64_t idx) {  //
    x2[idx] = _in[idx][1];
  });

  auto x3 = comm->rotate<ring2k_t>(x2, "a2p");

  pforeach(0, in.numel(), [&](int64_t idx) {
    // _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    _out[idx] = Field::add(_in[idx][0], _in[idx][1], x3[idx]);
  });

  return out;
}

ArrayRef A2PSH::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  using PShrT = uint64_t;
  using AShrT = uint64_t;

  using Field = SpdzWiseFieldState::Field;

  ArrayRef out(makeType<Pub2kTy>(field), in.numel());
  auto _in = ArrayView<std::array<AShrT, 2>>(in);
  auto _out = ArrayView<PShrT>(out);

  std::vector<AShrT> x2(in.numel());

  pforeach(0, in.numel(), [&](int64_t idx) {  //
    x2[idx] = _in[idx][1];
  });

  auto x3 = comm->rotate<AShrT>(x2, "a2p");  // comm => 1, k

  pforeach(0, in.numel(), [&](int64_t idx) {
    _out[idx] = Field::add(_in[idx][0], _in[idx][1], x3[idx]);
  });

  return out;
}

ArrayRef P2ASH::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();

  // TODO: we should expect Pub2kTy instead of Ring2k
  const auto* in_ty = in.eltype().as<Ring2k>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();

  using AShrT = uint64_t;
  using PShrT = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  ArrayRef out(makeType<aby3::AShrTy>(field), in.numel());
  auto _in = ArrayView<PShrT>(in);
  auto _out = ArrayView<std::array<AShrT, 2>>(out);

  pforeach(0, in.numel(), [&](int64_t idx) {
    _out[idx][0] = rank == 0 ? _in[idx] : 0;
    _out[idx][1] = rank == 2 ? _in[idx] : 0;
  });

  std::vector<AShrT> r0(in.numel());
  std::vector<AShrT> r1(in.numel());
  auto* prg_state = ctx->getState<PrgState>();
  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  for (int64_t idx = 0; idx < in.numel(); idx++) {
    r1[idx] = Field::modp(r1[idx]);
    r0[idx] = Field::sub(Field::modp(r0[idx]), r1[idx]);
  }
  r1 = comm->rotate<AShrT>(r0, "p2b.zero");

  for (int64_t idx = 0; idx < in.numel(); idx++) {
    _out[idx][0] = Field::add(_out[idx][0], r0[idx]);
    _out[idx][1] = Field::add(_out[idx][1], r1[idx]);
  }

  return out;
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto* comm = ctx->getState<Communicator>();
  const auto key = ctx->getState<SpdzWiseFieldState>()->key();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  auto rank = comm->getRank();

  using U = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  ArrayRef out(makeType<AShrTy>(field), lhs.numel());

  auto _lhs = ArrayView<std::array<U, 4>>(lhs);
  auto _rhs = ArrayView<U>(rhs);
  auto _out = ArrayView<std::array<U, 4>>(out);

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][0] = _lhs[idx][0];
    _out[idx][1] = _lhs[idx][1];
    _out[idx][2] = Field::add(_lhs[idx][2], Field::mul(key[0], _rhs[idx]));
    _out[idx][3] = Field::add(_lhs[idx][3], Field::mul(key[1], _rhs[idx]));
    if (rank == 0) _out[idx][1] += _rhs[idx];
    if (rank == 1) _out[idx][0] += _rhs[idx];
  });
  return out;
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  using U = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  ArrayRef out(makeType<AShrTy>(field), lhs.numel());

  auto _lhs = ArrayView<std::array<U, 4>>(lhs);
  auto _rhs = ArrayView<std::array<U, 4>>(rhs);
  auto _out = ArrayView<std::array<U, 4>>(out);

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][0] = Field::add(_lhs[idx][0], _rhs[idx][0]);
    _out[idx][1] = Field::add(_lhs[idx][1], _rhs[idx][1]);
    _out[idx][2] = Field::add(_lhs[idx][2], _rhs[idx][2]);
    _out[idx][3] = Field::add(_lhs[idx][3], _rhs[idx][3]);
  });
  return out;
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  using U = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  ArrayRef out(makeType<AShrTy>(field), lhs.numel());

  auto _lhs = ArrayView<std::array<U, 4>>(lhs);
  auto _rhs = ArrayView<U>(rhs);
  auto _out = ArrayView<std::array<U, 4>>(out);

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][0] = Field::mul(_lhs[idx][0], _rhs[idx]);
    _out[idx][1] = Field::mul(_lhs[idx][1], _rhs[idx]);
    _out[idx][2] = Field::mul(_lhs[idx][2], _rhs[idx]);
    _out[idx][3] = Field::mul(_lhs[idx][3], _rhs[idx]);
  });
  return out;
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  using U = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  auto _lhs = ArrayView<std::array<U, 4>>(lhs);
  auto _rhs = ArrayView<std::array<U, 4>>(rhs);

  std::vector<U> r0(lhs.numel() * 2);
  std::vector<U> r1(lhs.numel() * 2);
  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    r0[idx] = Field::add(Field::mul(_lhs[idx][0], _rhs[idx][0]),  //
                         Field::mul(_lhs[idx][0], _rhs[idx][1]),  //
                         Field::mul(_lhs[idx][1], _rhs[idx][0]),  //
                         Field::sub(r0[idx], r1[idx]));
  });

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    r0[idx + lhs.numel()] =
        Field::add(Field::mul(_lhs[idx][2], _rhs[idx][2]),  //
                   Field::mul(_lhs[idx][2], _rhs[idx][3]),  //
                   Field::mul(_lhs[idx][3], _rhs[idx][2]),  //
                   Field::sub(r0[idx + lhs.numel()], r1[idx + lhs.numel()]));
  });

  r1 = comm->rotate<U>(r0, "mulaa");  // comm => 1, 2 * k

  ArrayRef out(makeType<AShrTy>(field), lhs.numel());
  auto _out = ArrayView<std::array<U, 2>>(out);
  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][0] = r0[idx];
    _out[idx][1] = r1[idx];
  });

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][2] = r0[lhs.numel() + idx];
    _out[idx][3] = r1[lhs.numel() + idx];
  });

  return out;
}

ArrayRef MulAASemiHonest::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                               const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  using U = uint64_t;
  using Field = SpdzWiseFieldState::Field;

  std::vector<U> r0(lhs.numel());
  std::vector<U> r1(lhs.numel());
  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  auto _lhs = ArrayView<std::array<U, 2>>(lhs);
  auto _rhs = ArrayView<std::array<U, 2>>(rhs);

  // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
  pforeach(0, lhs.numel(), [&](int64_t idx) {
    r0[idx] = Field::modp(r0[idx]);
    r1[idx] = Field::modp(r1[idx]);

    r0[idx] = Field::add(Field::mul(_lhs[idx][0], _rhs[idx][0]),
                         Field::mul(_lhs[idx][0], _rhs[idx][1]),
                         Field::mul(_lhs[idx][1], _rhs[idx][0]),
                         Field::sub(r0[idx], r1[idx]));
  });

  r1 = comm->rotate<U>(r0, "mulaa.sh");  // comm => 1, k

  ArrayRef out(makeType<aby3::AShrTy>(FM64), lhs.numel());
  auto _out = ArrayView<std::array<U, 2>>(out);

  pforeach(0, lhs.numel(), [&](int64_t idx) {
    _out[idx][0] = r0[idx];
    _out[idx][1] = r1[idx];
  });

  return out;
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  ArrayRef mul_p(makeType<Pub2kTy>(FM64), in.numel());
  auto _mul_p = ArrayView<uint64_t>(mul_p);

  using Field = SpdzWiseFieldState::Field;

  pforeach(0, in.numel(),
           [&](uint64_t idx) { _mul_p[idx] = Field::modp(1 << bits); });

  return ctx->caller()->call("mulap", in, mul_p);
}

ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                      size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  return ArrayRef();
}

}  // namespace spu::mpc::spdzwisefield