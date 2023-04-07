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

#include "libspu/mpc/beaver/arithmetic.h"

#include <functional>
#include <future>

#include "spdlog/spdlog.h"

#include "libspu/core/trace.h"
#include "libspu/mpc/beaver/state.h"
#include "libspu/mpc/beaver/type.h"
#include "libspu/mpc/beaver/value.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::beaver {
namespace {}  // namespace
ArrayRef RandA::proc(KernelEvalContext* ctx, size_t size) {
  SPU_TRACE_MPC_LEAF(ctx, size);

  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();

  ArrayRef out(makeType<AShrTy>(field), size);
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using AShrT = ring2k_t;

    std::vector<AShrT> r0(size);
    std::vector<AShrT> r1(size);
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    auto _out = ArrayView<std::array<AShrT, 2>>(out);
    pforeach(0, size, [&](int64_t idx) {
      // NOTES for ring_rshift to 2 bits.
      //
      // Refer to: New Primitives for Actively-Secure MPC over Rings with
      // Applications to Private Machine Learning
      // - https://eprint.iacr.org/2019/599.pdf
      //
      // It's safer to keep the number within [-2**(k-2), 2**(k-2)) for
      // comparsion operations.
      _out[idx][0] = r0[idx] >> 2;
      _out[idx][1] = r1[idx] >> 2;
    });

    return out;
  });
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using S = std::make_unsigned_t<ring2k_t>;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
    auto _in = ArrayView<std::array<S, 2>>(in);
    auto _out = ArrayView<std::array<S, 2>>(out);

    // neg(x) = not(x) + 1
    // not(x) = neg(x) - 1
    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = -_in[idx][0];
      _out[idx][1] = -_in[idx][1];
      if (rank == 0) {
        _out[idx][1] -= 1;
      } else if (rank == 1) {
        _out[idx][0] -= 1;
      }
    });

    return out;
  });
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using PShrT = ring2k_t;
    using AShrT = ring2k_t;

    ArrayRef out(makeType<Pub2kTy>(field), in.numel());
    auto _in = ArrayView<std::array<AShrT, 2>>(in);
    auto _out = ArrayView<PShrT>(out);

    std::vector<AShrT> x2(in.numel());

    pforeach(0, in.numel(), [&](int64_t idx) {  //
      x2[idx] = _in[idx][1];
    });

    auto x3 = comm->rotate<AShrT>(x2, "a2p");  // comm => 1, k

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    });

    return out;
  });
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();

  // TODO: we should expect Pub2kTy instead of Ring2k
  const auto* in_ty = in.eltype().as<Ring2k>();
  const auto field = in_ty->field();

  auto rank = comm->getRank();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using AShrT = ring2k_t;
    using PShrT = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
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
      r0[idx] = r0[idx] - r1[idx];
    }
    r1 = comm->rotate<AShrT>(r0, "p2b.zero");

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      _out[idx][0] += r0[idx];
      _out[idx][1] += r1[idx];
    }

    return out;
  });
}

ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  auto* comm = ctx->getState<Communicator>();
  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();

  auto rank = comm->getRank();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<U>(rhs);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0];
      _out[idx][1] = _lhs[idx][1];
      if (rank == 0) _out[idx][1] += _rhs[idx];
      if (rank == 1) _out[idx][0] += _rhs[idx];
    });
    return out;
  });
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<AShrTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<std::array<U, 2>>(rhs);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] + _rhs[idx][0];
      _out[idx][1] = _lhs[idx][1] + _rhs[idx][1];
    });
    return out;
  });
}

ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto* lhs_ty = lhs.eltype().as<AShrTy>();
  const auto* rhs_ty = rhs.eltype().as<Pub2kTy>();

  SPU_ENFORCE(lhs_ty->field() == rhs_ty->field());
  const auto field = lhs_ty->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<U>(rhs);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = _lhs[idx][0] * _rhs[idx];
      _out[idx][1] = _lhs[idx][1] * _rhs[idx];
    });
    return out;
  });
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  // auto* beaver_state = ctx->getState<BeaverState>();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    std::vector<U> r0(lhs.numel());
    std::vector<U> r1(lhs.numel());
    prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

    auto _lhs = ArrayView<std::array<U, 2>>(lhs);
    auto _rhs = ArrayView<std::array<U, 2>>(rhs);

    // z1 = (x1 * y1) + (x1 * y2) + (x2 * y1) + (r0 - r1);
    pforeach(0, lhs.numel(), [&](int64_t idx) {
      r0[idx] = (_lhs[idx][0] * _rhs[idx][0]) +  //
                (_lhs[idx][0] * _rhs[idx][1]) +  //
                (_lhs[idx][1] * _rhs[idx][0]) +  //
                (r0[idx] - r1[idx]);
    });

    r1 = comm->rotate<U>(r0, "mulaa");  // comm => 1, k

    ArrayRef out(makeType<AShrTy>(field), lhs.numel());
    auto _out = ArrayView<std::array<U, 2>>(out);
    // auto untrusted = std::vector<beaver::Triple>(lhs.numel());

    pforeach(0, lhs.numel(), [&](int64_t idx) {
      _out[idx][0] = r0[idx];
      _out[idx][1] = r1[idx];
      // untrusted[idx][0] = _lhs[idx];
      // untrusted[idx][1] = _rhs[idx];
      // untrusted[idx][2] = _out[idx];
    });

    return out;
  });
}

ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();

  ArrayRef z(makeType<AShrTy>(field), m * n);

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto z1 = getFirstShare(z);
  auto z2 = getSecondShare(z);

  ring_mmul_(z1, x1, y, m, n, k);
  ring_mmul_(z2, x2, y, m, n, k);

  return z;
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t m, size_t n, size_t k) const {
  SPU_TRACE_MPC_LEAF(ctx, x, y);

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  auto r = std::async([&] {
    auto [r0, r1] = prg_state->genPrssPair(field, m * n);
    return ring_sub(r0, r1);
  });

  auto x1 = getFirstShare(x);
  auto x2 = getSecondShare(x);

  auto y1 = getFirstShare(y);
  auto y2 = getSecondShare(y);

  // z1 := x1*y1 + x1*y2 + x2*y1 + k1
  // z2 := x2*y2 + x2*y3 + x3*y2 + k2
  // z3 := x3*y3 + x3*y1 + x1*y3 + k3
  ArrayRef out(makeType<AShrTy>(field), m * n);
  auto o1 = getFirstShare(out);
  auto o2 = getSecondShare(out);

  auto t2 = std::async(ring_mmul, x2, y1, m, n, k);
  auto t0 = ring_mmul(x1, ring_add(y1, y2), m, n, k);  //
  auto z1 = ring_sum({t0, t2.get(), r.get()});

  auto f = std::async([&] { ring_assign(o1, z1); });
  ring_assign(o2, comm->rotate(z1, kBindName));  // comm => 1, k
  f.get();
  return out;
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto* in_ty = in.eltype().as<AShrTy>();
  const auto field = in_ty->field();
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using U = ring2k_t;

    ArrayRef out(makeType<AShrTy>(field), in.numel());
    auto _in = ArrayView<std::array<U, 2>>(in);
    auto _out = ArrayView<std::array<U, 2>>(out);

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx][0] = _in[idx][0] << bits;
      _out[idx][1] = _in[idx][1] << bits;
    });

    return out;
  });
}

ArrayRef TruncA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                      size_t bits) const {
  SPU_TRACE_MPC_LEAF(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto r_future =
      std::async([&] { return prg_state->genPrssPair(field, in.numel()); });

  // in
  const auto& x1 = getFirstShare(in);
  const auto& x2 = getSecondShare(in);

  const auto kComm = x1.elsize() * x1.numel();

  // we only record the maximum communication, we need to manually add comm
  comm->addCommStatsManually(1, kComm);  // comm => 1, 2

  // ret
  switch (comm->getRank()) {
    case 0: {
      const auto z1 = ring_arshift(x1, bits);
      const auto z2 = comm->recv(1, x1.eltype(), kBindName);
      return makeAShare(z1, z2, field);
    }

    case 1: {
      auto r1 = r_future.get().second;
      const auto z1 = ring_sub(ring_arshift(ring_add(x1, x2), bits), r1);
      comm->sendAsync(0, z1, kBindName);
      return makeAShare(z1, r1, field);
    }

    case 2: {
      const auto z2 = ring_arshift(x2, bits);
      return makeAShare(r_future.get().first, z2, field);
    }

    default:
      SPU_THROW("Party number exceeds 3!");
  }
}

}  // namespace spu::mpc::beaver