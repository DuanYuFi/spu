// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/spdzwise/arithmetic.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/replicated.h"
#include "libspu/mpc/spdzwise/state.h"

namespace spu::mpc::spdzwise {
namespace {

RepShare Values(const AuthShare& x) { return {x[0], x[1]}; }
RepShare Macs(const AuthShare& x) { return {x[2], x[3]}; }

void RequirePublic(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype() == makeType<Pub2kTy>(FM64),
              "expected an FM64 public value");
}

}  // namespace

// Dispatch RING operations to its independent arithmetic state.
NdArrayRef SpdzWiseState::publicValue(const NdArrayRef& value) const {
  requireRunning();
  if (isRing()) return ring_->publicValue(value);
  RequirePublic(value);
  NdArrayRef result(makeType<AShrTy>(session_), value.shape());
  const NdArrayView<uint64_t> in(value);
  NdArrayView<AuthShare> out(result);
  for (int64_t i = 0; i < value.numel(); ++i) {
    const auto x = FromRing(in[i]);
    out[i] = {rank() == 0 ? x : 0, rank() == 2 ? x : 0, Mul(key_[0], x),
              Mul(key_[1], x)};
  }
  return result;
}

NdArrayRef SpdzWiseState::add(const NdArrayRef& x, const NdArrayRef& y) const {
  requireRunning();
  if (isRing()) return ring_->add(x, y);
  validateInternal(x);
  validateInternal(y);
  SPU_ENFORCE(x.shape() == y.shape(), "add shape mismatch");
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> a(x), b(y);
  NdArrayView<AuthShare> out(result);
  for (int64_t i = 0; i < x.numel(); ++i) {
    for (size_t c = 0; c < 4; ++c) {
      out[i][c] = Add(a[i][c], b[i][c]);
    }
  }
  return result;
}

NdArrayRef SpdzWiseState::negate(const NdArrayRef& x) const {
  requireRunning();
  if (isRing()) return ring_->negate(x);
  validateInternal(x);
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> in(x);
  NdArrayView<AuthShare> out(result);
  for (int64_t i = 0; i < x.numel(); ++i) {
    for (size_t c = 0; c < 4; ++c) {
      out[i][c] = Neg(in[i][c]);
    }
  }
  return result;
}

NdArrayRef SpdzWiseState::leftShift(const NdArrayRef& x,
                                    const Sizes& bits) const {
  requireRunning();
  SPU_ENFORCE(isRing(), "lshift_a requires RING");
  return ring_->leftShift(x, bits);
}

NdArrayRef SpdzWiseState::randomArithmetic(const Shape& shape) {
  try {
    requireRunning();
    SPU_ENFORCE(isRing(), "rand_a requires RING");
    // Full-ring random values: shifting each PRSS share does not bound the
    // reconstructed secret. MSB(x-y) comparison still requires the caller
    // to ensure that the signed difference does not overflow.
    return ring_->randomAuthenticated(shape);
  } catch (...) {
    abort();
    throw;
  }
}

NdArrayRef SpdzWiseState::multiplyPublic(const NdArrayRef& x,
                                         const NdArrayRef& y) const {
  requireRunning();
  if (isRing()) return ring_->multiplyPublic(x, y);
  validateInternal(x);
  RequirePublic(y);
  SPU_ENFORCE(x.shape() == y.shape(), "multiply shape mismatch");
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> a(x);
  const NdArrayView<uint64_t> b(y);
  NdArrayView<AuthShare> out(result);
  for (int64_t i = 0; i < x.numel(); ++i) {
    for (size_t c = 0; c < 4; ++c) {
      out[i][c] = Mul(a[i][c], FromRing(b[i]));
    }
  }
  return result;
}

NdArrayRef SpdzWiseState::multiply(const NdArrayRef& x, const NdArrayRef& y) {
  requireRunning();
  if (isRing()) {
    try {
      return ring_->multiply(x, y);
    } catch (...) {
      abort();
      throw;
    }
  }
  validateInternal(x);
  validateInternal(y);
  SPU_ENFORCE(x.shape() == y.shape(), "multiply shape mismatch");
  const NdArrayView<AuthShare> a(x), b(y);
  NdArrayRef result(x.eltype(), x.shape());
  auto* out = result.data<AuthShare>();
  MultiplyF61(
      mask_prg_.get(), link_, 2 * x.numel(),
      [&](int64_t i) { return i % 2 == 0 ? Values(a[i / 2]) : Macs(a[i / 2]); },
      [&](int64_t i) { return Values(b[i / 2]); },
      [&](int64_t i, const RepShare& product) {
        out[i / 2][2 * (i % 2)] = product[0];
        out[i / 2][2 * (i % 2) + 1] = product[1];
      },
      tag("multiply"));
  if (x.numel() != 0) {
    pending_.insert(pending_.end(), out, out + x.numel());
  }
  return result;
}

NdArrayRef SpdzWiseState::matmul(const NdArrayRef& x, const NdArrayRef& y,
                                 bool public_rhs) {
  requireRunning();
  if (isRing()) {
    try {
      return ring_->matmul(x, y, public_rhs);
    } catch (...) {
      abort();
      throw;
    }
  }
  validateInternal(x);
  if (public_rhs) {
    RequirePublic(y);
  } else {
    validateInternal(y);
  }
  SPU_ENFORCE(x.ndim() == 2 && y.ndim() == 2 && x.shape()[1] == y.shape()[0],
              "matmul requires compatible matrices");
  const int64_t m = x.shape()[0], k = x.shape()[1], n = y.shape()[1];
  NdArrayRef result(x.eltype(), {m, n});
  NdArrayView<AuthShare> out(result);
  const NdArrayView<AuthShare> a(x);
  if (public_rhs) {
    const NdArrayView<uint64_t> b(y);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        out[i * n + j] = {};
        for (int64_t t = 0; t < k; ++t) {
          const auto scalar = FromRing(b[t * n + j]);
          for (size_t c = 0; c < 4; ++c) {
            out[i * n + j][c] =
                Add(out[i * n + j][c], Mul(a[i * k + t][c], scalar));
          }
        }
      }
    }
    return result;
  }
  const NdArrayView<AuthShare> b(y);
  const auto masks = RandomF61Rep(mask_prg_.get(), 2 * result.numel());
  std::vector<uint64_t> first(masks.size());
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const size_t pos = 2 * (i * n + j);
      for (int64_t t = 0; t < k; ++t) {
        first[pos] = Add(first[pos], LocalF61Product(Values(a[i * k + t]),
                                                     Values(b[t * n + j])));
        first[pos + 1] =
            Add(first[pos + 1],
                LocalF61Product(Macs(a[i * k + t]), Values(b[t * n + j])));
      }
      for (size_t c = 0; c < 2; ++c) {
        first[pos + c] =
            Add(first[pos + c], Sub(masks[pos + c][0], masks[pos + c][1]));
      }
    }
  }
  const auto second = exchange(first, "matmul");
  for (int64_t i = 0; i < result.numel(); ++i) {
    out[i] = {first[2 * i], second[2 * i], first[2 * i + 1], second[2 * i + 1]};
    pending_.push_back(out[i]);
  }
  return result;
}

namespace field {

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->publicValue(in);
}

NdArrayRef Negate::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->negate(in);
}

NdArrayRef AddAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->add(x, y);
}

NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  auto* state = ctx->getState<SpdzWiseState>();
  return state->add(x, state->publicValue(y));
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->multiply(x, y);
}

NdArrayRef MulAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->multiplyPublic(x, y);
}

NdArrayRef MatmulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->matmul(x, y, false);
}

NdArrayRef MatmulAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->matmul(x, y, true);
}

void registerArithmeticKernels(Object* object) {
  object->regKernel<P2A, Negate, AddAA, AddAP, MulAA, MulAP, MatmulAA,
                    MatmulAP>();
}

}  // namespace field
}  // namespace spu::mpc::spdzwise
