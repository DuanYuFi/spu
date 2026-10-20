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

#include "libspu/mpc/spdzwise/protocol.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/standard_shape/protocol.h"

namespace spu::mpc::spdzwise {
namespace {

class Import final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "import_s"; }
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    ctx->pushOutput(Value(
        ctx->getState<SpdzWiseState>()->importValue(in.data()), in.dtype()));
  }
};

class Export final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "export_s"; }
  void evaluate(KernelEvalContext* ctx) const override {
    const auto& in = ctx->getParam<Value>(0);
    const auto& type = ctx->getParam<Type>(1);
    ctx->pushOutput(
        Value(ctx->getState<SpdzWiseState>()->exportValue(in.data(), type),
              in.dtype()));
  }
};

class Finalize final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "finalize_protocol"; }
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->getState<SpdzWiseState>()->finalize();
  }
};

class P2A final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2a"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    return ctx->getState<SpdzWiseState>()->publicValue(in);
  }
};

class Negate final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_a"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    return ctx->getState<SpdzWiseState>()->negate(in);
  }
};

class AddAA final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_aa"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->add(x, y);
  }
};

class AddAP final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_ap"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    auto* state = ctx->getState<SpdzWiseState>();
    return state->add(x, state->publicValue(y));
  }
};

class MulAA final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_aa"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->multiply(x, y);
  }
};

class MulAP final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ap"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->multiplyPublic(x, y);
  }
};

class MatmulAA final : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_aa"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->matmul(x, y, false);
  }
};

class MatmulAP final : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_ap"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->matmul(x, y, true);
  }
};

class P2B final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2b"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    return ctx->getState<SpdzWiseState>()->publicBoolean(in);
  }
};
template <bool IsAnd, bool PublicRhs>
class BooleanBinary final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() {
    return IsAnd ? (PublicRhs ? "and_bp" : "and_bb")
                 : (PublicRhs ? "xor_bp" : "xor_bb");
  }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->booleanBinary(x, y, IsAnd,
                                                         PublicRhs);
  }
};

class DenyReveal final : public Kernel {
 public:
  void evaluate(KernelEvalContext*) const override {
    SPU_THROW("SPDZWISE v1 only exports checked shares to the external client");
  }
};

}  // namespace

void registerArithmeticPrototype(
    SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& link) {
  registerTypes();
  ctx->prot()->addState<SpdzWiseState>(ctx->config(), link);
  ctx->prot()->addState<Z2kState>(FM64);
  regPV2kKernels(ctx->prot());
  regStandardShapeOps(ctx);
  ctx->prot()
      ->regKernel<Import, Export, Finalize, P2A, Negate, AddAA, AddAP, MulAA,
                  MulAP, MatmulAA, MatmulAP, P2B, BooleanBinary<true, false>,
                  BooleanBinary<true, true>, BooleanBinary<false, false>,
                  BooleanBinary<false, true>>();
  for (const auto* name : {"a2p", "b2p", "a2v", "b2v", "s2p", "s2v"}) {
    ctx->prot()->regKernel<DenyReveal>(name);
  }
}

}  // namespace spu::mpc::spdzwise
