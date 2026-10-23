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
#include "libspu/mpc/spdzwise/arithmetic.h"
#include "libspu/mpc/spdzwise/boolean.h"
#include "libspu/mpc/spdzwise/conversion.h"
#include "libspu/mpc/spdzwise/ring_arithmetic.h"
#include "libspu/mpc/spdzwise/ring_conversion.h"
#include "libspu/mpc/spdzwise/ring_truncation.h"
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

class DenyReveal final : public Kernel {
 public:
  void evaluate(KernelEvalContext*) const override {
    SPU_THROW("SPDZWISE v1 only exports checked shares to the external client");
  }
};

class RandP final : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_p"; }
  Kind kind() const override { return Kind::Dynamic; }
  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override {
    return ctx->getState<SpdzWiseState>()->randomPublic(shape);
  }
};

// Reuse standard kernels through Object's existing registration API, while
// keeping SPDZwise's public-random implementation local to this protocol.
class StandardKernel final : public Kernel {
  std::shared_ptr<const Object> owner_;
  const Kernel* kernel_;

 public:
  StandardKernel(std::shared_ptr<const Object> owner, const std::string& name)
      : owner_(std::move(owner)), kernel_(owner_->getKernel(name)) {}
  Kind kind() const override { return kernel_->kind(); }
  ce::CExpr latency() const override { return kernel_->latency(); }
  ce::CExpr comm() const override { return kernel_->comm(); }
  float getCommTolerance() const override {
    return kernel_->getCommTolerance();
  }
  void evaluate(KernelEvalContext* ctx) const override {
    kernel_->evaluate(ctx);
  }
};

void registerCommonKernels(SPUContext* ctx) {
  ctx->prot()->addState<Z2kState>(FM64);
  const auto standard = std::make_shared<Object>("spdzwise/standard");
  regPV2kKernels(standard.get());
  for (const auto& name : standard->getKernelNames()) {
    if (name != RandP::kBindName()) {
      ctx->prot()->regKernel(name,
                             std::make_unique<StandardKernel>(standard, name));
    }
  }
  ctx->prot()->regKernel<RandP>();
  regStandardShapeOps(ctx);
  ctx->prot()->regKernel<Import, Export, Finalize>();
  for (const auto* name : {"a2p", "b2p", "a2v", "b2v", "s2p", "s2v"}) {
    ctx->prot()->regKernel<DenyReveal>(name);
  }
}

}  // namespace

void registerFieldArithmeticPrototype(
    SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& link) {
  SPU_ENFORCE(ctx->config().field == FM64,
              "SPDZWISE FIELD and RING domains require FM64");
  SPU_ENFORCE(ctx->config().spdzwise_config.domain == SpdzWiseConfig::FIELD,
              "SPDZWISE field arithmetic requires domain FIELD");
  registerTypes();
  ctx->prot()->addState<SpdzWiseState>(ctx->config(), link);
  registerCommonKernels(ctx);
  field::registerArithmeticKernels(ctx->prot());
  registerBooleanKernels(ctx->prot());
  field::registerConversionKernels(ctx->prot());
  // Domain-specific PPML kernels (truncation, comparison) remain unimplemented.
}

void registerRingArithmeticPrototype(
    SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& link) {
  SPU_ENFORCE(ctx->config().field == FM64,
              "SPDZWISE FIELD and RING domains require FM64");
  SPU_ENFORCE(ctx->config().spdzwise_config.domain == SpdzWiseConfig::RING,
              "SPDZWISE ring arithmetic requires domain RING");
  registerTypes();
  ctx->prot()->addState<SpdzWiseState>(ctx->config(), link);
  registerCommonKernels(ctx);
  ring::registerArithmeticKernels(ctx->prot());
  ctx->prot()
      ->regKernel<ring::TruncASelector, ring::A2B, ring::B2A, ring::MsbA2B,
                  ring::MulA1B>();
  registerBooleanKernels(ctx->prot());
}

void registerArithmeticPrototype(
    SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& link) {
  switch (ctx->config().spdzwise_config.domain) {
    case SpdzWiseConfig::FIELD:
      return registerFieldArithmeticPrototype(ctx, link);
    case SpdzWiseConfig::RING:
      return registerRingArithmeticPrototype(ctx, link);
    default:
      SPU_THROW("Invalid SPDZWISE arithmetic domain {}",
                static_cast<int>(ctx->config().spdzwise_config.domain));
  }
}

void regSpdzWiseProtocol(SPUContext* ctx,
                         const std::shared_ptr<yacl::link::Context>& link) {
  SPU_ENFORCE(ctx->config().spdzwise_config.domain == SpdzWiseConfig::RING,
              "SPDZWISE runtime requires spdzwise_config.domain=RING; "
              "FIELD only supports the arithmetic development entrypoint");
  registerRingArithmeticPrototype(ctx, link);
}

}  // namespace spu::mpc::spdzwise
