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

void registerCommonKernels(SPUContext* ctx) {
  ctx->prot()->addState<Z2kState>(FM64);
  regPV2kKernels(ctx->prot());
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
  ctx->prot()->regKernel<ring::TruncASelector, ring::A2B, ring::B2A,
                        ring::MsbA2B>();
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

}  // namespace spu::mpc::spdzwise
