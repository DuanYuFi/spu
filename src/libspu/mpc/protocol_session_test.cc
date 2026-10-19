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

#include "libspu/mpc/protocol_session.h"

#include "gtest/gtest.h"

#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc {
namespace {

class TestSession final : public ProtocolSession {
 protected:
  void onBegin(const ExecutableProto&, const std::vector<Value>&) override {}
  Type exportType(const std::string&, const Value& value) const override {
    return value.storage_type();
  }
  void onAbort() noexcept override { aborted = true; }

 public:
  bool fail_check = false;
  bool aborted = false;
  size_t checks = 0;
  void validateExternalInput(const Value&) const override { requireHealthy(); }
};

class TestFinalize final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "finalize_protocol"; }
  void evaluate(KernelEvalContext* ctx) const override {
    auto* state = ctx->getState<TestSession>();
    ++state->checks;
    SPU_ENFORCE(!state->fail_check, "injected check failure");
  }
};

RuntimeConfig Config() {
  RuntimeConfig config;
  config.protocol = REF2K;
  config.field = FM64;
  return config;
}

Value PublicValue() {
  NdArrayRef data(makeType<Pub2kTy>(FM64), {1});
  NdArrayView<uint64_t>{data}[0] = 42;
  return Value(data, DT_I64);
}

TEST(ProtocolSession, LegacyContextHasNoHooks) {
  SPUContext ctx(Config(), nullptr);
  EXPECT_EQ(getProtocolSession(&ctx), nullptr);
  ProtocolRunGuard guard(&ctx);
  EXPECT_NO_THROW(guard.commit());
}

TEST(ProtocolSession, ExportIsInvisibleUntilCommitAndInvalidatedOnSet) {
  SPUContext ctx(Config(), nullptr);
  ctx.prot()->addState<TestSession>();
  ctx.prot()->regKernel<TestFinalize>();
  auto* state = ctx.prot()->getState<TestSession>();
  ProtocolRunGuard guard(&ctx);
  state->begin(ExecutableProto{}, {});
  EXPECT_ANY_THROW(state->exportedValue("x"));
  state->prepareOutputs(&ctx, {"x"}, {PublicValue()});
  EXPECT_ANY_THROW(state->exportedValue("x"));
  guard.commit();
  EXPECT_NO_THROW(state->exportedValue("x"));
  EXPECT_EQ(state->checks, 1);
  state->invalidateExport("x");
  EXPECT_ANY_THROW(state->exportedValue("x"));
}

TEST(ProtocolSession, CheckFailurePermanentlyDisablesExports) {
  SPUContext ctx(Config(), nullptr);
  ctx.prot()->addState<TestSession>();
  ctx.prot()->regKernel<TestFinalize>();
  auto* state = ctx.prot()->getState<TestSession>();
  {
    ProtocolRunGuard guard(&ctx);
    state->begin(ExecutableProto{}, {});
    state->prepareOutputs(&ctx, {"previous"}, {PublicValue()});
    guard.commit();
  }
  state->fail_check = true;
  state->begin(ExecutableProto{}, {});
  EXPECT_ANY_THROW(state->prepareOutputs(&ctx, {"next"}, {PublicValue()}));
  EXPECT_TRUE(state->failed());
  EXPECT_TRUE(state->aborted);
  EXPECT_ANY_THROW(state->exportedValue("previous"));
  EXPECT_ANY_THROW(state->exportedValue("next"));
  EXPECT_ANY_THROW(state->begin(ExecutableProto{}, {}));
}

TEST(ProtocolSession, MissingFinalizerIsNotSilentlySkipped) {
  SPUContext ctx(Config(), nullptr);
  ctx.prot()->addState<TestSession>();
  auto* state = ctx.prot()->getState<TestSession>();
  state->begin(ExecutableProto{}, {});
  EXPECT_ANY_THROW(state->prepareOutputs(&ctx, {}, {}));
  EXPECT_TRUE(state->failed());
}

TEST(ProtocolSession, EmptyOutputsStillFinalize) {
  SPUContext ctx(Config(), nullptr);
  ctx.prot()->addState<TestSession>();
  ctx.prot()->regKernel<TestFinalize>();
  auto* state = ctx.prot()->getState<TestSession>();
  ProtocolRunGuard guard(&ctx);
  state->begin(ExecutableProto{}, {});
  state->prepareOutputs(&ctx, {}, {});
  guard.commit();
  EXPECT_EQ(state->checks, 1);
}

TEST(ProtocolSession, InvalidOutputTransactionPoisonsSession) {
  SPUContext ctx(Config(), nullptr);
  ctx.prot()->addState<TestSession>();
  ctx.prot()->regKernel<TestFinalize>();
  auto* state = ctx.prot()->getState<TestSession>();
  state->begin(ExecutableProto{}, {});
  EXPECT_ANY_THROW(state->prepareOutputs(&ctx, {"missing"}, {}));
  EXPECT_TRUE(state->failed());
  EXPECT_ANY_THROW(state->commit());
}

TEST(ProtocolSession, ExecutionExceptionAbortsWithoutNetworkInDestructor) {
  SPUContext ctx(Config(), nullptr);
  ctx.prot()->addState<TestSession>();
  auto* state = ctx.prot()->getState<TestSession>();
  {
    ProtocolRunGuard guard(&ctx);
    state->begin(ExecutableProto{}, {});
    // Leaving the scope without commit models parsing, evaluation, or IO error.
  }
  EXPECT_TRUE(state->failed());
  EXPECT_TRUE(state->aborted);
}

}  // namespace
}  // namespace spu::mpc
