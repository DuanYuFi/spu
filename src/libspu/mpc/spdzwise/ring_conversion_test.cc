// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include "libspu/mpc/spdzwise/ring_conversion.h"

#include <random>
#include "gtest/gtest.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
class BooleanTestAccess {
 public:
  static void corrupt(SpdzWiseState* state, size_t party, bool consistent) {
    auto& triple = state->boolean_->pending_batches_.front().front();
    if (state->rank() == party) triple[2][0] ^= 1;
    if (consistent && state->rank() == (party + 2) % 3) triple[2][1] ^= 1;
  }
};
namespace {
RuntimeConfig Config() {
  RuntimeConfig c(SPDZWISE, FM64);
  c.spdzwise_config.domain = SpdzWiseConfig::RING;
  return c;
}
NdArrayRef Plain(int64_t n) {
  NdArrayRef x(makeType<RingTy>(FM64), {n});
  std::mt19937_64 random(1234);
  for (int64_t i = 0; i < n; ++i) {
    const auto bit = size_t(i / 2) % 64;
    x.data<uint64_t>()[i] = i < 128 ? (i % 2 ? (uint64_t(1) << bit) - 1 : uint64_t(1) << bit) : random();
  }
  if (n) x.data<uint64_t>()[n - 1] = ~uint64_t(0);
  return x;
}
void Begin(SpdzWiseState* state, const std::vector<Value>& inputs = {}) {
  std::vector<std::string> names(inputs.size());
  for (size_t i = 0; i < names.size(); ++i) names[i] = std::to_string(i);
  state->begin(ExecutableProto("a2b-test", names, {"out"}, "test"), inputs);
}
TEST(RingConversion, BoundariesShapesStridesAndAccumulatedSmallCalls) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const std::vector<int64_t> sizes{0, 1, 63, 64, 65, 130, 1025};
  std::vector<NdArrayRef> plains;
  std::vector<std::vector<NdArrayRef>> shares;
  for (auto n : sizes) {
    plains.push_back(Plain(n));
    shares.push_back(io.toShares(plains.back(), VIS_SECRET));
  }
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    std::vector<Value> inputs;
    for (const auto& batch : shares) inputs.emplace_back(batch[link->Rank()], DT_I64);
    Begin(state, inputs);
    std::vector<Value> outputs;
    std::vector<std::string> names;
    for (size_t j = 0; j < sizes.size(); ++j) {
      auto x = import_s(&ctx, inputs[j]);
      if (sizes[j] == 130) x = Value(x.data().reshape({65, 2}).transpose({1, 0}), x.dtype());
      auto y = a2b(&ctx, x);
      EXPECT_EQ(y.shape(), x.shape());
      EXPECT_EQ(y.dtype(), x.dtype());
      EXPECT_EQ(y.storage_type(), makeType<BShrTy>(state->session(), 64));
      EXPECT_EQ(state->booleanChecks(), 0);
      outputs.push_back(y);
      names.push_back(std::to_string(j));
    }
    EXPECT_ANY_THROW(state->exportValue(outputs.back().data(), makeType<BOutputTy>(link->Rank(), 64, "test")));
    state->finalize();
    EXPECT_EQ(state->booleanChecks(), 1);
    EXPECT_EQ(state->pendingBooleanBits(), 0);
    state->prepareOutputs(&ctx, names, outputs);
    guard.commit();
    std::vector<NdArrayRef> result;
    for (const auto& name : names) result.push_back(state->exportedValue(name).data());
    return result;
  });
  for (size_t j = 0; j < sizes.size(); ++j) {
    const auto actual = io.fromShares({results[0][j], results[1][j], results[2][j]});
    auto expected = plains[j];
    if (sizes[j] == 130) expected = expected.reshape({65, 2}).transpose({1, 0});
    const NdArrayView<uint64_t> view(expected);
    for (int64_t i = 0; i < sizes[j]; ++i) {
      ASSERT_EQ(actual.data<uint64_t>()[i], view[i]) << j << ":" << i;
      for (size_t rank = 0; rank < 3; ++rank)
        ASSERT_EQ(results[rank][j].data<RepShare>()[i][1], results[(rank + 1) % 3][j].data<RepShare>()[i][0]);
    }
  }
}

TEST(RingConversion, ArithmeticCompositionHighLimbsAndBooleanConsumer) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto plain = Plain(1025);
  const auto shares = io.toShares(plain, VIS_SECRET);
  const auto result = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Value input(shares[link->Rank()], DT_U64);
    Begin(state, {input});
    auto x = import_s(&ctx, input);
    auto square = mul_aa(&ctx, x, x);
    square.setDtype(x.dtype());
    bool high = false;
    for (int64_t i = 0; i < square.numel(); ++i)
      high |= (square.data().data<ring::AuthShare>()[i][0] >> 64) != 0;
    EXPECT_TRUE(high);
    const auto before = state->arithmeticChecks();
    auto b = a2b(&ctx, square);
    auto z = and_bb(&ctx, b, b);
    z.setDtype(b.dtype());
    EXPECT_EQ(state->arithmeticChecks(), before);
    state->finalize();
    EXPECT_EQ(state->arithmeticChecks(), before + 1);
    EXPECT_EQ(state->booleanChecks(), 1);
    state->prepareOutputs(&ctx, {"out"}, {z});
    guard.commit();
    return state->exportedValue("out").data();
  });
  const auto actual = io.fromShares(result);
  for (int64_t i = 0; i < plain.numel(); ++i) {
    const auto x = plain.data<uint64_t>()[i];
    EXPECT_EQ(actual.data<uint64_t>()[i], x * x);
  }
}
TEST(RingConversion, WrongDomainSessionAndLifecycleRejected) {
  for (int fault = 0; fault < 4; ++fault) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      if (fault != 3) Begin(state);
      const Type type = fault == 0 ? makeType<AShrTy>(state->session()) :
                        fault == 1 ? makeType<ring::AShrTy>("wrong-session") :
                        fault == 2 ? makeType<ring::InputTy>(link->Rank()) :
                                     makeType<ring::AShrTy>("no-session");
      NdArrayRef input(type, {0});
      EXPECT_ANY_THROW(state->a2b(input));
      EXPECT_TRUE(state->failed());
      EXPECT_ANY_THROW(state->exportedValue("out"));
    });
  }
}
TEST(RingConversion, EmptyNeedsNoAndAndSmallCallKeepsExistingCheckThreshold) {
  for (int64_t n : {0, 1}) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      Begin(state);
      auto x = state->publicValue(Plain(n).as(makeType<Pub2kTy>(FM64)));
      EXPECT_NO_THROW(state->a2b(x));
      if (n) {
        EXPECT_GT(state->pendingBooleanBits(), 0);
        EXPECT_ANY_THROW(state->finalize());
      } else {
        EXPECT_EQ(state->pendingBooleanBits(), 0);
        EXPECT_NO_THROW(state->finalize());
        EXPECT_EQ(state->booleanChecks(), 0);
      }
    });
  }
}
TEST(RingConversion, CorruptConversionAndRejectedAtEveryParty) {
  for (size_t party = 0; party < 3; ++party) {
    for (bool consistent : {false, true}) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto x = state->publicValue(Plain(1025).as(makeType<Pub2kTy>(FM64)));
        state->a2b(x);
        BooleanTestAccess::corrupt(state, party, consistent);
        try {
          state->finalize();
          FAIL() << "accepted corrupt conversion";
        } catch (const std::exception& e) {
          EXPECT_NE(std::string(e.what()).find(consistent ? "ABFL sacrifice failed" : "Boolean replicated copies disagree"), std::string::npos);
        }
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->exportedValue("out"));
      });
    }
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise
