// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include "libspu/mpc/spdzwise/ring_conversion.h"

#include <random>
#include "gtest/gtest.h"
#include "libspu/kernel/hal/integer.h"
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
  static void corruptMsb(SpdzWiseState* state, size_t party, bool consistent,
                         bool padding) {
    // Last layer exercises the new carry tree, including its padded SIMD lane.
    auto& triple = state->boolean_->pending_batches_.back().back();
    const uint64_t bit = uint64_t{1} << (padding ? 63 : 0);
    if (state->rank() == party) triple[2][0] ^= bit;
    if (consistent && state->rank() == (party + 2) % 3) triple[2][1] ^= bit;
  }
  static void corruptBit(SpdzWiseState* state, size_t party, bool consistent) {
    auto& triple = state->boolean_->pending_.front();
    if (state->rank() == party) triple[2][0] ^= 1;
    if (consistent && state->rank() == (party + 2) % 3) triple[2][1] ^= 1;
  }
};
namespace ring {
class ConversionTestAccess {
 public:
  static RepShare key(SpdzWiseState* s) { return s->ring_->key_; }
  static size_t gates(SpdzWiseState* s) { return s->b2a_and_bits_; }
  static void observeOpening(SpdzWiseState* s, bool& opened,
                             bool* bit_opened = nullptr) {
    auto original = s->boolean_->tag_;
    s->boolean_->tag_ = [original, &opened, bit_opened](std::string_view name) {
      if (name == "b2a-open" || name == "bit2a-open") opened = true;
      if (bit_opened && name == "bit2a-open") *bit_opened = true;
      return original(name);
    };
  }
  static void tamperOpening(SpdzWiseState* s, std::shared_ptr<bool> active) {
    auto original = s->boolean_->tag_;
    s->boolean_->tag_ = [original, active](std::string_view name) {
      *active = name == "b2a-open" || name == "bit2a-open";
      return original(name);
    };
  }
  static void fault(SpdzWiseState* s, size_t party, int fault,
                    int64_t count = 1025) {
    auto mask = s->ring_->randomAuthenticated({count});
    auto masked = s->a2b(mask);  // Valid B2A masked result for x=0.
    const bool first = s->rank() == party;
    const bool second = s->rank() == (party + 2) % 3;
    if (fault < 2) {
      auto& record = s->ring_->pending_.front();
      if (first) record[2] ^= 1;
      if (fault == 1 && second) record[3] ^= 1;
    } else if (fault < 4) {
      auto& record = s->boolean_->pending_batches_.front().front();
      if (first) record[2][0] ^= 1;
      if (fault == 3 && second) record[2][1] ^= 1;
    } else if (fault == 5) {
      // The final SIMD block has one active lane; bit 63 is padding.
      auto& record = s->boolean_->pending_batches_.front().back();
      if (first) record[2][0] ^= uint64_t{1} << 63;
      if (second) record[2][1] ^= uint64_t{1} << 63;
    } else if (first) {
      // Alter the share encoded into the opening payload, after the AND
      // snapshots were frozen. Duplicate-holder checking must reject it.
      masked.data<spdzwise::RepShare>()[0][0] ^= 1;
    }
    try {
      s->finishB2A(mask, masked);
    } catch (...) {
      s->abort();
      throw;
    }
  }
};
}  // namespace ring
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
    auto bit = msb_a2b(&ctx, square);
    auto z = and_bb(&ctx, b, b);
    z.setDtype(b.dtype());
    EXPECT_EQ(state->arithmeticChecks(), before);
    state->finalize();
    EXPECT_EQ(state->arithmeticChecks(), before + 1);
    EXPECT_EQ(state->booleanChecks(), 1);
    state->prepareOutputs(&ctx, {"out", "msb"}, {z, bit});
    guard.commit();
    return std::array<NdArrayRef, 2>{state->exportedValue("out").data(),
                                   state->exportedValue("msb").data()};
  });
  const auto actual = io.fromShares({result[0][0], result[1][0], result[2][0]});
  const auto msb = io.fromShares({result[0][1], result[1][1], result[2][1]});
  for (int64_t i = 0; i < plain.numel(); ++i) {
    const auto x = plain.data<uint64_t>()[i];
    EXPECT_EQ(actual.data<uint64_t>()[i], x * x);
    EXPECT_EQ(msb.data<uint64_t>()[i], (x * x) >> 63);
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
TEST(RingConversion, EmptyNeedsNoAndAndSmallCallUsesZeroPadding) {
  for (int64_t n : {0, 1}) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      Begin(state);
      auto x = state->publicValue(Plain(n).as(makeType<Pub2kTy>(FM64)));
      EXPECT_NO_THROW(state->a2b(x));
      EXPECT_EQ(state->pendingBooleanBits(), n ? 125 * 64 : 0);
      EXPECT_NO_THROW(state->finalize());
      EXPECT_EQ(state->pendingBooleanBits(), 0);
      EXPECT_EQ(state->booleanChecks(), n ? 1 : 0);
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

TEST(RingConversion, MsbBoundariesShapesStridesAndBooleanOutput) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const std::vector<int64_t> sizes{0, 1, 63, 64, 65, 130, 1025};
  std::vector<NdArrayRef> plains;
  std::vector<std::vector<NdArrayRef>> shares;
  for (auto n : sizes) {
    auto plain = Plain(n);
    if (n >= 130) {
      plain.data<uint64_t>()[128] = 0xaaaaaaaaaaaaaaaaULL;
      plain.data<uint64_t>()[129] = 0x5555555555555555ULL;
    }
    plains.push_back(plain);
    shares.push_back(io.toShares(plain, VIS_SECRET));
  }
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    EXPECT_NE(dynamic_cast<ring::MsbA2B*>(ctx.prot()->getKernel("msb_a2b")),
              nullptr);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    std::vector<Value> inputs, outputs;
    std::vector<std::string> names;
    for (const auto& batch : shares)
      inputs.emplace_back(batch[link->Rank()], DT_I64);
    Begin(state, inputs);
    for (size_t j = 0; j < sizes.size(); ++j) {
      auto x = import_s(&ctx, inputs[j]);
      if (sizes[j] == 1) x = Value(x.data().reshape({}), x.dtype());
      if (sizes[j] == 130)
        x = Value(x.data().reshape({65, 2}).transpose({1, 0}), x.dtype());
      const auto pending = state->pendingBooleanBits();
      const auto checks = state->arithmeticChecks();
      const auto bytes = link->GetStats()->sent_bytes.load();
      const auto bit = msb_a2b(&ctx, x);
      EXPECT_EQ(bit.shape(), x.shape());
      EXPECT_EQ(bit.dtype(), x.dtype());
      EXPECT_EQ(bit.storage_type(), makeType<BShrTy>(state->session(), 1));
      EXPECT_EQ(state->arithmeticChecks(), checks);
      EXPECT_EQ(state->booleanChecks(), 0);
      EXPECT_EQ(state->pendingBooleanBits() - pending,
                241 * ((sizes[j] + 63) / 64) * 64);
      if (!sizes[j]) {
        EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
      }
      // Exercise a Boolean consumer and compare against full A2B independently.
      outputs.push_back(and_bb(&ctx, bit, bit).setDtype(DT_I1));
      names.push_back("msb-" + std::to_string(j));
      outputs.push_back(a2b(&ctx, x));
      names.push_back("full-" + std::to_string(j));
    }
    EXPECT_ANY_THROW(state->exportValue(
        outputs[outputs.size() - 2].data(),
        makeType<BOutputTy>(link->Rank(), 1, "test")));
    state->finalize();
    EXPECT_EQ(state->booleanChecks(), 1);
    EXPECT_EQ(state->pendingBooleanBits(), 0);
    state->prepareOutputs(&ctx, names, outputs);
    guard.commit();
    std::vector<NdArrayRef> result;
    for (const auto& name : names)
      result.push_back(state->exportedValue(name).data());
    return result;
  });
  for (size_t j = 0; j < sizes.size(); ++j) {
    const auto actual = io.fromShares(
        {results[0][2 * j], results[1][2 * j], results[2][2 * j]});
    const auto full = io.fromShares(
        {results[0][2 * j + 1], results[1][2 * j + 1], results[2][2 * j + 1]});
    auto expected = plains[j];
    if (sizes[j] == 130)
      expected = expected.reshape({65, 2}).transpose({1, 0});
    const NdArrayView<uint64_t> view(expected);
    for (int64_t i = 0; i < sizes[j]; ++i) {
      EXPECT_EQ(actual.data<uint64_t>()[i], view[i] >> 63);
      EXPECT_EQ(actual.data<uint64_t>()[i], full.data<uint64_t>()[i] >> 63);
      for (size_t rank = 0; rank < 3; ++rank) {
        const auto& share = results[rank][2 * j].data<RepShare>()[i];
        EXPECT_LE(share[0], 1);
        EXPECT_EQ(share[1], results[(rank + 1) % 3][2 * j].data<RepShare>()[i][0]);
      }
    }
  }
}

TEST(RingConversion, MsbHalComparisonSecretAndPublicOperands) {
  constexpr int64_t n = 1025;
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  auto px = Plain(n), py = Plain(n);
  std::mt19937_64 random(20261022);
  const std::array<int64_t, 5> edges{0, -1, 1, -(int64_t{1} << 62),
                                    (int64_t{1} << 62) - 1};
  for (int64_t i = 0; i < n; ++i) {
    const auto x = i < 25 ? edges[i / 5]
                         : int64_t(random() >> 1) - (int64_t{1} << 62);
    const auto y = i < 25 ? edges[i % 5]
                         : int64_t(random() >> 1) - (int64_t{1} << 62);
    px.data<uint64_t>()[i] = x;
    py.data<uint64_t>()[i] = y;
    if (i >= 513) {
      px.data<uint64_t>()[i] = random();
      py.data<uint64_t>()[i] = random();
    }
  }
  // Overflowing differences retain the original HAL modular MSB semantics.
  px.data<uint64_t>()[513] = uint64_t{1} << 63;
  py.data<uint64_t>()[513] = 1;
  px.data<uint64_t>()[514] = (uint64_t{1} << 63) - 1;
  py.data<uint64_t>()[514] = ~uint64_t{0};
  const auto sx = io.toShares(px, VIS_SECRET), sy = io.toShares(py, VIS_SECRET);
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Value ix(sx[link->Rank()], DT_I64), iy(sy[link->Rank()], DT_I64);
    Begin(state, {ix, iy});
    const auto x = import_s(&ctx, ix), y = import_s(&ctx, iy);
    const Value xp(px.as(makeType<Pub2kTy>(FM64)), DT_I64);
    const Value yp(py.as(makeType<Pub2kTy>(FM64)), DT_I64);
    std::vector<Value> outputs{
        kernel::hal::i_less(&ctx, x, y), kernel::hal::i_less(&ctx, x, yp),
        kernel::hal::i_less(&ctx, xp, y)};
    for (const auto& out : outputs) {
      EXPECT_EQ(out.dtype(), DT_I1);
      EXPECT_EQ(out.storage_type(), makeType<BShrTy>(state->session(), 1));
    }
    EXPECT_EQ(state->booleanChecks(), 0);
    state->prepareOutputs(&ctx, {"ss", "sp", "ps"}, outputs);
    guard.commit();
    std::vector<NdArrayRef> result;
    for (const auto* name : {"ss", "sp", "ps"})
      result.push_back(state->exportedValue(name).data());
    return result;
  });
  for (size_t j = 0; j < 3; ++j) {
    const auto actual = io.fromShares({results[0][j], results[1][j], results[2][j]});
    for (int64_t i = 0; i < n; ++i) {
      const auto x = px.data<uint64_t>()[i], y = py.data<uint64_t>()[i];
      EXPECT_EQ(actual.data<uint64_t>()[i], (x - y) >> 63);
      if (i < 513) {
        EXPECT_EQ(actual.data<uint64_t>()[i], int64_t(x) < int64_t(y));
      }
    }
  }
}

TEST(RingConversion, MsbRejectsWrongDomainSessionAndLifecycle) {
  for (int fault = 0; fault < 5; ++fault) {
    utils::simulate(3, [&](const auto& link) {
      auto config = Config();
      if (fault == 4) config.spdzwise_config.domain = SpdzWiseConfig::FIELD;
      SPUContext ctx(config, link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      if (fault != 3) Begin(state);
      const Type type = fault == 0 ? makeType<AShrTy>(state->session()) :
                        fault == 1 ? makeType<ring::AShrTy>("wrong-session") :
                        fault == 2 ? makeType<ring::InputTy>(link->Rank()) :
                                     makeType<ring::AShrTy>("no-session");
      EXPECT_ANY_THROW(state->msbA2B(NdArrayRef(type, {0})));
      EXPECT_TRUE(state->failed());
    });
  }
}

TEST(RingConversion, MsbRejectsCorruptCarryAndPaddingAtEveryParty) {
  for (size_t party = 0; party < 3; ++party) {
    for (int fault = 0; fault < 3; ++fault) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto x = state->publicValue(Plain(65).as(makeType<Pub2kTy>(FM64)));
        state->msbA2B(x);
        BooleanTestAccess::corruptMsb(state, party, fault != 0, fault == 2);
        EXPECT_ANY_THROW(state->finalize());
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->exportedValue("out"));
      });
    }
  }
}

TEST(RingConversion, B2ACheckThresholdPreservesOpeningChecksAndGateCounts) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto plain = Plain(65);
  const auto shares = io.toBooleanShares(plain, 64);
  const auto results = utils::simulate(3, [&](const auto& link) {
    auto config = Config();
    config.spdzwise_config.check_threshold = 32;
    SPUContext ctx(config, link);
    registerArithmeticPrototype(&ctx, link);
    auto* s = ctx.prot()->getState<SpdzWiseState>();
    Value input(shares[link->Rank()], DT_U64);
    Begin(s, {input});
    const auto x = import_s(&ctx, input);
    const auto y = b2a(&ctx, x);
    EXPECT_EQ(s->arithmeticChecks(), 1);
    EXPECT_GT(s->booleanChecks(), 1);
    EXPECT_EQ(s->pendingBooleanBits(), 0);
    EXPECT_EQ(ring::ConversionTestAccess::gates(s), 188 * 2 * 64);
    s->prepareOutputs(&ctx, {"out"}, {y});
    s->commit();
    return s->exportedValue("out").data();
  });
  const auto actual = io.fromShares(results);
  for (int64_t i = 0; i < plain.numel(); ++i)
    EXPECT_EQ(actual.data<uint64_t>()[i], plain.data<uint64_t>()[i]);
}

TEST(RingConversion, B2AWidthsShapesAndAuthenticatedOutput) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  struct Result {
    NdArrayRef internal, external;
    ring::RepShare key;
  };
  for (size_t width : {1U, 5U, 8U, 16U, 32U, 63U, 64U}) {
    auto plain = Plain(1026);
    for (int64_t i = 0; i < plain.numel(); ++i)
      plain.data<uint64_t>()[i] &= BooleanMask(width);
    plain.data<uint64_t>()[0] = 0;
    const auto shares = io.toBooleanShares(plain, width);
    const auto results = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      Value input(shares[link->Rank()], DT_U64);
      Begin(state, {input});
      auto x = import_s(&ctx, input);
      x = Value(x.data().reshape({513, 2}).transpose({1, 0}), x.dtype());
      EXPECT_FALSE(x.data().isCompact());
      bool opened = false, bit_opened = false;
      ring::ConversionTestAccess::observeOpening(state, opened, &bit_opened);
      const auto y = b2a(&ctx, x);
      EXPECT_EQ(y.dtype(), x.dtype());
      EXPECT_EQ(y.shape(), x.shape());
      EXPECT_EQ(y.storage_type(), makeType<ring::AShrTy>(state->session()));
      EXPECT_EQ(state->arithmeticChecks(), 1);
      EXPECT_TRUE(opened);
      EXPECT_EQ(bit_opened, width == 1);
      EXPECT_EQ(state->booleanChecks(), width == 1 ? 0 : 1);
      EXPECT_EQ(state->pendingBooleanBits(), 0);
      EXPECT_EQ(ring::ConversionTestAccess::gates(state),
                width == 1 ? 0 : 188 * 17 * 64);
      state->finalize();
      EXPECT_EQ(state->booleanChecks(), width == 1 ? 0 : 1);
      EXPECT_EQ(state->arithmeticChecks(), 1);
      const auto key = ring::ConversionTestAccess::key(state);
      // Check composability and the output MAC through an actual
      // multiplication.
      const auto z = mul_aa(&ctx, y, y);
      state->prepareOutputs(&ctx, {"out"}, {z});
      guard.commit();
      return Result{y.data(), state->exportedValue("out").data(), key};
    });
    ring::Word alpha = 0;
    std::vector<NdArrayRef> outputs;
    for (const auto& r : results) {
      alpha += r.key[0];
      outputs.push_back(r.external);
    }
    const auto actual = io.fromShares(outputs);
    const auto expected_array = plain.reshape({513, 2}).transpose({1, 0});
    const NdArrayView<uint64_t> expected(expected_array);
    for (int64_t i = 0; i < plain.numel(); ++i) {
      ring::Word value = 0, mac = 0;
      for (size_t rank = 0; rank < 3; ++rank) {
        const auto& a = results[rank].internal.data<ring::AuthShare>()[i];
        const auto& next =
            results[(rank + 1) % 3].internal.data<ring::AuthShare>()[i];
        EXPECT_EQ(a[1], next[0]);
        EXPECT_EQ(a[3], next[2]);
        value += a[0];
        mac += a[2];
      }
      ASSERT_EQ(uint64_t(value), expected[i]);
      if (width == 1) {
        ASSERT_EQ(value, ring::Word(expected[i]));
      }
      ASSERT_EQ(mac, alpha * value);
      ASSERT_EQ(actual.data<uint64_t>()[i], expected[i] * expected[i]);
    }
  }
}

TEST(RingConversion, B2ARoundtripConsumersAndFreshRepeatedMasks) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto plain = Plain(1025);
  const auto shares = io.toShares(plain, VIS_SECRET);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Value input(shares[link->Rank()], DT_I64);
    Begin(state, {input});
    auto a = import_s(&ctx, input);
    auto b = a2b(&ctx, a);
    b = and_bb(&ctx, b, b);
    b.setDtype(DT_I64);
    const auto y = b2a(&ctx, b);
    const auto z = b2a(&ctx, b);
    EXPECT_EQ(state->booleanChecks(), 2);
    EXPECT_EQ(state->arithmeticChecks(), 3);  // Input plus two fresh masks.
    EXPECT_NE(y.data().data<ring::AuthShare>()[0][0],
              z.data().data<ring::AuthShare>()[0][0]);
    state->prepareOutputs(&ctx, {"y", "z"}, {y, z});
    guard.commit();
    return std::array<NdArrayRef, 2>{state->exportedValue("y").data(),
                                     state->exportedValue("z").data()};
  });
  for (size_t j = 0; j < 2; ++j) {
    const auto actual =
        io.fromShares({outputs[0][j], outputs[1][j], outputs[2][j]});
    for (int64_t i = 0; i < plain.numel(); ++i)
      ASSERT_EQ(actual.data<uint64_t>()[i], plain.data<uint64_t>()[i]);
  }
}

TEST(RingConversion, B2AEmptyAndSmallCallsWithZeroPadding) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  for (size_t width : {1U, 64U}) {
    for (int64_t n : {0, 1, 63, 64, 65, 320, 321}) {
      auto plain = Plain(n);
      for (int64_t i = 0; i < n; ++i)
        plain.data<uint64_t>()[i] &= BooleanMask(width);
      const auto shares = io.toBooleanShares(plain, width);
      const auto outputs = utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        ProtocolRunGuard guard(&ctx);
        Value input(shares[link->Rank()], DT_I64);
        Begin(state, {input});
        const auto x = import_s(&ctx, input);
        bool opened = false;
        ring::ConversionTestAccess::observeOpening(state, opened);
        const auto bytes = link->GetStats()->sent_bytes.load();
        const auto y = b2a(&ctx, x);
        EXPECT_EQ(opened, n != 0);
        EXPECT_EQ(state->pendingBooleanBits(), 0);
        EXPECT_EQ(ring::ConversionTestAccess::gates(state),
                  width == 1 ? 0 : ((n + 63) / 64) * 64 * 188);
        EXPECT_EQ(state->booleanChecks(), n && width != 1 ? 1 : 0);
        if (!n) {
          EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
          EXPECT_EQ(state->booleanChecks(), 0);
          EXPECT_EQ(state->arithmeticChecks(), 0);
        }
        state->prepareOutputs(&ctx, {"out"}, {y});
        guard.commit();
        return state->exportedValue("out").data();
      });
      const auto actual = io.fromShares(outputs);
      for (int64_t i = 0; i < n; ++i)
        ASSERT_EQ(actual.data<uint64_t>()[i], plain.data<uint64_t>()[i]);
    }
  }
}

TEST(RingConversion, B2AOneBitRejectsUpstreamAndBeforeOpening) {
  for (size_t party = 0; party < 3; ++party) {
    for (bool consistent : {false, true}) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto plain = Plain(65).as(makeType<Pub2kTy>(FM64));
        for (int64_t i = 0; i < plain.numel(); ++i)
          plain.data<uint64_t>()[i] &= 1;
        const auto bit = state->publicBoolean(plain);
        const auto narrow = bit.as(makeType<BShrTy>(state->session(), 1));
        const auto input = state->booleanBinary(narrow, narrow, true, false);
        BooleanTestAccess::corruptBit(state, party, consistent);
        bool opened = false;
        ring::ConversionTestAccess::observeOpening(state, opened);
        EXPECT_ANY_THROW(state->b2a(input));
        EXPECT_FALSE(opened);
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->exportedValue("out"));
      });
    }
  }
}

TEST(RingConversion, B2AInvalidInputAndLifecycleRejected) {
  for (int fault = 0; fault < 6; ++fault) {
    utils::simulate(3, [&](const auto& link) {
      auto config = Config();
      if (fault == 4) config.spdzwise_config.domain = SpdzWiseConfig::FIELD;
      SPUContext ctx(config, link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      if (fault != 3) Begin(state);
      const Type type =
          fault == 0   ? makeType<ring::AShrTy>(state->session())
          : fault == 1 ? makeType<BShrTy>("wrong-session", 64)
          : fault == 2
              ? makeType<BInputTy>(link->Rank(), 64)
              : makeType<BShrTy>(fault == 3 ? "no-session" : state->session(),
                                 fault == 5 ? 1 : 64);
      NdArrayRef input(type, {fault == 5 ? 1 : 0});
      if (fault == 5) input.data<RepShare>()[0] = {2, 0};
      EXPECT_ANY_THROW(state->b2a(input));
      EXPECT_TRUE(state->failed());
      EXPECT_ANY_THROW(state->exportedValue("out"));
    });
  }
}

TEST(RingConversion, B2ARejectsFaultsBeforeOpeningOrOutput) {
  for (int64_t count : {1, 1025}) {
    for (size_t party = 0; party < 3; ++party) {
      for (int fault = 0; fault < 6; ++fault) {
        utils::simulate(3, [&](const auto& link) {
          SPUContext ctx(Config(), link);
          registerArithmeticPrototype(&ctx, link);
          auto* state = ctx.prot()->getState<SpdzWiseState>();
          Begin(state);
          bool opened = false;
          ring::ConversionTestAccess::observeOpening(state, opened);
          EXPECT_ANY_THROW(
              ring::ConversionTestAccess::fault(state, party, fault, count));
          EXPECT_EQ(opened, fault == 4);
          EXPECT_TRUE(state->failed());
          EXPECT_ANY_THROW(state->exportedValue("out"));
        });
      }
    }
  }
}
// Test-only channel decorator: mutate actual wire messages without adding
// fault-injection branches to the protocol implementation.
class B2AFaultChannel : public yacl::link::transport::IChannel {
  std::shared_ptr<yacl::link::transport::IChannel> inner_;
  std::shared_ptr<bool> active_;
  yacl::Buffer change(yacl::Buffer value) {
    if (*active_ && value.size()) {
      value.data<char>()[0] ^= 1;
    }
    return value;
  }

 public:
  B2AFaultChannel(std::shared_ptr<yacl::link::transport::IChannel> inner,
                  std::shared_ptr<bool> active)
      : inner_(std::move(inner)), active_(std::move(active)) {}
  void SendAsync(const std::string& k, yacl::Buffer v) override {
    inner_->SendAsync(k, change(std::move(v)));
  }
  void SendAsyncThrottled(const std::string& k, yacl::Buffer v) override {
    inner_->SendAsyncThrottled(k, change(std::move(v)));
  }
  void Send(const std::string& k, yacl::ByteContainerView v) override {
    auto b = change(yacl::Buffer(v));
    inner_->Send(k, yacl::ByteContainerView(b));
  }
  yacl::Buffer Recv(const std::string& k) override { return inner_->Recv(k); }
  void SetRecvTimeout(uint64_t t) override { inner_->SetRecvTimeout(t); }
  uint64_t GetRecvTimeout() const override { return inner_->GetRecvTimeout(); }
  void WaitLinkTaskFinish() override { inner_->WaitLinkTaskFinish(); }
  void Abort() override { inner_->Abort(); }
  void SetThrottleWindowSize(size_t s) override {
    inner_->SetThrottleWindowSize(s);
  }
  void TestSend(uint32_t t) override { inner_->TestSend(t); }
  void TestRecv() override { inner_->TestRecv(); }
  void SetChunkParallelSendSize(size_t s) override {
    inner_->SetChunkParallelSendSize(s);
  }
};
TEST(RingConversion, B2ARejectsOpeningWireEquivocation) {
  for (size_t width : {1U, 64U}) {
    for (size_t party = 0; party < 3; ++party) {
      utils::simulate(3, [&](const auto& base) {
        auto active = std::make_shared<bool>(false);
        yacl::link::ContextDesc desc;
        desc.id = base->Id();
        std::vector<std::shared_ptr<yacl::link::transport::IChannel>> channels(3);
        for (size_t i = 0; i < 3; ++i) {
          desc.parties.push_back({std::to_string(i), "memory"});
          if (i != base->Rank()) {
            channels[i] = base->GetChannel(i);
            // Corrupt only one recipient's payload: copies and ordered views
            // must bind the exact messages, not just the local reconstruction.
            if (base->Rank() == party && i == (party + 1) % 3)
              channels[i] =
                  std::make_shared<B2AFaultChannel>(channels[i], active);
          }
        }
        auto link = std::make_shared<yacl::link::Context>(desc, base->Rank(),
                                                          channels, nullptr);
        link->SetRecvTimeout(5000);
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto plain = Plain(1025).as(makeType<Pub2kTy>(FM64));
        for (int64_t i = 0; i < plain.numel(); ++i)
          plain.data<uint64_t>()[i] &= BooleanMask(width);
        auto x = state->publicBoolean(plain).as(
            makeType<BShrTy>(state->session(), width));
        ring::ConversionTestAccess::tamperOpening(state, active);
        try {
          state->b2a(x);
          FAIL() << "accepted opening equivocation";
        } catch (const std::exception& e) {
          EXPECT_NE(std::string(e.what()).find("Boolean transcript disagreement"),
                    std::string::npos);
        }
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->exportedValue("out"));
      });
    }
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise
