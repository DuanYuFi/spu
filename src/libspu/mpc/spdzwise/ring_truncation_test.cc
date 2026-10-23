// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include "libspu/mpc/spdzwise/ring_truncation.h"

#include <limits>
#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/arithmetic.h"
#include "libspu/mpc/aby3/io.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
class BooleanTestAccess {
 public:
  static void observeOpening(SpdzWiseState* s, std::function<void()> observe) {
    auto original = s->boolean_->tag_;
    s->boolean_->tag_ = [original, observe](std::string_view name) {
      if (name == "b2a-open") observe();
      return original(name);
    };
  }
};
}  // namespace spu::mpc::spdzwise

namespace spu::mpc::spdzwise::ring {
// Only the test binary can inspect keys/material; no runtime reveal interface.
class TruncationTestAccess {
 public:
  static RepShare key(SpdzWiseState* s) { return s->ring_->key_; }
  static size_t pending(SpdzWiseState* s) { return s->ring_->pending_.size(); }
  static void checkBits(ArithmeticState& s, const NdArrayRef& b) {
    std::vector<uint64_t> coefficients;
    s.checkRandomBits(b, coefficients);
  }
  static void observeOpening(SpdzWiseState* s, std::function<void()> observe) {
    auto original = s->ring_->tag_;
    s->ring_->tag_ = [original, observe](std::string_view name) {
      if (name == "trunc-open") observe();
      return original(name);
    };
  }
  static void corruptMac(SpdzWiseState* s, size_t party, bool consistent) {
    auto& record = s->ring_->pending_.front();
    if (s->rank() == party) record[2] ^= 1;
    if (consistent && s->rank() == (party + 2) % 3) record[3] ^= 1;
  }
};
namespace {
RuntimeConfig Config() {
  RuntimeConfig c(SPDZWISE, FM64);
  c.spdzwise_config.domain = SpdzWiseConfig::RING;
  return c;
}
int64_t Floor(int64_t x, size_t m) {
  if (m == 63) return x < 0 ? -1 : 0;
  const int64_t d = int64_t(1) << m;
  return x / d - (x % d < 0);
}
NdArrayRef Plain(const std::vector<int64_t>& values) {
  NdArrayRef out(makeType<RingTy>(FM64), {int64_t(values.size())});
  for (size_t i = 0; i < values.size(); ++i)
    out.data<uint64_t>()[i] = values[i];
  return out;
}

std::vector<int64_t> RunTruncation(const std::vector<int64_t>& values, size_t m,
                                   bool strided = false, bool naive = false,
                                   SignType sign = SignType::Unknown) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto shares = io.toShares(Plain(values), VIS_SECRET);
  struct Result {
    NdArrayRef internal, external;
    RepShare key;
  };
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    const auto* selector =
        dynamic_cast<TruncASelector*>(ctx.prot()->getKernel("trunc_a"));
    EXPECT_NE(selector, nullptr);
    if (selector) {
      EXPECT_EQ(selector->kind(), Kernel::Kind::Dynamic);
      EXPECT_FALSE(selector->hasMsbError());
      EXPECT_EQ(selector->lsbRounding(), TruncLsbRounding::Floor);
    }
    auto* s = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Value input(shares[link->Rank()], DT_I64);
    s->begin(ExecutableProto("trunc-test", {"x"}, {"y"}, "test"), {input});
    auto x = import_s(&ctx, input);
    if (strided && x.numel()) {
      auto wide = NdArrayRef(x.storage_type(), {x.numel(), 2});
      for (int64_t i = 0; i < x.numel(); ++i)
        wide.data<AuthShare>()[2 * i] = x.data().data<AuthShare>()[i];
      x = Value(wide.slice({0, 0}, {x.numel(), 1}, {1, 1}).reshape(x.shape()),
                DT_I64);
      EXPECT_FALSE(x.data().isCompact());
    }
    EXPECT_EQ(TruncationTestAccess::pending(s), 0);
    const auto checks = s->arithmeticChecks();
    const auto bytes = link->GetStats()->sent_bytes.load();
    auto y = naive ? trunc_a(&ctx, x, m, sign)
                   : Value(s->truncPrRandomBit(x.data(), m, sign), x.dtype());
    EXPECT_EQ(y.storage_type(), x.storage_type());
    EXPECT_EQ(y.dtype(), x.dtype());
    EXPECT_EQ(y.shape(), x.shape());
    const bool opened = !values.empty() && m != 0;
    EXPECT_EQ(s->arithmeticChecks(), checks + (opened ? 1 : 0));
    EXPECT_EQ(TruncationTestAccess::pending(s), 0);
    if (!values.empty() && m) {
      if (!naive) {
        EXPECT_GT(s->truncMaterialMs(), 0);
      }
      if (naive) {
        EXPECT_EQ(s->booleanChecks(), 1);
        EXPECT_EQ(s->pendingBooleanBits(), 0);
      }
      EXPECT_GT(link->GetStats()->sent_bytes.load(), bytes);
    } else {
      EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
    }
    const auto key = TruncationTestAccess::key(s);
    // A subsequent authenticated multiplication must accept the output.
    auto one = Plain(std::vector<int64_t>(values.size(), 1))
                   .as(makeType<Pub2kTy>(FM64));
    auto z = s->multiply(y.data(), s->publicValue(one));
    s->prepareOutputs(&ctx, {"y"}, {Value(z, DT_I64)});
    guard.commit();
    // Zero-shift direct kernels may return the original non-contiguous view.
    return Result{y.data().clone(), s->exportedValue("y").data(), key};
  });
  Word alpha = 0;
  std::vector<NdArrayRef> outputs;
  for (const auto& r : results) {
    alpha += r.key[0];
    outputs.push_back(r.external);
  }
  const auto plain = io.fromShares(outputs);
  std::vector<int64_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    Word value = 0, mac = 0;
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& s = results[rank].internal.data<AuthShare>()[i];
      const auto& next = results[(rank + 1) % 3].internal.data<AuthShare>()[i];
      EXPECT_EQ(s[1], next[0]);
      EXPECT_EQ(s[3], next[2]);
      value += s[0];
      mac += s[2];
    }
    EXPECT_EQ(mac, alpha * value);
    out[i] = static_cast<int64_t>(plain.data<uint64_t>()[i]);
    const auto floor = Floor(values[i], m);
    if (naive) {
      EXPECT_EQ(out[i], floor);
    } else {
      EXPECT_TRUE(out[i] == floor || (m && out[i] == floor + 1));
    }
    if ((uint64_t(values[i]) & ((uint64_t(1) << m) - 1)) == 0) {
      EXPECT_EQ(out[i], floor);
    }
  }
  return out;
}

TEST(SpdzWiseTruncation, NaiveFullWidthExactFloorAndStrides) {
  std::vector<int64_t> values{
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min() + 1,
      std::numeric_limits<int64_t>::max(), 0, 1, -1, 7, -7,
      65535, -65535, 65536, -65536, 65537, -65537};
  std::mt19937_64 rng(0x5452554e43);
  // Two full bit-sliced blocks and one tail lane.
  while (values.size() < 129) values.push_back(static_cast<int64_t>(rng()));
  for (size_t m = 0; m <= 63; ++m) RunTruncation(values, m, true, true);
  RunTruncation({}, 16, false, true);
  RunTruncation({-7}, 2, false, true);
  RunTruncation({0, 7, 65537}, 16, false, true, SignType::Positive);
  RunTruncation({-1, -7, -65537}, 16, false, true, SignType::Negative);
}

TEST(SpdzWiseTruncation, NaiveInvalidInputsAbort) {
  for (int fault = 0; fault < 4; ++fault) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* s = ctx.prot()->getState<SpdzWiseState>();
      s->begin(ExecutableProto("bad-naive", {}, {}, "test"), {});
      auto x = s->publicValue(Plain({7}).as(makeType<Pub2kTy>(FM64)));
      if (fault == 1) x = x.as(makeType<AShrTy>("other-session"));
      if (fault == 2) x = NdArrayRef(makeType<spdzwise::AShrTy>(s->session()), {1});
      EXPECT_ANY_THROW(s->truncNaive(x, fault == 0 ? 64 : 16,
          fault == 3 ? static_cast<SignType>(999) : SignType::Unknown));
      EXPECT_TRUE(s->failed());
      EXPECT_ANY_THROW(s->commit());
    });
  }
}

TEST(SpdzWiseTruncation, ExhaustiveSignedSmallRingFormula) {
  for (unsigned ell = 3; ell <= 7; ++ell) {
    const unsigned k = ell - 1;
    for (unsigned m = 1; m < k; ++m) {
      for (int x = -(1 << (k - 1)); x < (1 << (k - 1)); ++x) {
        for (unsigned r = 0; r < (1U << ell); ++r) {
          const unsigned c = (x + (1 << (k - 1)) + r) & ((1U << ell) - 1);
          const int upper = (r >> m) & ((1U << (k - m)) - 1);
          const int b = (r >> k) ^ (c >> k);
          const int y = int((c & ((1U << k) - 1)) >> m) - upper +
                        (b << (k - m)) - (1 << (k - m - 1));
          ASSERT_EQ(y, Floor(x + int(r & ((1U << m) - 1)), m));
        }
      }
    }
  }
}

TEST(SpdzWiseTruncation, SignedEdgesStridesAndShiftBounds) {
  const int64_t bound = int64_t(1) << 62;
  const std::vector<int64_t> values = {
      0,     1,      -1,    7,      -7,     65535,      -65535,
      65536, -65536, 65537, -65537, -bound, -bound + 1, bound - 1};
  for (size_t m : {0U, 1U, 2U, 16U, 62U}) RunTruncation(values, m, true);
  RunTruncation({}, 16);
}

TEST(SpdzWiseTruncation, VerifiesBothDomainsBeforeEveryMaskedOpening) {
  for (bool naive : {false, true}) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      state->begin(ExecutableProto("trunc-checkpoint", {}, {}, "test"), {});
      const auto p = Plain({7}).as(makeType<Pub2kTy>(FM64));
      auto x = state->publicValue(p);
      const auto b = state->publicBoolean(p);
      size_t openings = 0;
      const auto observe = naive ? BooleanTestAccess::observeOpening
                                : TruncationTestAccess::observeOpening;
      observe(state, [&] {
        ++openings;
        EXPECT_EQ(state->arithmeticChecks(), openings);
        EXPECT_EQ(state->booleanChecks(), openings);
        EXPECT_EQ(TruncationTestAccess::pending(state), 0);
        EXPECT_EQ(state->pendingBooleanBits(), 0);
      });
      for (size_t batch = 0; batch < 2; ++batch) {
        x = state->multiply(x, x);
        state->booleanBinary(b, b, true, false);
        EXPECT_EQ(state->pendingBooleanBits(), 64);
        x = naive ? state->truncNaive(x, 1, SignType::Unknown)
                  : state->truncPrRandomBit(x, 1, SignType::Unknown);
        EXPECT_EQ(openings, batch + 1);
      }
      state->prepareOutputs(&ctx, {}, {});
      state->commit();
    });
  }
}

TEST(SpdzWiseTruncation, PendingFaultsRejectBeforeMaskedOpening) {
  for (bool naive : {false, true}) {
    for (size_t party = 0; party < 3; ++party) {
      for (int fault = 0; fault < 3; ++fault) {
        utils::simulate(3, [&](const auto& link) {
          SPUContext ctx(Config(), link);
          registerArithmeticPrototype(&ctx, link);
          auto* state = ctx.prot()->getState<SpdzWiseState>();
          state->begin(ExecutableProto("trunc-checkpoint-fault", {}, {}, "test"),
                       {});
          const auto p = Plain({7}).as(makeType<Pub2kTy>(FM64));
          auto x = state->publicValue(p);
          x = state->multiply(x, x);
          if (fault < 2) {
            TruncationTestAccess::corruptMac(state, party, fault == 1);
          } else {
            auto b = state->publicBoolean(p);
            if (state->rank() == party) b.data<spdzwise::RepShare>()[0][0] ^= 1;
            state->booleanBinary(b, b, true, false);
          }
          bool opened = false;
          const auto observe = naive ? BooleanTestAccess::observeOpening
                                    : TruncationTestAccess::observeOpening;
          observe(state, [&] { opened = true; });
          try {
            if (naive) {
              state->truncNaive(x, 1, SignType::Unknown);
            } else {
              state->truncPrRandomBit(x, 1, SignType::Unknown);
            }
            FAIL() << "accepted corrupt pending work";
          } catch (const std::exception& error) {
            const std::string reason = error.what();
            EXPECT_NE(
                reason.find(fault < 2 ? "SPDZWISE ring check failed"
                                      : "Boolean replicated copies disagree"),
                std::string::npos)
                << reason;
          }
          EXPECT_FALSE(opened);
          EXPECT_TRUE(state->failed());
          EXPECT_ANY_THROW(state->exportedValue("out"));
        });
      }
    }
  }
}

TEST(SpdzWiseTruncation, RotatingOwnersAndPartialTile) {
  // Three full tiles rotate through all owners; the final partial tile reuses
  // the same scratch allocations at a different length.
  std::vector<int64_t> values(3 * 4096 + 1);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = i % 2 ? -int64_t(i * 65537) : int64_t(i * 65537);
  RunTruncation(values, 16, true);
}

TEST(SpdzWiseTruncation, SignedRoundingFrequency) {
  // 8192 independent masks: a generous >8-sigma tolerance, avoiding flaky
  // tests while detecting wrong sign handling and deterministic rounding.
  std::vector<int64_t> values(8192);
  for (size_t i = 0; i < values.size(); ++i) values[i] = i % 2 ? -7 : 7;
  const auto output = RunTruncation(values, 2);
  size_t negative_up = 0, positive_up = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % 2)
      negative_up += output[i] == -1;
    else
      positive_up += output[i] == 2;
  }
  EXPECT_NEAR(double(negative_up) / 4096, 0.25, 0.06);
  EXPECT_NEAR(double(positive_up) / 4096, 0.75, 0.06);
}

TEST(SpdzWiseTruncation, BadParametersAndBindingsPoisonSession) {
  for (int fault = 0; fault < 3; ++fault) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* s = ctx.prot()->getState<SpdzWiseState>();
      s->begin(ExecutableProto("bad-trunc", {}, {}, "test"), {});
      auto x = s->publicValue(Plain({7}).as(makeType<Pub2kTy>(FM64)));
      if (fault == 1) x = x.as(makeType<AShrTy>("other-session"));
      if (fault == 2)
        x = NdArrayRef(makeType<spdzwise::AShrTy>(s->session()), {1});
      EXPECT_ANY_THROW(s->truncPrRandomBit(x, fault == 0 ? 63 : 16, SignType::Unknown));
      EXPECT_TRUE(s->failed());
      EXPECT_ANY_THROW(s->commit());
    });
  }
  utils::simulate(3, [&](const auto& link) {
    RuntimeConfig config(SPDZWISE, FM64);
    SPUContext ctx(config, link);
    registerArithmeticPrototype(&ctx, link);
    EXPECT_FALSE(ctx.prot()->hasKernel("trunc_a"));
  });
}

TEST(SpdzWiseTruncation, BitnessRejectsNonBitsIncludingHighValuation) {
  for (uint64_t bad : {uint64_t(2), (uint64_t(1) << 63) + 1}) {
    utils::simulate(3, [&](const auto& link) {
      size_t message = 0;
      ArithmeticState s(link, "bit-test", [&](std::string_view op) {
        return std::to_string(message++) + "/" + std::string(op);
      });
      auto b = Plain({static_cast<int64_t>(bad)}).as(makeType<Pub2kTy>(FM64));
      EXPECT_ANY_THROW(TruncationTestAccess::checkBits(s, s.publicValue(b)));
    });
  }
}

TEST(SpdzWiseTruncation, NativeAby3RegisteredTruncAPr2MixedSigns) {
  auto io = aby3::makeAby3Io(FM64, 3);
  const int64_t bound = int64_t(1) << 62;
  const std::vector<int64_t> values{0,     1,      -1,         7,        -7,
                                    65536, -65536, -bound + 1, bound - 1};
  const auto shares = io->toShares(Plain(values), VIS_SECRET, -1);
  const auto output = utils::simulate(3, [&](const auto& link) {
    auto ctx = makeAby3Protocol(RuntimeConfig(ABY3, FM64), link);
    EXPECT_NE(dynamic_cast<aby3::TruncAPr2*>(ctx->prot()->getKernel("trunc_a")),
              nullptr);
    return trunc_a(ctx.get(), Value(shares[link->Rank()], DT_I64), 16,
                   SignType::Unknown)
        .data();
  });
  const auto plain = io->fromShares(output);
  for (size_t i = 0; i < values.size(); ++i) {
    const auto y = static_cast<int64_t>(plain.data<uint64_t>()[i]);
    EXPECT_LE(std::abs(y - Floor(values[i], 16)), 1);
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise::ring
