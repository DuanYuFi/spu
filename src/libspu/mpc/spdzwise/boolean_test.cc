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

#include "libspu/mpc/spdzwise/boolean.h"

#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {

class BooleanTestAccess {
 public:
  static void shuffleOracle() {
    for (size_t count : {0U, 1U, 63U, 64U, 65U, 4099U, 131075U, 1048576U}) {
      const uint128_t seed = 0x123456789abcdefULL;
      yacl::crypto::Prg<uint64_t, 4096> prg(seed);
      std::vector<size_t> expected(count);
      std::iota(expected.begin(), expected.end(), 0);
      for (size_t i = 0; i < count; ++i) {
        const uint64_t range = count - i, threshold = -range % range;
        uint64_t draw;
        do {
          draw = prg();
        } while (draw < threshold);
        std::swap(expected[i], expected[i + draw % range]);
      }
      EXPECT_EQ(BooleanState::shuffledIndices(count, seed), expected);
    }
  }
  static void generationOracle(SpdzWiseState* state) {
    auto& checker = *state->boolean_;
    for (size_t words :
         {0U, 1U, 3U, 65535U, 65536U, 65537U, 131071U, 131072U, 131075U}) {
      // Independent serial reference, including the PRSS continuation. This
      // catches a reordered/reused stream even if multiplication still works.
      auto random = checker.random_prg_;
      auto masks = checker.mask_prg_;
      std::vector<uint64_t> a0(words), a1(words), b0(words), b1(words),
          r0(words), r1(words);
      // Serial reference uses the protocol's public 65536-word tile layout.
      // Compare all fields and the subsequent counters, including odd tails.
      for (size_t first = 0; first < words; first += 65536) {
        const auto size = std::min(size_t{65536}, words - first);
        random.fillPrssPair(a0.data() + first, a1.data() + first, size,
                            PrgState::GenPrssCtrl::Both);
        random.fillPrssPair(b0.data() + first, b1.data() + first, size,
                            PrgState::GenPrssCtrl::Both);
        masks.fillPrssPair(r0.data() + first, r1.data() + first, size,
                           PrgState::GenPrssCtrl::Both);
      }
      auto pool = checker.generate(words);
      ASSERT_EQ(pool.size(), words);
      for (size_t i = 0; i < words; ++i) {
        ASSERT_EQ(pool[i][0], (RepShare{a0[i], a1[i]}));
        ASSERT_EQ(pool[i][1], (RepShare{b0[i], b1[i]}));
        ASSERT_EQ(pool[i][2][0], r0[i] ^ r1[i] ^ (a0[i] & b0[i]) ^
                                     (a0[i] & b1[i]) ^ (a1[i] & b0[i]));
      }
      std::array<uint64_t, 2> expected0, expected1, actual0, actual1;
      for (bool mask : {false, true}) {
        auto& reference = mask ? masks : random;
        auto& actual = mask ? checker.mask_prg_ : checker.random_prg_;
        reference.fillPrssPair(expected0.data(), expected1.data(), 2,
                               PrgState::GenPrssCtrl::Both);
        actual.fillPrssPair(actual0.data(), actual1.data(), 2,
                            PrgState::GenPrssCtrl::Both);
        ASSERT_EQ(expected0, actual0);
        ASSERT_EQ(expected1, actual1);
      }
    }
  }
  static std::vector<uint64_t> open(BooleanState& checker,
                                    const std::vector<RepShare>& shares) {
    return checker.open(shares);
  }
  static void corrupt(SpdzWiseState* state, size_t party, bool consistent,
                      size_t position = 0) {
    auto& checker = *state->boolean_;
    auto word = position / 64;
    auto* selected = &checker.overflow_;
    if (word < checker.pending_.size()) {
      selected = &checker.pending_.at(word);
    } else {
      word -= checker.pending_.size();
      for (auto& batch : checker.pending_batches_) {
        if (word < batch.size()) {
          selected = &batch.at(word);
          break;
        }
        word -= batch.size();
      }
    }
    auto& triple = *selected;
    if (state->rank() == party) {
      triple[2][0] ^= uint64_t{1} << (position % 64);
    }
    if (consistent && state->rank() == (party + 2) % 3) {
      triple[2][1] ^= uint64_t{1} << (position % 64);
    }
  }
  static void randomFault(SpdzWiseState* state, size_t party, bool all) {
    auto& checker = *state->boolean_;
    const auto words = checker.bits_ / 64 + (checker.bits_ % 64 != 0);
    auto pool = checker.generate((AbflBucketSize(words) - 1) * words + 3);
    for (size_t i = 0; i < pool.size(); ++i) {
      const uint64_t error = all ? ~uint64_t{0} : (i == 0 ? 1 : 0);
      if (state->rank() == party) {
        pool[i][2][0] ^= error;
      }
      if (state->rank() == (party + 2) % 3) {
        pool[i][2][1] ^= error;
      }
    }
    try {
      checker.checkPrepared(std::move(pool));
    } catch (...) {
      state->abort();
      throw;
    }
  }
  static void packingOracle(SpdzWiseState* state) {
    auto& checker = *state->boolean_;
    using Triple = BooleanState::Triple;
    std::mt19937_64 rng(87342);
    for (size_t width = 1; width <= 64; ++width) {
      for (size_t used = 0; used < 64; ++used) {
        for (size_t count : {0U, 1U, 63U, 64U, 129U}) {
          checker.pending_.clear();
          checker.overflow_bits_ = used;
          checker.bits_ = used;
          for (auto& share : checker.overflow_) {
            for (auto& word : share) {
              word = used ? rng() & BooleanMask(used) : 0;
            }
          }
          const auto initial = checker.overflow_;
          std::vector<RepShare> a(count), b(count), c(count);
          for (auto* arr : {&a, &b, &c}) {
            for (auto& share : *arr) {
              for (auto& word : share) word = rng() & BooleanMask(width);
            }
          }
          const auto total = used + count * width;
          std::vector<Triple> expected(total / 64 + (total % 64 != 0));
          size_t bit = 0;
          auto add = [&](const Triple& t, size_t nbits) {
            for (size_t k = 0; k < nbits; ++k, ++bit) {
              for (size_t j = 0; j < 3; ++j) {
                for (size_t s = 0; s < 2; ++s) {
                  expected[bit / 64][j][s] |= ((t[j][s] >> k) & 1)
                                              << (bit % 64);
                }
              }
            }
          };
          if (width != 64) add(initial, used);
          for (size_t i = 0; i < count; ++i) add({a[i], b[i], c[i]}, width);
          if (width == 64) add(initial, used);
          checker.append(a, b, c, width);
          ASSERT_EQ(checker.bits_, total);
          ASSERT_EQ(checker.pending_.size(), total / 64);
          ASSERT_EQ(checker.overflow_bits_, total % 64);
          for (size_t i = 0; i < checker.pending_.size(); ++i) {
            ASSERT_EQ(checker.pending_[i], expected[i]);
          }
          ASSERT_EQ(checker.overflow_, total % 64 ? expected.back() : Triple{});
        }
      }
    }
    checker.pending_.clear();
    checker.overflow_ = {};
    checker.overflow_bits_ = 0;
    checker.bits_ = 0;
  }
  static void phase(SpdzWiseState* state,
                    std::function<void(std::string_view)> fn) {
    auto old = state->boolean_->tag_;
    state->boolean_->tag_ = [old, fn](std::string_view name) {
      fn(name);
      return old(name);
    };
  }
};

namespace {
RuntimeConfig Config() {
  RuntimeConfig cfg;
  cfg.protocol = SPDZWISE;
  cfg.field = FM64;
  return cfg;
}
NdArrayRef Plain(int64_t count, size_t width, uint64_t salt = 0) {
  NdArrayRef out(makeType<RingTy>(FM64), {count});
  for (int64_t i = 0; i < count; ++i) {
    out.data<uint64_t>()[i] =
        (uint64_t(i) * 0x9e3779b97f4a7c15ULL + salt) & BooleanMask(width);
  }
  return out;
}
void Begin(SpdzWiseState* state, const std::vector<Value>& inputs = {}) {
  std::vector<std::string> names;
  for (size_t i = 0; i < inputs.size(); ++i) {
    names.push_back(std::to_string(i));
  }
  state->begin(ExecutableProto("boolean", names, {"out"}, "test"), inputs);
}

void ExpectFailure(const std::function<void()>& operation,
                   std::string_view reason) {
  try {
    operation();
    FAIL() << "expected protocol rejection";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find(reason), std::string::npos)
        << error.what();
  }
}

TEST(Abfl, BucketBoundaries) {
  EXPECT_EQ(AbflBucketSize(0), 0);
  for (size_t n : {1, 1023}) {
    EXPECT_ANY_THROW(AbflBucketSize(n));
  }
  EXPECT_EQ(AbflBucketSize(1024), 5);
  EXPECT_EQ(AbflBucketSize(10367), 5);
  EXPECT_EQ(AbflBucketSize(10368), 4);
  EXPECT_EQ(AbflBucketSize(1048575), 4);
  EXPECT_EQ(AbflBucketSize(1048576), 3);
}

TEST(Abfl, BatchedShuffleMatchesSerialFisherYates) {
  BooleanTestAccess::shuffleOracle();
}

TEST(Abfl, ParallelGenerationPreservesSerialRandomStreams) {
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Begin(state);
    BooleanTestAccess::generationOracle(state);
  });
}

TEST(Abfl, DirectBufferOpeningChecksLastReplicaAndLengths) {
  utils::simulate(3, [&](const auto& link) {
    size_t sequence = 0;
    BooleanState checker(link, [&](std::string_view name) {
      return fmt::format("direct-open/{}/{}", sequence++, name);
    });
    auto component = [](size_t rank, size_t i) {
      return uint64_t(i + 1) * 0x9e3779b97f4a7c15ULL ^
             (uint64_t{0xfedcba9876543210} >> (rank * 7));
    };
    for (size_t count : {0U, 1U, 7U, 65537U}) {
      std::vector<RepShare> shares;
      for (size_t i = 0; i < count; ++i) {
        shares.push_back(
            {component(link->Rank(), i), component(link->NextRank(), i)});
      }
      const auto opened = BooleanTestAccess::open(checker, shares);
      ASSERT_EQ(opened.size(), count);
      for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(opened[i],
                  component(0, i) ^ component(1, i) ^ component(2, i));
      }
      if (!count) continue;
      // All parties see the same transcript, but its last replica is wrong.
      if (link->Rank() == 1) shares.back()[1] ^= uint64_t{1} << 63;
      ExpectFailure([&] { BooleanTestAccess::open(checker, shares); },
                    "Boolean opening copies disagree");
      // A consistently broadcast short message must fail before direct loads.
      if (link->Rank() == 1) shares.pop_back();
      ExpectFailure([&] { BooleanTestAccess::open(checker, shares); },
                    "Boolean wire length mismatch");
    }
  });
}

TEST(Abfl, PackingAllWidthsAndOverflowOccupancies) {
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Begin(state);
    BooleanTestAccess::packingOracle(state);
    state->prepareOutputs(&ctx, {}, {});
    state->commit();
  });
}

TEST(Abfl, AllBucketsAndWidthsEndToEnd) {
  SpdzWiseIo io(FM64, 3);
  for (const auto& [count, width] :
       std::vector<std::pair<int64_t, size_t>>{{65536, 1},
                                               {9363, 7},
                                               {20736, 32},
                                               {65535, 64},
                                               {65536, 64},
                                               {65537, 64},
                                               {599187, 7},
                                               {1048576, 64}}) {
    const auto a = Plain(count, width, 17), b = Plain(count, width, 91);
    const auto xs = io.toBooleanShares(a, width),
               ys = io.toBooleanShares(b, width);
    const auto outputs = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      Value x(xs[link->Rank()], DT_U64), y(ys[link->Rank()], DT_U64);
      Begin(state, {x, y});
      x = import_s(&ctx, x);
      y = import_s(&ctx, y);
      EXPECT_ANY_THROW(x.toProto(1024));
      const auto stats = link->GetStats();
      const auto actions = stats->sent_actions.load(),
                 bytes = stats->sent_bytes.load();
      auto z = dynDispatch(&ctx, "and_bb", x, y);
      EXPECT_EQ(stats->sent_actions.load() - actions, 1);
      EXPECT_EQ(stats->sent_bytes.load() - bytes, 8 * count);
      EXPECT_EQ(state->pendingBooleanBits(), count * width);
      EXPECT_EQ(state->booleanChecks(), 0);
      EXPECT_ANY_THROW(state->exportedValue("out"));
      EXPECT_ANY_THROW(state->exportValue(
          z.data(),
          makeType<BOutputTy>(link->Rank(), width, std::string(64, 'a'))));
      z.setDtype(DT_U64);
      state->prepareOutputs(&ctx, {"out"}, {z});
      EXPECT_EQ(state->booleanChecks(), 1);
      EXPECT_EQ(state->pendingBooleanBits(), 0);
      const auto after = stats->sent_actions.load();
      state->finalize();
      EXPECT_EQ(stats->sent_actions.load(), after);
      guard.commit();
      return state->exportedValue("out").data();
    });
    const auto actual = io.fromShares(outputs);
    for (int64_t i = 0; i < count; ++i) {
      EXPECT_EQ(actual.data<uint64_t>()[i],
                a.data<uint64_t>()[i] & b.data<uint64_t>()[i]);
    }
  }
}

TEST(Abfl, FullWidthStridedAndEmptyFastPath) {
  SpdzWiseIo io(FM64, 3);
  for (int64_t rows : {0, 32}) {
    const auto input = Plain(rows * 64, 64, 139).reshape({rows, 64});
    const auto shares = io.toBooleanShares(input, 64);
    const auto outputs = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      Value x(shares[link->Rank()], DT_U64);
      Begin(state, {x});
      x = dynDispatch(&ctx, "transpose", import_s(&ctx, x), Axes{1, 0});
      const auto stats = link->GetStats();
      const auto before = stats->sent_actions.load();
      auto z = dynDispatch(&ctx, "and_bb", x, x);
      EXPECT_EQ(stats->sent_actions.load() - before, rows ? 1 : 0);
      if (rows) x.data().data<RepShare>()[0][0] ^= ~uint64_t{0};
      state->prepareOutputs(&ctx, {"out"}, {z});
      state->commit();
      return state->exportedValue("out").data();
    });
    const auto actual = io.fromShares(outputs);
    for (int64_t i = 0; i < rows * 64; ++i) {
      EXPECT_EQ(actual.data<uint64_t>()[i],
                input.data<uint64_t>()[(i % rows) * 64 + i / rows]);
    }
  }
}

TEST(Abfl, MixedWidthsViewsAndDeferredBatch) {
  SpdzWiseIo io(FM64, 3);
  auto a = Plain(75, 7, 21).reshape({3, 25});
  auto b = Plain(75, 32, 177).reshape({3, 25});
  const auto xs = io.toBooleanShares(a, 7), ys = io.toBooleanShares(b, 32);
  const auto arithmetic = io.toShares(Plain(1, 7), VIS_SECRET);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Value x(xs[link->Rank()], DT_U64), y(ys[link->Rank()], DT_U64);
    Begin(state, {x, y});
    x = dynDispatch(&ctx, "transpose", import_s(&ctx, x), Axes{1, 0});
    y = dynDispatch(&ctx, "transpose", import_s(&ctx, y), Axes{1, 0});
    auto z = dynDispatch(&ctx, "and_bb", x, y);
    EXPECT_EQ(state->pendingBooleanBits(), 525);
    // An intervening arithmetic import must not trigger Boolean sacrifice.
    auto imported = import_s(&ctx, Value(arithmetic[link->Rank()], DT_U64));
    (void)imported;
    auto p = Value(Plain(1, 7).as(makeType<Pub2kTy>(FM64)), DT_U64);
    auto aa = dynDispatch(&ctx, "p2a", p);
    auto product = dynDispatch(&ctx, "mul_aa", aa, aa);
    (void)product;
    state->check();
    EXPECT_ANY_THROW(state->exportValue(
        product.data(),
        makeType<OutputTy>(link->Rank(), std::string(64, 'a'))));
    EXPECT_EQ(state->pendingBooleanBits(), 525);
    z = dynDispatch(&ctx, "and_bb", z, x);
    EXPECT_EQ(state->pendingBooleanBits(), 1050);
    // A full-width batch must bypass the existing overflow without losing it.
    auto full =
        state->publicBoolean(Plain(1024, 64).as(makeType<Pub2kTy>(FM64)));
    state->booleanBinary(full, full, true, false);
    EXPECT_EQ(state->pendingBooleanBits(), 1050 + 1024 * 64);
    state->booleanBinary(x.data(), y.data(), true, false);
    EXPECT_EQ(state->pendingBooleanBits(), 1575 + 1024 * 64);
    auto q = Value(
        Plain(75, 64, 9).reshape({25, 3}).as(makeType<Pub2kTy>(FM64)), DT_U64);
    z = dynDispatch(&ctx, "xor_bb", z, dynDispatch(&ctx, "p2b", q));
    z = dynDispatch(&ctx, "xor_bp", z, q);
    z = dynDispatch(&ctx, "and_bp", z, q);
    z.setDtype(DT_U64);
    state->prepareOutputs(&ctx, {"out"}, {z});
    guard.commit();
    return state->exportedValue("out").data();
  });
  auto actual = io.fromShares(outputs);
  auto q = Plain(75, 64, 9);
  for (int64_t i = 0; i < 75; ++i) {
    const auto original = (i % 3) * 25 + i / 3;
    EXPECT_EQ(actual.data<uint64_t>()[i], a.data<uint64_t>()[original] &
                                              b.data<uint64_t>()[original] &
                                              q.data<uint64_t>()[i]);
  }
}

TEST(Abfl, EmptySmallAndNoOutputExecutions) {
  for (int64_t count : {0, 1, 1023, 1024, 65472, 65473, 65535, 65536}) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      Begin(state);
      auto p = Plain(count, 1).as(makeType<Pub2kTy>(FM64));
      auto b =
          state->publicBoolean(p).as(makeType<BShrTy>(state->session(), 1));
      state->booleanBinary(b, b, true, false);
      if (count > 0 && count <= 65472) {
        EXPECT_ANY_THROW(state->prepareOutputs(&ctx, {}, {}));
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(Begin(state));
      } else {
        state->prepareOutputs(&ctx, {}, {});
        EXPECT_EQ(state->booleanChecks(), count == 0 ? 0 : 1);
        state->commit();
      }
    });
  }
}

TEST(Abfl, SnapshotAndReuse) {
  SpdzWiseIo io(FM64, 3);
  const auto input = Plain(1024, 64, 11);
  const auto shares = io.toBooleanShares(input, 64);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value x(shares[link->Rank()], DT_U64);
    Begin(state, {x});
    x = import_s(&ctx, x);
    auto z = dynDispatch(&ctx, "and_bb", x, x);
    // A mutation after AND must not alter its saved inputs.
    x.data().data<RepShare>()[0][0] ^= 1;
    state->prepareOutputs(&ctx, {"out"}, {z});
    state->commit();
    const auto first = state->exportedValue("out");
    Begin(state, {z});
    z = import_s(&ctx, z);
    z = dynDispatch(&ctx, "and_bb", z, z);
    state->prepareOutputs(&ctx, {"out"}, {z});
    state->commit();
    EXPECT_EQ(state->booleanChecks(), 2);
    EXPECT_NE(first.storage_type(), state->exportedValue("out").storage_type());
    // Re-import a committed Boolean output through the external IO path.
    Begin(state, {first});
    auto again = import_s(&ctx, first);
    state->prepareOutputs(&ctx, {"out"}, {again});
    state->commit();
    auto wrong = z.data().as(makeType<BShrTy>("another-session", 64));
    Begin(state);
    EXPECT_ANY_THROW(state->validateBoolean(wrong));
    state->prepareOutputs(&ctx, {}, {});
    state->commit();
    return first.data();
  });
  const auto actual = io.fromShares(outputs);
  for (int64_t i = 0; i < 1024; ++i) {
    EXPECT_EQ(actual.data<uint64_t>()[i], input.data<uint64_t>()[i]);
  }
}

TEST(Abfl, OnlineAndRandomFaultsAtEveryParty) {
  for (size_t party = 0; party < 3; ++party) {
    for (int fault = 0; fault < 4; ++fault) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto x = state->publicBoolean(
            Plain(1024, 64, 73).as(makeType<Pub2kTy>(FM64)));
        state->booleanBinary(x, x, true, false);
        if (fault < 2) {
          BooleanTestAccess::corrupt(state, party, fault == 1);
          ExpectFailure([&] { state->finalize(); },
                        fault == 0 ? "Boolean replicated copies disagree"
                                   : "ABFL sacrifice failed");
        } else {
          ExpectFailure(
              [&] { BooleanTestAccess::randomFault(state, party, fault == 3); },
              fault == 3 ? "ABFL cut-and-choose failed" : "ABFL");
        }
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->exportedValue("out"));
        EXPECT_ANY_THROW(Begin(state));
      });
    }
  }
}

TEST(Abfl, DetectsFaultInLastPartialWord) {
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Begin(state);
    auto x = state->publicBoolean(Plain(9363, 7).as(makeType<Pub2kTy>(FM64)))
                 .as(makeType<BShrTy>(state->session(), 7));
    state->booleanBinary(x, x, true, false);
    EXPECT_EQ(state->pendingBooleanBits(), 65541);
    BooleanTestAccess::corrupt(state, 1, true, 65540);
    EXPECT_ANY_THROW(state->finalize());
    EXPECT_TRUE(state->failed());
  });
}

TEST(Abfl, DetectsCorruptionInZeroExtendedPadding) {
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Begin(state);
    auto x = state->publicBoolean(Plain(9363, 7).as(makeType<Pub2kTy>(FM64)))
                 .as(makeType<BShrTy>(state->session(), 7));
    state->booleanBinary(x, x, true, false);
    // Last word has five valid bits. Even its padding is checked, not masked.
    BooleanTestAccess::corrupt(state, 2, true, 65599);
    ExpectFailure([&] { state->finalize(); }, "ABFL sacrifice failed");
    EXPECT_TRUE(state->failed());
  });
}

TEST(Abfl, DetectsFaultInLastSacrificeTile) {
  for (size_t party = 0; party < 3; ++party) {
    for (bool consistent : {false, true}) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto x =
            state->publicBoolean(Plain(65537, 64).as(makeType<Pub2kTy>(FM64)));
        state->booleanBinary(x, x, true, false);
        BooleanTestAccess::corrupt(state, party, consistent, 65537 * 64 - 1);
        ExpectFailure([&] { state->finalize(); },
                      consistent ? "ABFL sacrifice failed"
                                 : "Boolean replicated copies disagree");
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(Begin(state));
      });
    }
  }
}

// Test-only channel decorator: mutate actual wire messages without adding
// fault-injection branches to the protocol implementation.
class FaultChannel : public yacl::link::transport::IChannel {
  std::shared_ptr<yacl::link::transport::IChannel> inner_;
  std::shared_ptr<bool> active_;
  yacl::Buffer change(yacl::Buffer value) {
    if (*active_ && value.size()) {
      value.data<char>()[0] ^= 1;
    }
    return value;
  }

 public:
  FaultChannel(std::shared_ptr<yacl::link::transport::IChannel> inner,
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
TEST(Abfl, RejectsWireTampering) {
  for (size_t party = 0; party < 3; ++party) {
    for (const auto* phase : {"boolean-and", "boolean-copies", "boolean-commit",
                              "boolean-reveal", "boolean-open"}) {
      utils::simulate(3, [&](const auto& base) {
        auto active = std::make_shared<bool>(false);
        yacl::link::ContextDesc desc;
        desc.id = base->Id();
        std::vector<std::shared_ptr<yacl::link::transport::IChannel>> channels(
            3);
        for (size_t i = 0; i < 3; ++i) {
          desc.parties.push_back({std::to_string(i), "memory"});
          if (i != base->Rank()) {
            channels[i] =
                std::make_shared<FaultChannel>(base->GetChannel(i), active);
          }
        }
        auto link = std::make_shared<yacl::link::Context>(desc, base->Rank(),
                                                          channels, nullptr);
        link->SetRecvTimeout(5000);
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto x =
            state->publicBoolean(Plain(1024, 64).as(makeType<Pub2kTy>(FM64)));
        BooleanTestAccess::phase(
            state,
            [active, party, phase, rank = link->Rank()](std::string_view name) {
              *active = rank == party && name == phase;
            });
        state->booleanBinary(x, x, true, false);
        ExpectFailure([&] { state->finalize(); },
                      (std::string_view(phase) == "boolean-and" ||
                       std::string_view(phase) == "boolean-copies")
                          ? "Boolean replicated copies disagree"
                          : "Boolean transcript disagreement");
        EXPECT_TRUE(state->failed());
      });
    }
  }
}

TEST(AbflIo, PackedAndBitInputsAndMetadata) {
  SpdzWiseIo io(FM64, 3);
  EXPECT_TRUE(io.hasBitSecretSupport());
  EXPECT_EQ(io.getBitSecretShareSize(9), 9 * sizeof(RepShare));
  const std::vector<int32_t> data = {-3, -2, -1, 0, 1, 2, 3};
  auto bits = io.makeBitSecret(PtBufferView(data));
  auto out = io.fromShares(bits);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(out.data<uint64_t>()[i], uint64_t(data[i]) & 1);
  }
  const uint8_t packed = 0xa5;
  out = io.fromShares(
      io.makeBitSecret(PtBufferView(&packed, PT_I1, {8}, {1}, true)));
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(out.data<uint64_t>()[i], (packed >> i) & 1);
  }
  const auto input = Plain(3, 7);
  auto shares = io.toBooleanShares(input, 7);
  EXPECT_ANY_THROW(io.toBooleanShares(input, 0));
  EXPECT_ANY_THROW(io.toBooleanShares(input, 65));
  EXPECT_ANY_THROW(io.toBooleanShares(Plain(3, 64, 128), 7));
  auto wrong = shares;
  wrong[1] = shares[1].as(makeType<BInputTy>(1, 8));
  EXPECT_ANY_THROW(io.fromShares(wrong));
  wrong = shares;
  std::swap(wrong[0], wrong[1]);
  EXPECT_ANY_THROW(io.fromShares(wrong));
  for (size_t r = 0; r < 3; ++r) {
    shares[r] = shares[r].as(
        makeType<BOutputTy>(r, 7, std::string(64, r == 1 ? 'b' : 'a')));
  }
  EXPECT_ANY_THROW(io.fromShares(shares));
  BInputTy parsed;
  parsed.fromString("2:7");
  const BInputTy expected(2, 7);
  EXPECT_TRUE(parsed.equals(&expected));
  EXPECT_ANY_THROW(parsed.fromString("2:7junk"));
  BOutputTy output;
  output.fromString("2:7:" + std::string(64, 'a'));
  EXPECT_EQ(output.nbits(), 7);
  BShrTy internal("test", 7);
  EXPECT_FALSE(internal.isSerializable());
  EXPECT_ANY_THROW(internal.fromString("test:7"));
}
}  // namespace
}  // namespace spu::mpc::spdzwise
