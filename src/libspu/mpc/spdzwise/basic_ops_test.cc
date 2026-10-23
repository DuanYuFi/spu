// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

#include "gtest/gtest.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {

class BasicOpsTestAccess {
 public:
  static size_t pending(SpdzWiseState* s) { return s->ring_->pending_.size(); }
  static void deterministicStreams(SpdzWiseState* s) {
    // Default PrgState has zero PRSS seeds. TEST ONLY: both replicas agree,
    // and deterministic full-width draws make truncation regressions visible.
    *s->ring_->mask_prg_ = PrgState();
    s->boolean_->random_prg_ = PrgState();
  }
  static std::vector<ring::RepShare> arithmeticOracle(SpdzWiseState* s,
                                                      size_t count,
                                                      size_t skip = 0) {
    auto prg = *s->ring_->mask_prg_;
    if (skip) {
      std::vector<ring::Word> a(skip), b(skip);
      // RandA consumes fresh values followed by MAC multiplication masks.
      for (int i = 0; i < 2; ++i)
        prg.fillPrssPair(a.data(), b.data(), skip, PrgState::GenPrssCtrl::Both);
    }
    std::vector<ring::Word> a(count), b(count);
    if (count)
      prg.fillPrssPair(a.data(), b.data(), count, PrgState::GenPrssCtrl::Both);
    std::vector<ring::RepShare> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = {a[i], b[i]};
    return out;
  }
  static std::vector<RepShare> booleanOracle(SpdzWiseState* s, size_t count,
                                             size_t skip = 0) {
    auto prg = s->boolean_->random_prg_;
    if (skip) {
      std::vector<uint64_t> a(skip), b(skip);
      prg.fillPrssPair(a.data(), b.data(), skip, PrgState::GenPrssCtrl::Both);
    }
    std::vector<uint64_t> a(count), b(count);
    if (count)
      prg.fillPrssPair(a.data(), b.data(), count, PrgState::GenPrssCtrl::Both);
    std::vector<RepShare> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = {a[i] & 1, b[i] & 1};
    return out;
  }
  static void frozen(SpdzWiseState* s, const Value& value) {
    if (!value.numel()) return;
    auto data = value.data();
    auto& mutable_data = data.data<ring::AuthShare>()[0];
    const auto saved = mutable_data;
    const auto index = s->ring_->pending_.size() - value.numel();
    const auto snapshot = s->ring_->pending_[index];
    mutable_data[0] ^= 1;
    mutable_data[2] ^= 1;
    EXPECT_EQ(s->ring_->pending_[index], snapshot);
    mutable_data = saved;
  }
  static void corrupt(SpdzWiseState* s, size_t party, bool consistent) {
    auto& record = s->ring_->pending_.front();
    if (s->rank() == party) record[2] ^= 1;
    if (consistent && s->rank() == (party + 2) % 3) record[3] ^= 1;
  }
};

namespace {
RuntimeConfig Config(SpdzWiseConfig::Domain domain = SpdzWiseConfig::RING) {
  RuntimeConfig c(SPDZWISE, FM64);
  c.spdzwise_config.domain = domain;
  return c;
}

void Begin(SpdzWiseState* state, const std::vector<Value>& inputs = {}) {
  std::vector<std::string> names;
  for (size_t i = 0; i < inputs.size(); ++i) names.push_back(std::to_string(i));
  state->begin(ExecutableProto("basic-ops", names, {"out"}, "test"), inputs);
}

std::vector<NdArrayRef> Export(SPUContext* ctx, std::vector<Value> values) {
  auto* state = ctx->prot()->getState<SpdzWiseState>();
  std::vector<std::string> names;
  for (size_t i = 0; i < values.size(); ++i) {
    names.push_back(std::to_string(i));
    values[i].setDtype(DT_I64);
  }
  state->prepareOutputs(ctx, names, values);
  state->commit();
  std::vector<NdArrayRef> out;
  for (const auto& name : names)
    out.push_back(state->exportedValue(name).data());
  return out;
}

NdArrayRef Plain(size_t width = 64) {
  NdArrayRef out(makeType<RingTy>(FM64), {4, 2});
  const std::array<uint64_t, 8> edge = {0,
                                        1,
                                        ~uint64_t{0},
                                        uint64_t{1} << 63,
                                        (uint64_t{1} << 63) - 1,
                                        uint64_t(-8),
                                        0x0123456789abcdefULL,
                                        0xfedcba9876543210ULL};
  for (size_t i = 0; i < edge.size(); ++i)
    out.data<uint64_t>()[i] = edge[i] & BooleanMask(width);
  return out.transpose({1, 0});
}

NdArrayRef Map(const NdArrayRef& x,
               const std::function<uint64_t(uint64_t, int64_t)>& f) {
  NdArrayRef out(makeType<RingTy>(FM64), x.shape());
  const NdArrayView<uint64_t> input(x);
  for (int64_t i = 0; i < x.numel(); ++i)
    out.data<uint64_t>()[i] = f(input[i], i);
  return out;
}

void ExpectOutputs(const std::vector<std::vector<NdArrayRef>>& results,
                   const std::vector<NdArrayRef>& expected) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  ASSERT_EQ(results.size(), 3);
  for (const auto& party : results) ASSERT_EQ(party.size(), expected.size());
  for (size_t j = 0; j < expected.size(); ++j) {
    SCOPED_TRACE(j);
    const auto actual =
        io.fromShares({results[0][j], results[1][j], results[2][j]});
    ASSERT_EQ(actual.shape(), expected[j].shape());
    const NdArrayView<uint64_t> a(actual), e(expected[j]);
    for (int64_t i = 0; i < actual.numel(); ++i) EXPECT_EQ(a[i], e[i]);
    for (size_t rank = 0; rank < 3; ++rank) {
      const NdArrayView<RepShare> x(results[rank][j]);
      const NdArrayView<RepShare> y(results[(rank + 1) % 3][j]);
      for (int64_t i = 0; i < actual.numel(); ++i) EXPECT_EQ(x[i][1], y[i][0]);
    }
  }
}

struct BooleanCase {
  std::string name;
  Sizes shifts;
  size_t start = 0, end = 0, width = 64;
};

std::vector<BooleanCase> Cases(size_t width) {
  std::vector<BooleanCase> cases;
  for (const auto* op : {"lshift_b", "rshift_b", "arshift_b"}) {
    if (std::string_view(op) == "arshift_b" && width != 64) continue;
    for (int64_t k : {0, 1, 31, 32, 63}) cases.push_back({op, {k}});
    cases.push_back({op, {0, 1, 31, 32, 63, 7, 8, 2}});
  }
  for (size_t target : {1, 8, 32, 63, 64})
    cases.push_back({"cast_type_b", {}, 0, 0, target});
  for (auto [start, end] : std::vector<std::pair<size_t, size_t>>{
           {0, 0}, {7, 7}, {64, 64}, {0, 64}, {3, 19}, {63, 64}})
    cases.push_back({"bitrev_b", {}, start, end});
  return cases;
}

size_t OutputWidth(size_t width, const BooleanCase& c) {
  if (c.name == "cast_type_b") return c.width;
  if (c.name == "bitrev_b") return std::max(width, c.end);
  if (c.name == "lshift_b")
    return std::min<size_t>(
        64, width + *std::max_element(c.shifts.begin(), c.shifts.end()));
  if (c.name == "rshift_b")
    return size_t(std::max<int64_t>(
        1,
        int64_t(width) - *std::min_element(c.shifts.begin(), c.shifts.end())));
  return 64;
}

uint64_t BooleanReference(uint64_t x, int64_t index, const BooleanCase& c) {
  // Independent per-bit oracle, including sign extension and reversal.
  uint64_t out = 0;
  for (int64_t dst = 0; dst < 64; ++dst) {
    int64_t src = dst;
    if (c.name == "cast_type_b" && size_t(dst) >= c.width) continue;
    if (c.name == "bitrev_b" && size_t(dst) >= c.start && size_t(dst) < c.end)
      src = c.start + c.end - 1 - dst;
    if (!c.shifts.empty()) {
      const auto shift = c.shifts.size() == 1 ? c.shifts[0] : c.shifts[index];
      src += c.name == "lshift_b" ? -shift : shift;
      if (c.name == "arshift_b") src = std::min<int64_t>(63, src);
    }
    if (src >= 0 && src < 64) out |= ((x >> src) & 1) << dst;
  }
  return out;
}

TEST(SpdzWiseBasicOps, BooleanViewsWidthsAndDeferredComposition) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  std::vector<std::vector<NdArrayRef>> inputs;
  std::vector<NdArrayRef> expected;
  for (size_t width : {1, 8, 32, 63, 64}) {
    const auto plain = Plain(width);
    inputs.push_back(io.toBooleanShares(plain, width));
    for (const auto& c : Cases(width))
      expected.push_back(Map(plain, [&](uint64_t x, int64_t i) {
        return BooleanReference(x, i, c);
      }));
  }
  expected.push_back(Map(Plain(), [](uint64_t x, int64_t) { return x >> 1; }));
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    std::vector<Value> external;
    for (const auto& batch : inputs)
      external.emplace_back(batch[link->Rank()], DT_I64);
    Begin(state, external);
    std::vector<Value> outputs;
    Value full;
    for (auto value : external) {
      auto x = import_s(&ctx, value);
      const auto width = x.storage_type().as<BShrTy>()->nbits();
      // A real secret AND leaves work pending before the local operations.
      x = and_bb(&ctx, x, x);
      x = Value(x.data().transpose({1, 0}).clone().transpose({1, 0}), DT_I64);
      if (width == 64) full = x;
      const auto original = x.data().clone();
      const auto pending = state->pendingBooleanBits();
      const auto checks = state->booleanChecks();
      const auto bytes = link->GetStats()->sent_bytes.load();
      for (const auto& c : Cases(width)) {
        Value out;
        if (c.name == "cast_type_b") {
          const auto target = makeType<BShrTy>(state->session(), c.width);
          EXPECT_EQ(
              common_type_b(&ctx, x.storage_type(), target),
              makeType<BShrTy>(state->session(), std::max(width, c.width)));
          out = cast_type_b(&ctx, x, target);
        } else if (c.name == "bitrev_b") {
          out = bitrev_b(&ctx, x, c.start, c.end);
        } else {
          out = dynDispatch(&ctx, c.name, x, c.shifts);
        }
        EXPECT_EQ(out.storage_type().as<BShrTy>()->nbits(),
                  OutputWidth(width, c));
        state->validateBoolean(out.data());
        outputs.push_back(out);
      }
      EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
      EXPECT_EQ(state->pendingBooleanBits(), pending);
      EXPECT_EQ(state->booleanChecks(), checks);
      EXPECT_EQ(x.storage_type().as<BShrTy>()->nbits(), width);
      const NdArrayView<RepShare> before(original), after(x.data());
      for (int64_t i = 0; i < x.numel(); ++i) EXPECT_EQ(before[i], after[i]);
    }
    auto shifted = rshift_b(&ctx, full, {1});
    outputs.push_back(b2a(&ctx, and_bb(&ctx, shifted, shifted)));
    return Export(&ctx, outputs);
  });
  ExpectOutputs(results, expected);
}

TEST(SpdzWiseBasicOps, ArithmeticShiftPreservesExtendedMacAndComposes) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto plain = Plain();
  const auto shares = io.toShares(plain, VIS_SECRET);
  std::vector<Sizes> shifts{{0},  {1},  {31},
                            {32}, {63}, {0, 1, 31, 32, 63, 7, 8, 2}};
  std::vector<NdArrayRef> expected;
  for (const auto& bits : shifts)
    expected.push_back(Map(plain, [&](uint64_t x, int64_t i) {
      return x << (bits.size() == 1 ? bits[0] : bits[i]);
    }));
  expected.push_back(
      Map(plain, [](uint64_t x, int64_t) { return (x << 1) * x; }));
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value external(shares[link->Rank()], DT_I64);
    Begin(state, {external});
    auto x = import_s(&ctx, external);
    x = Value(x.data().transpose({1, 0}).clone().transpose({1, 0}), DT_I64);
    // Leave a MAC batch pending and ensure shifts do not flush or append it.
    auto square = mul_aa(&ctx, x, x);
    const auto pending = BasicOpsTestAccess::pending(state);
    const auto checks = state->arithmeticChecks();
    const auto bytes = link->GetStats()->sent_bytes.load();
    std::vector<Value> out;
    const NdArrayView<ring::AuthShare> before(x.data());
    for (const auto& bits : shifts) {
      auto shifted = lshift_a(&ctx, x, bits);
      const NdArrayView<ring::AuthShare> after(shifted.data());
      for (int64_t i = 0; i < x.numel(); ++i)
        for (size_t c = 0; c < 4; ++c)
          EXPECT_EQ(after[i][c],
                    before[i][c] * (ring::Word{1}
                                    << (bits.size() == 1 ? bits[0] : bits[i])));
      out.push_back(shifted);
    }
    EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
    EXPECT_EQ(BasicOpsTestAccess::pending(state), pending);
    EXPECT_EQ(state->arithmeticChecks(), checks);
    // Authenticate both the shifted values and MACs through multiplication.
    out.push_back(a2b(&ctx, mul_aa(&ctx, out[1], x)));
    EXPECT_GT(square.numel(), 0);
    return Export(&ctx, out);
  });
  ExpectOutputs(results, expected);
}

TEST(SpdzWiseBasicOps, EmptyInputsAndInvalidParameters) {
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Begin(state);
    auto p = make_p(&ctx, 1, {2});
    auto a = p2a(&ctx, p), b = p2b(&ctx, p);
    for (const auto* op : {"lshift_a", "lshift_b", "rshift_b", "arshift_b"}) {
      const auto& x = std::string_view(op) == "lshift_a" ? a : b;
      for (const auto& invalid :
           std::vector<Sizes>{{-1}, {64}, {1000}, {}, {0, 1, 2}})
        EXPECT_ANY_THROW(dynDispatch(&ctx, op, x, invalid));
    }
    auto narrow = cast_type_b(&ctx, b, makeType<BShrTy>(state->session(), 8));
    EXPECT_ANY_THROW(arshift_b(&ctx, narrow, {1}));
    EXPECT_ANY_THROW(bitrev_b(&ctx, b, 5, 4));
    EXPECT_ANY_THROW(bitrev_b(&ctx, b, 0, 65));
    const auto foreign = makeType<BShrTy>("different-session", 8);
    EXPECT_ANY_THROW(common_type_b(&ctx, b.storage_type(), foreign));
    EXPECT_ANY_THROW(cast_type_b(&ctx, b, foreign));
    EXPECT_ANY_THROW(common_type_b(&ctx, b.storage_type(), a.storage_type()));
    EXPECT_ANY_THROW(cast_type_b(&ctx, b, a.storage_type()));
    auto malformed = narrow.data().clone();
    malformed.data<RepShare>()[0][0] |= uint64_t{1} << 8;
    EXPECT_ANY_THROW(lshift_b(&ctx, Value(malformed, DT_I64), {0}));
    auto alien = b.data().as(makeType<BShrTy>("different-session", 64));
    EXPECT_ANY_THROW(rshift_b(&ctx, Value(alien, DT_I64), {1}));
    auto alien_a = a.data().as(makeType<ring::AShrTy>("different-session"));
    EXPECT_ANY_THROW(lshift_a(&ctx, Value(alien_a, DT_I64), {1}));

    auto empty_p = make_p(&ctx, 0, {0, 2});
    auto empty_a = p2a(&ctx, empty_p), empty_b = p2b(&ctx, empty_p);
    std::vector<Value> outputs;
    const auto bytes = link->GetStats()->sent_bytes.load();
    for (const Sizes& bits : std::vector<Sizes>{{}, {63}}) {
      outputs.push_back(lshift_a(&ctx, empty_a, bits));
      outputs.push_back(arshift_b(&ctx, empty_b, bits));
      for (size_t width : {1, 8, 32, 63, 64}) {
        auto x = cast_type_b(&ctx, empty_b,
                             makeType<BShrTy>(state->session(), width));
        for (const auto* op : {"lshift_b", "rshift_b"}) {
          Value out = dynDispatch(&ctx, op, x, bits);
          EXPECT_EQ(out.storage_type().as<BShrTy>()->nbits(), width);
          outputs.push_back(out);
        }
      }
    }
    outputs.push_back(bitrev_b(&ctx, empty_b, 0, 64));
    EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
    for (const auto& out : outputs) EXPECT_EQ(out.shape(), empty_p.shape());
    Export(&ctx, outputs);
  });
}

TEST(SpdzWiseBasicOps, RandomStreamsFullWidthAndCheckedExport) {
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Begin(state);
    BasicOpsTestAccess::deterministicStreams(state);
    std::vector<Value> out;
    std::vector<NdArrayRef> expected;
    bool saw_top_bit = false;
    for (int64_t n : {0, 1, 7, 0, 65, 3}) {
      const auto ref_a = BasicOpsTestAccess::arithmeticOracle(state, n);
      const auto ref_b = BasicOpsTestAccess::booleanOracle(state, n);
      const auto continuation_a =
          BasicOpsTestAccess::arithmeticOracle(state, 3, n);
      const auto continuation_b =
          BasicOpsTestAccess::booleanOracle(state, 3, n);
      const auto pending = BasicOpsTestAccess::pending(state);
      const auto checks = state->arithmeticChecks();
      const auto boolean_pending = state->pendingBooleanBits();
      const auto boolean_checks = state->booleanChecks();
      const auto bytes = link->GetStats()->sent_bytes.load();
      auto b = rand_b(&ctx, {n});
      EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
      EXPECT_EQ(state->pendingBooleanBits(), boolean_pending);
      EXPECT_EQ(state->booleanChecks(), boolean_checks);
      EXPECT_EQ(b.storage_type(), makeType<BShrTy>(state->session(), 1));
      auto a = rand_a(&ctx, {n});
      EXPECT_EQ(state->arithmeticChecks(), checks);
      EXPECT_EQ(BasicOpsTestAccess::pending(state), pending + n);
      EXPECT_EQ(BasicOpsTestAccess::arithmeticOracle(state, 3), continuation_a);
      EXPECT_EQ(BasicOpsTestAccess::booleanOracle(state, 3), continuation_b);
      if (!n) {
        EXPECT_EQ(link->GetStats()->sent_bytes.load(), bytes);
      }
      NdArrayRef ea(makeType<RingTy>(FM64), {n}),
          eb(makeType<RingTy>(FM64), {n});
      for (int64_t i = 0; i < n; ++i) {
        const auto value = a.data().data<ring::AuthShare>()[i];
        EXPECT_EQ(value[0], ref_a[i][0]);
        EXPECT_EQ(value[1], ref_a[i][1]);
        EXPECT_EQ(b.data().data<RepShare>()[i], ref_b[i]);
        saw_top_bit |= (uint64_t(value[0]) >> 63) != 0;
        // Deterministic zero seeds give the same component at all ranks.
        ea.data<uint64_t>()[i] = uint64_t(ref_a[i][0] * 3);
        eb.data<uint64_t>()[i] = ref_b[i][0];
      }
      BasicOpsTestAccess::frozen(state, a);
      out.push_back(a);
      out.push_back(b);
      expected.push_back(ea);
      expected.push_back(eb);
      // Interleave both randomness consumers and deferred Boolean checking.
      auto product = and_bb(&ctx, b, b);
      auto converted = b2a(&ctx, product);
      out.push_back(converted);
      expected.push_back(eb);
    }
    EXPECT_TRUE(saw_top_bit);
    return std::make_pair(Export(&ctx, out), expected);
  });
  std::vector<std::vector<NdArrayRef>> external;
  for (const auto& result : results) external.push_back(result.first);
  ExpectOutputs(external, results[0].second);
}

TEST(SpdzWiseBasicOps, RandomMacFaultsRejectFrozenRecords) {
  for (size_t party = 0; party < 3; ++party) {
    for (bool consistent : {false, true}) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        Begin(state);
        auto a = rand_a(&ctx, {3});
        EXPECT_EQ(BasicOpsTestAccess::pending(state), 3);
        BasicOpsTestAccess::corrupt(state, party, consistent);
        EXPECT_ANY_THROW(state->prepareOutputs(&ctx, {"out"}, {a}));
        EXPECT_ANY_THROW(state->exportedValue("out"));
        EXPECT_ANY_THROW(rand_b(&ctx, {1}));
      });
    }
  }
}

TEST(SpdzWiseBasicOps, RegistrationDomainsAndSessionLifecycle) {
  for (auto domain : {SpdzWiseConfig::FIELD, SpdzWiseConfig::RING}) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(domain), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      for (const auto* op : {"lshift_b", "rshift_b", "arshift_b", "bitrev_b",
                             "common_type_b", "cast_type_b", "rand_b"})
        EXPECT_TRUE(ctx.hasKernel(op));
      EXPECT_EQ(ctx.hasKernel("lshift_a"), domain == SpdzWiseConfig::RING);
      EXPECT_EQ(ctx.hasKernel("rand_a"), domain == SpdzWiseConfig::RING);
      Begin(state);
      auto b = rand_b(&ctx, {3});
      Export(&ctx, {lshift_b(&ctx, b, {1})});
      EXPECT_ANY_THROW(lshift_b(&ctx, b, {1}));
      EXPECT_ANY_THROW(common_type_b(&ctx, b.storage_type(), b.storage_type()));
      EXPECT_ANY_THROW(rand_b(&ctx, {1}));
    });
  }
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    EXPECT_ANY_THROW(rand_a(&ctx, {1}));  // No session setup yet.
  });
}
}  // namespace
}  // namespace spu::mpc::spdzwise
