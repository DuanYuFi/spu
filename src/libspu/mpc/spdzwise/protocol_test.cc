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

#include "gtest/gtest.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/spdzwise/arithmetic.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/replicated.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
namespace {

NdArrayRef Matrix(const Shape& shape, std::initializer_list<int64_t> values) {
  NdArrayRef result(makeType<RingTy>(FM64), shape);
  NdArrayView<uint64_t> view(result);
  size_t i = 0;
  for (auto x : values) {
    view[i++] = static_cast<uint64_t>(x);
  }
  return result;
}

RuntimeConfig Config() {
  RuntimeConfig config;
  config.protocol = SPDZWISE;
  config.field = FM64;
  config.fxp_fraction_bits = 16;
  return config;
}

TEST(SpdzWiseProtocol, PrssCopiesStayAlignedAcrossOddAndEmptyBatches) {
  const auto draws = utils::simulate(3, [&](const auto& link) {
    PrgState prg(link);
    const auto before = link->GetStats()->sent_bytes.load();
    std::vector<std::vector<RepShare>> batches;
    for (size_t size : {0U, 1U, 7U, 1025U}) {
      batches.push_back(RandomF61Rep(&prg, size));
    }
    EXPECT_EQ(link->GetStats()->sent_bytes.load(), before);
    return batches;
  });
  for (size_t batch = 0; batch < draws[0].size(); ++batch) {
    for (size_t rank = 0; rank < 3; ++rank) {
      for (size_t i = 0; i < draws[rank][batch].size(); ++i) {
        EXPECT_LT(draws[rank][batch][i][0], kPrime);
        EXPECT_LT(draws[rank][batch][i][1], kPrime);
        EXPECT_EQ(draws[rank][batch][i][1], draws[(rank + 1) % 3][batch][i][0]);
      }
    }
  }
}

TEST(SpdzWiseProtocol, FieldRouteUsesF61MultiplicationAndSharedBooleanKernels) {
  auto config = Config();
  config.spdzwise_config.domain = SpdzWiseConfig::FIELD;
  SpdzWiseIo io(FM64, 3);
  const auto lhs = io.toShares(Matrix({1}, {kHalfPrime}), VIS_SECRET);
  const auto rhs = io.toShares(Matrix({1}, {2}), VIS_SECRET);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(config, link);
    registerArithmeticPrototype(&ctx, link);
    EXPECT_NE(dynamic_cast<field::MulAA*>(ctx.prot()->getKernel("mul_aa")),
              nullptr);
    EXPECT_NE(dynamic_cast<field::MulAP*>(ctx.prot()->getKernel("mul_ap")),
              nullptr);
    EXPECT_NE(dynamic_cast<field::MatmulAA*>(ctx.prot()->getKernel("mmul_aa")),
              nullptr);
    EXPECT_NE(dynamic_cast<field::MatmulAP*>(ctx.prot()->getKernel("mmul_ap")),
              nullptr);
    EXPECT_TRUE(ctx.hasKernel("and_bb"));
    EXPECT_TRUE(ctx.hasKernel("xor_bb"));
    EXPECT_FALSE(ctx.hasKernel("a2b"));
    EXPECT_FALSE(ctx.hasKernel("b2a"));
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value a(lhs[link->Rank()], DT_I64), b(rhs[link->Rank()], DT_I64);
    ProtocolRunGuard guard(&ctx);
    state->begin(ExecutableProto("field-route", {"a", "b"}, {"c"}, "test"),
                 {a, b});
    a = import_s(&ctx, a);
    b = import_s(&ctx, b);
    auto result = dynDispatch(&ctx, "mul_aa", a, b);
    state->prepareOutputs(&ctx, {"c"}, {result});
    guard.commit();
    return state->exportedValue("c").data();
  });
  // 2 * ((p - 1) / 2) = p - 1 represents -1 in F61, unlike ring64.
  EXPECT_EQ(NdArrayView<uint64_t>(io.fromShares(outputs))[0], uint64_t{0} - 1);
}

TEST(SpdzWiseProtocol, RingRouteRemainsEmptyAndCannotUseFieldState) {
  auto config = Config();
  config.spdzwise_config.domain = SpdzWiseConfig::RING;
  SPUContext ctx(config, nullptr);
  EXPECT_ANY_THROW(registerArithmeticPrototype(&ctx, nullptr));
  EXPECT_FALSE(ctx.hasKernel("mul_aa"));
  EXPECT_FALSE(ctx.hasKernel("and_bb"));
  EXPECT_FALSE(ctx.prot()->hasState(ProtocolSession::kBindName()));
  EXPECT_ANY_THROW(registerFieldArithmeticPrototype(&ctx, nullptr));
  utils::simulate(3, [&](const auto& link) {
    EXPECT_ANY_THROW(SpdzWiseState(config, link));
  });
}

TEST(SpdzWiseProtocol, MultiplicationPayloadAndConstantSizeVerification) {
  SpdzWiseIo io(FM64, 3);
  for (int64_t size : {1, 101, 65537}) {
    NdArrayRef input(makeType<RingTy>(FM64), {size});
    NdArrayView<uint64_t> plain(input);
    for (int64_t i = 0; i < size; ++i) {
      plain[i] = static_cast<uint64_t>(i - 50);
    }
    const auto shares = io.toShares(input, VIS_SECRET);
    const auto outputs = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      Value x(shares[link->Rank()], DT_I64);
      state->begin(ExecutableProto("comm", {"x"}, {"y"}, "test"), {x});
      x = import_s(&ctx, x);
      const auto stats = link->GetStats();
      auto before = stats->sent_bytes.load();
      Value y(state->multiply(x.data(), x.data()), DT_I64);
      EXPECT_EQ(stats->sent_bytes.load() - before, 16 * size);
      before = stats->sent_bytes.load();
      const auto actions = stats->sent_actions.load();
      const auto checks = state->arithmeticChecks();
      state->check();
      EXPECT_EQ(stats->sent_bytes.load() - before, 64);
      // One batched dot exchange, two scalar multiplications, and the final
      // opening's two peer sends. The old separate dots needed six sends.
      EXPECT_EQ(stats->sent_actions.load() - actions, 5);
      EXPECT_EQ(state->arithmeticChecks(), checks + 1);
      state->prepareOutputs(&ctx, {"y"}, {y});
      state->commit();
      EXPECT_EQ(state->arithmeticChecks(), checks + 1);
      return state->exportedValue("y").data();
    });
    const auto reconstructed = io.fromShares(outputs);
    const NdArrayView<uint64_t> actual(reconstructed);
    for (int64_t i = 0; i < size; ++i) {
      const auto x = static_cast<int64_t>(plain[i]);
      EXPECT_EQ(actual[i], static_cast<uint64_t>(x * x));
    }
  }
}

TEST(SpdzWiseProtocol, F61WireEncodingRejectsInvalidResiduesAndLengths) {
  const std::vector<uint64_t> words = {0, 1, kHalfPrime, kPrime - 1};
  EXPECT_EQ(DecodeF61(EncodeF61(words), words.size()), words);
  EXPECT_EQ(EncodeF61({1}), std::string("\1\0\0\0\0\0\0\0", 8));
  EXPECT_ANY_THROW(DecodeF61(EncodeF61({kPrime}), 1));
  EXPECT_ANY_THROW(DecodeF61("short", 1));
  EXPECT_TRUE(DecodeF61("", 0).empty());
}

TEST(SpdzWiseProtocol, ArithmeticTransactionAndMacFreeExport) {
  SpdzWiseIo io(FM64, 3);
  const auto input = Matrix({2, 3}, {-3, -2, -1, 0, 1, 2});
  const auto shares = io.toShares(input, VIS_SECRET);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value x(shares[link->Rank()], DT_I64);
    ProtocolRunGuard guard(&ctx);
    state->begin(ExecutableProto("arithmetic", {"x"}, {"result"}, "test"), {x});
    x = import_s(&ctx, x);
    EXPECT_EQ(x.dtype(), DT_I64);
    EXPECT_ANY_THROW(x.toProto(1024));
    auto square = state->multiply(x.data(), x.data());
    auto result = state->add(square, state->negate(x.data()));
    EXPECT_ANY_THROW(state->exportedValue("result"));
    state->prepareOutputs(&ctx, {"result"}, {Value(result, DT_I64)});
    EXPECT_ANY_THROW(state->exportedValue("result"));
    guard.commit();
    EXPECT_GE(state->arithmeticChecks(), 2);
    const auto output = state->exportedValue("result");
    EXPECT_EQ(output.elsize(), sizeof(RepShare));
    EXPECT_NO_THROW(output.toProto(1024));
    return output.data();
  });
  const auto output = io.fromShares(outputs);
  const NdArrayView<uint64_t> in(input), out(output);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const auto x = static_cast<int64_t>(in[i]);
    EXPECT_EQ(out[i], static_cast<uint64_t>(x * x - x));
  }
}

TEST(SpdzWiseProtocol, MatrixProductAndAuthenticatedReuse) {
  SpdzWiseIo io(FM64, 3);
  const auto x = io.toShares(Matrix({2, 3}, {-3, -2, -1, 0, 1, 2}), VIS_SECRET);
  const auto y = io.toShares(Matrix({3, 2}, {1, 2, 3, 4, 5, 6}), VIS_SECRET);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value a(x[link->Rank()], DT_I64), b(y[link->Rank()], DT_I64), result;
    {
      ProtocolRunGuard guard(&ctx);
      state->begin(ExecutableProto("matrix", {"a", "b"}, {"c"}, "test"),
                   {a, b});
      a = import_s(&ctx, a);
      b = import_s(&ctx, b);
      result = Value(state->matmul(a.data(), b.data(), false), DT_I64);
      state->prepareOutputs(&ctx, {"c"}, {result});
      guard.commit();
    }
    const auto checked = state->arithmeticChecks();
    {
      ProtocolRunGuard guard(&ctx);
      state->begin(ExecutableProto("identity", {"c"}, {"d"}, "test"), {result});
      result = import_s(&ctx, result);
      state->prepareOutputs(&ctx, {"d"}, {result});
      guard.commit();
    }
    EXPECT_EQ(checked, state->arithmeticChecks());
    return state->exportedValue("d").data();
  });
  const auto output = io.fromShares(outputs);
  const NdArrayView<uint64_t> out(output);
  const int64_t expected[] = {-14, -20, 13, 16};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(out[i], static_cast<uint64_t>(expected[i]));
  }
}

TEST(SpdzWiseProtocol, PublicArithmeticAndShapeViewsPreserveMacs) {
  SpdzWiseIo io(FM64, 3);
  const auto shares =
      io.toShares(Matrix({2, 3}, {-3, -2, -1, 0, 1, 2}), VIS_SECRET);
  const auto public_shares =
      io.toShares(Matrix({3, 2}, {-2, -2, -2, -2, -2, -2}), VIS_PUBLIC);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value x(shares[link->Rank()], DT_I64);
    Value p(public_shares[link->Rank()], DT_I64);
    ProtocolRunGuard guard(&ctx);
    state->begin(ExecutableProto("mixed", {"x", "p"}, {"y"}, "test"), {x, p});
    x = import_s(&ctx, x);
    // Exercise the actual registered shape and arithmetic kernels, not just
    // the state methods. A transpose produces a noncontiguous share view.
    x = dynDispatch(&ctx, "transpose", x, Axes{1, 0});
    auto y = dynDispatch(&ctx, "mul_ap", x, p);
    y = dynDispatch(&ctx, "add_ap", y, p);
    // A checked multiplication makes misplaced MACs observable.
    y = dynDispatch(&ctx, "mul_aa", y, y);
    y.setDtype(DT_I64);
    state->prepareOutputs(&ctx, {"y"}, {y});
    guard.commit();
    return state->exportedValue("y").data();
  });
  const auto output = io.fromShares(outputs);
  const NdArrayView<uint64_t> out(output);
  const uint64_t expected[] = {16, 4, 4, 16, 0, 36};
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(out[i], expected[i]);
  }
}

TEST(SpdzWiseProtocol, CorruptedMacPoisonsSessionForEveryParty) {
  SpdzWiseIo io(FM64, 3);
  const auto shares = io.toShares(Matrix({1}, {7}), VIS_SECRET);
  for (size_t corrupt = 0; corrupt < 3; ++corrupt) {
    const auto rejected = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      bool caught = false;
      try {
        ProtocolRunGuard guard(&ctx);
        Value x(shares[link->Rank()], DT_I64);
        state->begin(ExecutableProto("fault", {"x"}, {"y"}, "test"), {x});
        x = import_s(&ctx, x);
        if (link->Rank() == corrupt) {
          NdArrayView<AuthShare> view(x.data());
          view[0][2] = Add(view[0][2], 1);
        }
        Value y(state->multiply(x.data(), x.data()), DT_I64);
        state->prepareOutputs(&ctx, {"y"}, {y});
        guard.commit();
      } catch (const std::exception&) {
        caught = true;
      }
      EXPECT_TRUE(state->failed());
      EXPECT_ANY_THROW(state->exportedValue("y"));
      EXPECT_ANY_THROW(state->requireHealthy());
      return caught;
    });
    for (bool value : rejected) {
      EXPECT_TRUE(value);
    }
  }
}

TEST(SpdzWiseProtocol, ParallelBatchRejectsMacCorruptionAtDifferentPositions) {
  // Cross the parallel-reduction threshold with an odd length. Tamper near
  // the beginning, middle and end, covering different reduction chunks and
  // all three parties. Imports also exercise a valid large-batch check.
  constexpr int64_t size = 65537;
  NdArrayRef input(makeType<RingTy>(FM64), {size});
  auto* plain = input.data<uint64_t>();
  for (int64_t i = 0; i < size; ++i) {
    plain[i] = static_cast<uint64_t>(i + 7);
  }
  SpdzWiseIo io(FM64, 3);
  const auto shares = io.toShares(input, VIS_SECRET);
  const std::array<int64_t, 3> positions = {0, size / 2, size - 1};
  for (size_t corrupt = 0; corrupt < 3; ++corrupt) {
    const auto rejected = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      bool caught = false;
      try {
        ProtocolRunGuard guard(&ctx);
        Value x(shares[link->Rank()], DT_I64);
        state->begin(ExecutableProto("batch-fault", {"x"}, {"y"}, "test"), {x});
        x = import_s(&ctx, x);
        if (link->Rank() == corrupt) {
          auto& mac = x.data().data<AuthShare>()[positions[corrupt]][2];
          mac = Add(mac, 1);
        }
        Value y(state->multiply(x.data(), x.data()), DT_I64);
        state->prepareOutputs(&ctx, {"y"}, {y});
        guard.commit();
      } catch (const std::exception&) {
        caught = true;
      }
      EXPECT_TRUE(state->failed());
      EXPECT_ANY_THROW(state->exportedValue("y"));
      return caught;
    });
    for (bool value : rejected) {
      EXPECT_TRUE(value);
    }
  }
}

TEST(SpdzWiseProtocol, RejectsMixedOutputBatchesBeforeAuthentication) {
  SpdzWiseIo io(FM64, 3);
  auto shares = io.toShares(Matrix({1}, {7}), VIS_SECRET);
  for (size_t rank = 0; rank < 3; ++rank) {
    shares[rank] = shares[rank].as(
        makeType<OutputTy>(rank, std::string(64, rank == 1 ? 'b' : 'a')));
  }
  const auto rejected = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    try {
      state->begin(ExecutableProto("identity", {"x"}, {"y"}, "test"),
                   {Value(shares[link->Rank()], DT_I64)});
    } catch (const std::exception&) {
      return state->failed() && state->arithmeticChecks() == 0;
    }
    return false;
  });
  for (bool value : rejected) {
    EXPECT_TRUE(value);
  }
}

}  // namespace
}  // namespace spu::mpc::spdzwise
