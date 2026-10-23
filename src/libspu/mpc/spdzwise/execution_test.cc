// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

#include "libspu/core/xt_helper.h"
#include "libspu/device/api.h"
#include "libspu/device/io.h"
#include "libspu/device/pphlo/pphlo_executor.h"
#include "libspu/device/symbol_table.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/protocol_session.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::device {
namespace {

RuntimeConfig Config() {
  RuntimeConfig config(SPDZWISE, FM64);
  config.spdzwise_config.domain = SpdzWiseConfig::RING;
  config.enable_type_checker = true;
  return config;
}

constexpr char kMultiply[] = R"(
func.func @main(%x: tensor<4x!pphlo.secret<i64>>) -> tensor<4x!pphlo.secret<i64>> {
  %y = pphlo.multiply %x, %x : tensor<4x!pphlo.secret<i64>>
  return %y : tensor<4x!pphlo.secret<i64>>
})";

TEST(SpdzWiseExecution, CheckedOutputsAndInternalReuse) {
  const auto config = Config();
  IoClient io(3, config);
  const xt::xarray<int64_t> plain = {-3, -1, 0, 5};
  const auto shares = io.makeShares(plain, VIS_SECRET);
  const auto result = mpc::utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(config, link);
    mpc::Factory::RegisterProtocol(&ctx, link);
    auto* session = mpc::getProtocolSession(&ctx);
    EXPECT_NE(session, nullptr);
    SymbolTable env;
    env.setVar("x", shares[link->Rank()]);
    pphlo::PPHloExecutor executor;
    execute(&executor, &ctx, kMultiply, {"x"}, {"y"}, &env);
    EXPECT_FALSE(env.getVar("y").storage_type().isSerializable());
    EXPECT_ANY_THROW(env.getVar("y").toProto(32));
    const auto first =
        Value::fromProto(session->exportedValue("y").toProto(32));
    // Reuse the authenticated value in the environment, without SetVar/import
    // of an external representation or a new key/checking state.
    execute(&executor, &ctx, kMultiply, {"y"}, {"z"}, &env);
    EXPECT_ANY_THROW(session->exportedValue("y"));
    const auto second =
        Value::fromProto(session->exportedValue("z").toProto(32));
    return std::vector<Value>{first, second};
  });
  for (size_t j = 0; j < 2; ++j) {
    const std::vector<Value> outputs{result[0][j], result[1][j], result[2][j]};
    xt::xarray<int64_t> actual = xt::zeros<int64_t>({4});
    PtBufferView view(actual);
    io.combineShares(outputs, &view);
    for (size_t i = 0; i < 4; ++i) {
      const auto square = plain(i) * plain(i);
      EXPECT_EQ(actual(i), j == 0 ? square : square * square);
    }
  }
}

TEST(SpdzWiseExecution, BoundedSignedComparisonAndBooleanOutput) {
  IoClient io(3, Config());
  const xt::xarray<int64_t> x = {std::numeric_limits<int64_t>::min(),
                                 std::numeric_limits<int64_t>::max(), -1, 1};
  // All differences fit int64, as required by the original HAL comparison.
  const xt::xarray<int64_t> y = {0, 0, -1, 2};
  const auto sx = io.makeShares(x, VIS_SECRET);
  const auto sy = io.makeShares(y, VIS_SECRET);
  const std::string code = R"(
func.func @main(%x: tensor<4x!pphlo.secret<i64>>, %y: tensor<4x!pphlo.secret<i64>>) -> tensor<4x!pphlo.secret<i1>> {
  %z = pphlo.less %x, %y : (tensor<4x!pphlo.secret<i64>>, tensor<4x!pphlo.secret<i64>>) -> tensor<4x!pphlo.secret<i1>>
  return %z : tensor<4x!pphlo.secret<i1>>
})";
  const auto outputs = mpc::utils::simulate(3, [&](const auto& link) {
    auto config = Config();
    config.enable_hal_profile = link->Rank() == 0;
    SPUContext ctx(config, link);
    mpc::Factory::RegisterProtocol(&ctx, link);
    SymbolTable env;
    env.setVar("x", sx[link->Rank()]);
    env.setVar("y", sy[link->Rank()]);
    pphlo::PPHloExecutor executor;
    execute(&executor, &ctx, code, {"x", "y"}, {"z"}, &env);
    const auto* session = mpc::getProtocolSession(&ctx);
    return Value::fromProto(session->exportedValue("z").toProto(32));
  });
  xt::xarray<bool> actual = xt::zeros<bool>({4});
  PtBufferView view(actual);
  io.combineShares(outputs, &view);
  for (size_t i = 0; i < 4; ++i) EXPECT_EQ(actual(i), x(i) < y(i));
  auto reordered = outputs;
  std::swap(reordered[0], reordered[1]);
  EXPECT_ANY_THROW(io.combineShares(reordered, &view));
}

TEST(SpdzWiseExecution, EmptyOutputsStillCheckAndCommit) {
  IoClient io(3, Config());
  const xt::xarray<int64_t> plain = {-3, -1, 0, 5};
  const auto shares = io.makeShares(plain, VIS_SECRET);
  mpc::utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    mpc::Factory::RegisterProtocol(&ctx, link);
    SymbolTable env;
    env.setVar("x", shares[link->Rank()]);
    pphlo::PPHloExecutor executor;
    execute(&executor, &ctx, R"(
func.func @main(%x: tensor<4x!pphlo.secret<i64>>) {
  %y = pphlo.multiply %x, %x : tensor<4x!pphlo.secret<i64>>
  return
})",
            {"x"}, {}, &env);
    const auto* state = ctx.prot()->getState<mpc::spdzwise::SpdzWiseState>();
    EXPECT_GT(state->arithmeticChecks(), 0);
    EXPECT_FALSE(state->running());
    EXPECT_FALSE(state->failed());
  });
}

TEST(SpdzWiseExecution, FixedPointPpmlComposition) {
  IoClient io(3, Config());
  const xt::xarray<float> plain = {-0.5F, 0.0F, 0.5F, 1.0F};
  const auto shares = io.makeShares(plain, VIS_SECRET);
  const auto outputs = mpc::utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    mpc::Factory::RegisterProtocol(&ctx, link);
    SymbolTable env;
    env.setVar("x", shares[link->Rank()]);
    pphlo::PPHloExecutor executor;
    // LR's exp/reciprocal, NN's maximum, and BatchNorm's rsqrt compose
    // with fixed-point multiplication, truncation and secret matrix products.
    execute(&executor, &ctx, R"(
func.func @main(%x: tensor<4x!pphlo.secret<f32>>) -> (tensor<4x!pphlo.secret<f32>>, tensor<4x!pphlo.secret<f32>>, tensor<4x!pphlo.secret<f32>>, tensor<1x1x!pphlo.secret<f32>>) {
  %one = pphlo.constant dense<1.0> : tensor<4xf32>
  %zero = pphlo.constant dense<0.0> : tensor<4xf32>
  %neg = pphlo.negate %x : tensor<4x!pphlo.secret<f32>>
  %exp = pphlo.exponential %neg : tensor<4x!pphlo.secret<f32>>
  %den = pphlo.add %exp, %one : (tensor<4x!pphlo.secret<f32>>, tensor<4xf32>) -> tensor<4x!pphlo.secret<f32>>
  %sigmoid = pphlo.reciprocal %den : tensor<4x!pphlo.secret<f32>>
  %sq = pphlo.multiply %x, %x : tensor<4x!pphlo.secret<f32>>
  %pos = pphlo.add %sq, %one : (tensor<4x!pphlo.secret<f32>>, tensor<4xf32>) -> tensor<4x!pphlo.secret<f32>>
  %rsqrt = pphlo.rsqrt %pos : tensor<4x!pphlo.secret<f32>>
  %relu = pphlo.maximum %x, %zero : (tensor<4x!pphlo.secret<f32>>, tensor<4xf32>) -> tensor<4x!pphlo.secret<f32>>
  %row = pphlo.reshape %x : (tensor<4x!pphlo.secret<f32>>) -> tensor<1x4x!pphlo.secret<f32>>
  %col = pphlo.reshape %x : (tensor<4x!pphlo.secret<f32>>) -> tensor<4x1x!pphlo.secret<f32>>
  %dot = pphlo.dot %row, %col : (tensor<1x4x!pphlo.secret<f32>>, tensor<4x1x!pphlo.secret<f32>>) -> tensor<1x1x!pphlo.secret<f32>>
  return %sigmoid, %rsqrt, %relu, %dot : tensor<4x!pphlo.secret<f32>>, tensor<4x!pphlo.secret<f32>>, tensor<4x!pphlo.secret<f32>>, tensor<1x1x!pphlo.secret<f32>>
})",
            {"x"}, {"sigmoid", "rsqrt", "relu", "dot"}, &env);
    const auto* session = mpc::getProtocolSession(&ctx);
    std::vector<Value> values;
    for (const auto* name : {"sigmoid", "rsqrt", "relu", "dot"})
      values.push_back(
          Value::fromProto(session->exportedValue(name).toProto(32)));
    return values;
  });
  for (size_t j = 0; j < 4; ++j) {
    const std::vector<Value> values{outputs[0][j], outputs[1][j],
                                    outputs[2][j]};
    xt::xarray<float> actual = xt::zeros<float>(
        j == 3 ? std::vector<size_t>{1, 1} : std::vector<size_t>{4});
    PtBufferView view(actual);
    io.combineShares(values, &view);
    for (size_t i = 0; i < actual.size(); ++i) {
      const auto x = plain(i);
      const float expected = j == 0   ? 1.0F / (1.0F + std::exp(-x))
                             : j == 1 ? 1.0F / std::sqrt(x * x + 1.0F)
                             : j == 2 ? std::max(x, 0.0F)
                                      : 1.5F;
      EXPECT_NEAR(actual.flat(i), expected, 0.01F) << j << ":" << i;
    }
  }
}

TEST(SpdzWiseExecution, ExecutionFailurePoisonsSessionAndRevokesExports) {
  IoClient io(3, Config());
  const xt::xarray<int64_t> plain = {-3, -1, 0, 5};
  const auto shares = io.makeShares(plain, VIS_SECRET);
  mpc::utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    mpc::Factory::RegisterProtocol(&ctx, link);
    SymbolTable env;
    env.setVar("x", shares[link->Rank()]);
    pphlo::PPHloExecutor executor;
    execute(&executor, &ctx, kMultiply, {"x"}, {"y"}, &env);
    auto* session = mpc::getProtocolSession(&ctx);
    EXPECT_NO_THROW(session->exportedValue("y"));
    EXPECT_ANY_THROW(execute(&executor, &ctx, "invalid mlir", {}, {}, &env));
    EXPECT_TRUE(session->failed());
    EXPECT_ANY_THROW(session->exportedValue("y"));
    EXPECT_ANY_THROW(execute(&executor, &ctx, kMultiply, {"x"}, {"y"}, &env));
  });
}

TEST(SpdzWiseExecution, FactoryRejectsIncompleteFieldAndUnsafeOptions) {
  mpc::utils::simulate(3, [&](const auto& link) {
    auto config = Config();
    config.spdzwise_config.domain = SpdzWiseConfig::FIELD;
    SPUContext field(config, link);
    EXPECT_ANY_THROW(mpc::Factory::RegisterProtocol(&field, link));
    config = Config();
    config.enable_runtime_snapshot = true;
    SPUContext snapshot(config, link);
    EXPECT_ANY_THROW(mpc::Factory::RegisterProtocol(&snapshot, link));
    config = Config();
    config.experimental_enable_inter_op_par = true;
    SPUContext parallel(config, link);
    EXPECT_ANY_THROW(mpc::Factory::RegisterProtocol(&parallel, link));
  });
}

TEST(SpdzWiseExecution, MacFailureBlocksPublicationAtEveryParty) {
  IoClient io(3, Config());
  const xt::xarray<int64_t> plain = {-3, -1, 0, 5};
  const auto shares = io.makeShares(plain, VIS_SECRET);
  for (size_t corrupt_rank = 0; corrupt_rank < 3; ++corrupt_rank) {
    mpc::utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      mpc::Factory::RegisterProtocol(&ctx, link);
      SymbolTable env;
      env.setVar("x", shares[link->Rank()]);
      pphlo::PPHloExecutor executor;
      execute(&executor, &ctx, kMultiply, {"x"}, {"y"}, &env);
      if (link->Rank() == corrupt_rank) {
        env.getVar("y").data().data<mpc::spdzwise::ring::AuthShare>()[0][2] ^=
            1;
      }
      EXPECT_ANY_THROW(execute(&executor, &ctx, kMultiply, {"y"}, {"z"}, &env));
      const auto* session = mpc::getProtocolSession(&ctx);
      EXPECT_TRUE(session->failed());
      EXPECT_ANY_THROW(session->exportedValue("y"));
      EXPECT_ANY_THROW(session->exportedValue("z"));
      EXPECT_FALSE(env.hasVar("z"));
    });
  }
}

}  // namespace
}  // namespace spu::device
