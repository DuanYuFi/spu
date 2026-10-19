// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/kernel/hal/ring.h"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/kernel/test_util.h"

namespace spu::kernel::hal {

namespace {
// Deliberately returns its first argument to expose accidental operand swaps
// or fallback execution. This tests dispatch only, not an MPC comparison.
class EchoLessLeft final : public Kernel {
 public:
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->pushOutput(Value(ctx->getParam<Value>(0)));
  }
};
}  // namespace

TEST(RingTest, ProtocolLessDispatchPreservesOperandOrder) {
  auto ctx = test::makeSPUContext();
  const auto p = _make_p(&ctx, 3, {1});
  const auto s = _p2s(&ctx, _make_p(&ctx, 5, {1}));
  const auto t = _p2s(&ctx, _make_p(&ctx, 7, {1}));
  ctx.prot()->regKernel<EchoLessLeft>("less_ss");
  ctx.prot()->regKernel<EchoLessLeft>("less_sp");
  ctx.prot()->regKernel<EchoLessLeft>("less_ps");
  EXPECT_EQ(_less(&ctx, s, t).data().buf(), s.data().buf());
  EXPECT_EQ(_less(&ctx, s, p).data().buf(), s.data().buf());
  EXPECT_EQ(_less(&ctx, p, s).data().buf(), p.data().buf());
  // Public/public retains the existing ring comparison, even with hooks.
  EXPECT_NE(_less(&ctx, p, p).data().buf(), p.data().buf());
}

TEST(RingTest, LessWithoutProtocolHooksRetainsFallback) {
  auto ctx = test::makeSPUContext();
  const auto p = _make_p(&ctx, 3, {1});
  const auto q = _make_p(&ctx, 5, {1});
  const auto s = _p2s(&ctx, p);
  const auto t = _p2s(&ctx, q);
  for (auto result : {_less(&ctx, s, t), _less(&ctx, s, q),
                      _less(&ctx, p, t)}) {
    result = _s2p(&ctx, result).setDtype(DT_U64);
    EXPECT_EQ(dump_public_as<uint64_t>(&ctx, result)(0), 1);
  }
}

TEST(RingTest, _bit_parity) {
  auto ctx = test::makeSPUContext();
  std::vector<std::pair<uint64_t, int>> para = {
      {0x0505, 0},
      {0x10505, 1},
      {0x050500000505, 0},
      {0x1505050505050505, 1},
  };
  for (auto [init, exp] : para) {
    auto input = _make_p(&ctx, init, {3});

    auto output = _bit_parity(&ctx, input, 64);

    output.setDtype(DT_U64);
    auto p_ret = hal::dump_public_as<int64_t>(&ctx, output);
    xt::xarray<int64_t> expected{exp, exp, exp};
    EXPECT_EQ(p_ret, expected);
  }
}

TEST(RingTest, _popcount) {
  auto ctx = test::makeSPUContext();
  std::vector<std::pair<uint64_t, int>> para = {
      {0x0505, 4},
      {0x10505, 5},
      {0x05050505, 8},
      {0x0500000000000005, 4},
  };
  for (auto [init, exp] : para) {
    auto input = _make_p(&ctx, init, {3});

    auto output = _popcount(&ctx, input, 64);

    output.setDtype(DT_U64);
    auto p_ret = hal::dump_public_as<int64_t>(&ctx, output);
    xt::xarray<int64_t> expected{exp, exp, exp};
    EXPECT_EQ(p_ret, expected);
  }
}
}  // namespace spu::kernel::hal
