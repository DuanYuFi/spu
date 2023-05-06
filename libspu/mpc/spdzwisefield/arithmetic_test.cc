// Copyright 2021 Ant Group Co., Ltd.
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

#include "libspu/mpc/spdzwisefield/arithmetic.h"

#include "gtest/gtest.h"

#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/spdzwisefield/protocol.h"
#include "libspu/mpc/spdzwisefield/state.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

#define MYLOG(x) \
  if (comm->getRank() == 0) std::cout << x << std::endl
namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_field(field);
  return conf;
}

}  // namespace

using ArithTestParams = std::tuple<int, PtType, size_t>;
class ArithmeticTest : public ::testing::TestWithParam<ArithTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    SpdzwiseField, ArithmeticTest,
    testing::Combine(testing::Values(1),               //
                     testing::Values(PtType::PT_U64),  //
                     testing::Values(3)),              //
    [](const testing::TestParamInfo<ArithmeticTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(ArithmeticTest, IOTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();

    using Field = SpdzWiseFieldState::Field;

    const size_t test_size = 100000;

    std::vector<uint64_t> a(test_size);
    prg_state->fillPubl(absl::MakeSpan(a));

    ArrayRef pub(makeType<Pub2kTy>(FM64), test_size);
    auto _pub = ArrayView<uint64_t>(pub);

    pforeach(0, test_size, [&](uint64_t idx) {
      a[idx] = Field::modp(a[idx]);
      _pub[idx] = a[idx];
    });

    ArrayRef ashare = obj->call("p2a", pub);
    ArrayRef pub2 = obj->call("a2p", ashare);

    auto _pub2 = ArrayView<uint64_t>(pub2);

    pforeach(0, test_size,
             [&](uint64_t idx) { EXPECT_EQ(_pub[idx], _pub2[idx]); });
  });
}

TEST_P(ArithmeticTest, MulAATest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();
    auto* spdzwisefield_state = obj->getState<SpdzWiseFieldState>();

    using Field = SpdzWiseFieldState::Field;

    const size_t test_size = 100000;

    std::vector<uint64_t> a(test_size);
    std::vector<uint64_t> b(test_size);

    prg_state->fillPubl(absl::MakeSpan(a));
    prg_state->fillPubl(absl::MakeSpan(b));

    ArrayRef lhs(makeType<Pub2kTy>(FM64), test_size);
    ArrayRef rhs(makeType<Pub2kTy>(FM64), test_size);

    auto _lhs = ArrayView<uint64_t>(lhs);
    auto _rhs = ArrayView<uint64_t>(rhs);

    pforeach(0, test_size, [&](uint64_t idx) {
      a[idx] = Field::modp(a[idx]);
      b[idx] = Field::modp(b[idx]);

      _lhs[idx] = a[idx];
      _rhs[idx] = b[idx];
    });

    ArrayRef lhs_share = obj->call("p2a", lhs);
    ArrayRef rhs_share = obj->call("p2a", rhs);

    ArrayRef mul_share = obj->call("mul_aa", lhs_share, rhs_share);

    ArrayRef mul = obj->call("a2p", mul_share);
    auto _mul = ArrayView<uint64_t>(mul);

    pforeach(0, test_size, [&](uint64_t idx) {
      uint64_t expected = Field::mul(a[idx], b[idx]);
      SPU_ENFORCE(_mul[idx] == expected,
                  "failed at index = {}, expect = {}, actual = {}", idx,
                  expected, _mul[idx]);
    });

    spdzwisefield_state->verification(obj.get(), true);
  });
}

TEST_P(ArithmeticTest, MulAPTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();
    auto* spdzwisefield_state = obj->getState<SpdzWiseFieldState>();

    using Field = SpdzWiseFieldState::Field;

    const size_t test_size = 100000;

    std::vector<uint64_t> a(test_size);
    std::vector<uint64_t> b(test_size);

    prg_state->fillPubl(absl::MakeSpan(a));
    prg_state->fillPubl(absl::MakeSpan(b));

    ArrayRef lhs(makeType<Pub2kTy>(FM64), test_size);
    ArrayRef rhs(makeType<Pub2kTy>(FM64), test_size);

    auto _lhs = ArrayView<uint64_t>(lhs);
    auto _rhs = ArrayView<uint64_t>(rhs);

    pforeach(0, test_size, [&](uint64_t idx) {
      a[idx] >>= 4;
      b[idx] >>= 4;

      _lhs[idx] = a[idx];
      _rhs[idx] = b[idx];
    });

    ArrayRef lhs_share = obj->call("p2a", lhs);

    ArrayRef mul_share = obj->call("mul_ap", lhs_share, rhs);

    ArrayRef mul = obj->call("a2p", mul_share);
    auto _mul = ArrayView<uint64_t>(mul);

    pforeach(0, test_size, [&](uint64_t idx) {
      uint64_t expected = Field::mul(a[idx], b[idx]);
      SPU_ENFORCE(_mul[idx] == expected,
                  "failed at index = {}, expect = {}, actual = {}", idx,
                  expected, _mul[idx]);
    });

    spdzwisefield_state->verification(obj.get(), true);
  });
}

TEST_P(ArithmeticTest, TruncTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();

    using T = uint64_t;

    const size_t bit_length = 60;
    const uint128_t mod = 1ULL << bit_length;

    const size_t test_size = 100000;

    std::vector<conversion::BitStream> _lhs(test_size);
    std::vector<conversion::BitStream> _rhs(test_size);

    std::vector<T> data(test_size);

    prg_state->fillPubl(absl::MakeSpan(data));

    pforeach(0, test_size, [&](uint64_t idx) { data[idx] %= mod; });

    ArrayRef in(makeType<Pub2kTy>(FM64), test_size);
    auto _in = ArrayView<uint64_t>(in);

    pforeach(0, test_size, [&](uint64_t idx) { _in[idx] = data[idx]; });

    ArrayRef in_share = obj->call("p2a", in);
    ArrayRef truncated = trunc_a(obj.get(), in_share, 20);
    ArrayRef out = obj->call("a2p", truncated);

    auto _out = ArrayView<uint64_t>(out);

    pforeach(0, test_size, [&](uint64_t idx) {
      SPU_ENFORCE(abs(static_cast<int64_t>(_out[idx] - (data[idx] >> 20))) <= 1,
                  "Failed");
    });
  });
}

}  // namespace spu::mpc::test