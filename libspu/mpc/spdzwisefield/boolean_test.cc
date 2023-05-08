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

#include "libspu/mpc/spdzwisefield/boolean.h"

#include "gtest/gtest.h"

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

using CACOpTestParams = std::tuple<int, PtType, size_t>;
class BooleanTest : public ::testing::TestWithParam<CACOpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Beaver, BooleanTest,
    testing::Combine(testing::Values(1e6),             //
                     testing::Values(PtType::PT_U32),  //
                     testing::Values(3)),              //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(BooleanTest, IOTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    const size_t test_size = std::get<0>(GetParam());

    // auto* comm = obj->getState<Communicator>();
    auto* prg_state = obj->getState<PrgState>();

    std::vector<uint64_t> a(test_size);

    prg_state->fillPubl(absl::MakeSpan(a));

    ArrayRef lhs(makeType<Pub2kTy>(FM64), test_size);

    auto _lhs = ArrayView<uint64_t>(lhs);

    pforeach(0, test_size, [&](uint64_t idx) { _lhs[idx] = a[idx]; });

    ArrayRef lhs_share = obj->call("p2b", lhs);

    ArrayRef result = obj->call("b2p", lhs_share);
    auto _result = ArrayView<uint64_t>(result);

    pforeach(0, test_size, [&](uint64_t idx) {
      uint64_t expected = a[idx];
      uint64_t actual = _result[idx];
      SPU_ENFORCE(expected == actual, "idx: {}, expected: {}, actual: {}", idx,
                  expected, actual);
    });
  });
}

TEST_P(BooleanTest, AndTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    const size_t test_size = std::get<0>(GetParam());

    auto* prg_state = obj->getState<PrgState>();

    std::vector<uint64_t> a(test_size);
    std::vector<uint64_t> b(test_size);

    prg_state->fillPubl(absl::MakeSpan(a));
    prg_state->fillPubl(absl::MakeSpan(b));

    ArrayRef lhs(makeType<Pub2kTy>(FM64), test_size);
    ArrayRef rhs(makeType<Pub2kTy>(FM64), test_size);

    auto _lhs = ArrayView<uint64_t>(lhs);
    auto _rhs = ArrayView<uint64_t>(rhs);

    pforeach(0, test_size, [&](uint64_t idx) {
      _lhs[idx] = a[idx];
      _rhs[idx] = b[idx];
    });

    ArrayRef lhs_share = obj->call("p2b", lhs);
    ArrayRef rhs_share = obj->call("p2b", rhs);

    ArrayRef result_share = obj->call("and_bb", lhs_share, rhs_share);

    ArrayRef result = obj->call("b2p", result_share);
    auto _result = ArrayView<uint64_t>(result);

    pforeach(0, test_size, [&](uint64_t idx) {
      uint64_t expected = a[idx] & b[idx];
      uint64_t actual = _result[idx];
      SPU_ENFORCE(expected == actual, "idx: {}, expected: {}, actual: {}", idx,
                  expected, actual);
    });
  });
}

}  // namespace spu::mpc::test