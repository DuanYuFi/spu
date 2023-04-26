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

#include "gtest/gtest.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/spdzwisefield/protocol.h"
#include "libspu/mpc/spdzwisefield/state.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::test {
namespace {

RuntimeConfig makeConfig(FieldType field) {
  RuntimeConfig conf;
  conf.set_field(field);
  return conf;
}

}  // namespace

using CACOpTestParams = std::tuple<int, PtType, size_t>;
class CircuitTest : public ::testing::TestWithParam<CACOpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Edabit, CircuitTest,
    testing::Combine(testing::Values(1000000),         //
                     testing::Values(PtType::PT_U32),  //
                     testing::Values(3)),              //
    [](const testing::TestParamInfo<CircuitTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(CircuitTest, FullAdder) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    auto* comm = obj->getState<Communicator>();

    conversion::BitStream b1, b2;

    // 0b00111 + 0b11101 = 0b100100

    switch (comm->getRank()) {
      case 0: {
        b1.push_back({0, 1});
        b1.push_back({1, 1});
        b1.push_back({0, 0});
        b1.push_back({0, 0});
        b1.push_back({1, 0});

        b2.push_back({0, 0});
        b2.push_back({1, 1});
        b2.push_back({1, 0});
        b2.push_back({0, 1});
        b2.push_back({1, 0});

        break;
      }
      case 1: {
        b1.push_back({1, 0});
        b1.push_back({1, 1});
        b1.push_back({0, 1});
        b1.push_back({0, 0});
        b1.push_back({0, 1});

        b2.push_back({0, 1});
        b2.push_back({1, 0});
        b2.push_back({0, 0});
        b2.push_back({1, 0});
        b2.push_back({0, 0});

        break;
      }
      case 2: {
        b1.push_back({0, 0});
        b1.push_back({1, 1});
        b1.push_back({1, 0});
        b1.push_back({0, 0});
        b1.push_back({1, 1});

        b2.push_back({1, 0});
        b2.push_back({0, 1});
        b2.push_back({0, 1});
        b2.push_back({0, 0});
        b2.push_back({0, 1});

        break;
      }

      default: {
        break;
      }
    }

    std::vector<conversion::BitStream> lhs, rhs;
    lhs.push_back(b1);
    rhs.push_back(b2);

    auto res = full_adder_with_check(obj.get(), lhs, rhs)[0];

    sleep(comm->getRank());

    std::cout << "Player " << comm->getRank() << std::endl;
    for (auto& r : res) {
      std::cout << r[0] << " " << r[1] << std::endl;
    }
  });
}

}  // namespace spu::mpc::test