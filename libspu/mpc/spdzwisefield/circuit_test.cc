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
    auto* prg_state = obj->getState<PrgState>();
    // auto* comm = obj->getState<Communicator>();

    std::vector<uint32_t> data(2);
    prg_state->fillPubl(absl::MakeSpan(data));

    ArrayRef array_data(makeType<Pub2kTy>(FM32), 2);
    auto _array_data = ArrayView<uint32_t>(array_data);

    _array_data[0] = data[0];
    _array_data[1] = data[1];

    ArrayRef share = obj->call("p2b", array_data);
    auto _share = ArrayView<std::array<uint32_t, 2>>(share);

    conversion::BitStream lhs;
    conversion::BitStream rhs;

    for (int i = 0; i < 32; i++) {
      std::array<bool, 2> dataA, dataB;
      dataA[0] = (_share[0][0] >> i) & 1;
      dataA[1] = (_share[0][1] >> i) & 1;

      dataB[0] = (_share[1][0] >> i) & 1;
      dataB[1] = (_share[1][1] >> i) & 1;

      lhs.push_back(dataA);
      rhs.push_back(dataB);
    }

    std::vector<conversion::BitStream> _lhs(1);
    std::vector<conversion::BitStream> _rhs(1);

    _lhs[0] = lhs;
    _rhs[0] = rhs;

    auto result = full_adder(obj.get(), _lhs, _rhs, false)[0];

    ArrayRef result_share(makeType<spdzwisefield::BShrTy>(PT_U64, 33), 1);
    auto _result_share = ArrayView<std::array<uint64_t, 2>>(result_share);

    _result_share[0][0] = _result_share[0][1] = 0;

    for (int i = 0; i < 33; i++) {
      _result_share[0][0] |= (static_cast<uint64_t>(result[i][0]) << i);
      _result_share[0][1] |= (static_cast<uint64_t>(result[i][1]) << i);
    }

    ArrayRef result_pub = obj->call("b2p", result_share);
    auto _result_pub = ArrayView<uint64_t>(result_pub)[0];

    SPU_ENFORCE(MersennePrimeField::add(data[0], data[1]) == _result_pub);
  });
}

}  // namespace spu::mpc::test