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
#include "libspu/mpc/spdzwisefield/protocol.h"
#include "libspu/mpc/spdzwisefield/type.h"
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
class BooleanTest : public ::testing::TestWithParam<CACOpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Beaver, BooleanTest,
    testing::Combine(testing::Values(1),               //
                     testing::Values(PtType::PT_U32),  //
                     testing::Values(3)),              //
    [](const testing::TestParamInfo<BooleanTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(BooleanTest, AndTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    auto* comm = obj->getState<Communicator>();

    ArrayRef a(makeType<spu::mpc::spdzwisefield::BShrTy>(PT_U64, 64), 1);
    ArrayRef b(makeType<spu::mpc::spdzwisefield::BShrTy>(PT_U64, 64), 1);

    auto _a = ArrayView<std::array<uint64_t, 2>>(a);
    auto _b = ArrayView<std::array<uint64_t, 2>>(b);

    switch (comm->getRank()) {
      case 0: {
        _a[0] = {1701777869756606743ULL, 12882733032270490486ULL};
        _b[0] = {972803246877995573ULL, 2002847137855454938ULL};
        break;
      }
      case 1: {
        _a[0] = {12882733032270490486ULL, 14690618097154390971ULL};
        _b[0] = {2002847137855454938ULL, 2701453227443784868ULL};
        break;
      }
      case 2: {
        _a[0] = {14690618097154390971ULL, 1701777869756606743ULL};
        _b[0] = {2701453227443784868ULL, 972803246877995573ULL};
        break;
      }

      default:
        break;
    }

    ArrayRef res = obj->call("and_bb", a, b);
    auto _res = ArrayView<std::array<uint64_t, 2>>(res);

    sleep(comm->getRank());
    std::cout << "rank " << comm->getRank() << std::endl;
    std::cout << "res[0][0] = " << _res[0][0] << std::endl;
    std::cout << "res[0][1] = " << _res[0][1] << std::endl;
  });
}

}  // namespace spu::mpc::test