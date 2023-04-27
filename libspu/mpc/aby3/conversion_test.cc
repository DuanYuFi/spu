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

#include "libspu/mpc/aby3/conversion.h"

#include "gtest/gtest.h"

#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
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

using ConvTestParams = std::tuple<int, PtType, size_t>;
class ConversionTest : public ::testing::TestWithParam<ConvTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Aby3, ConversionTest,
    testing::Combine(testing::Values(1),               //
                     testing::Values(PtType::PT_U32),  //
                     testing::Values(3)),              //
    [](const testing::TestParamInfo<ConversionTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(ConversionTest, B2ATest) {
  const auto factory = makeAby3Protocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();
    auto* comm = obj->getState<Communicator>();

    std::vector<uint8_t> public_bits(10000);

    prg_state->fillPubl(absl::MakeSpan(public_bits));

    ArrayRef binaries(makeType<Pub2kTy>(FM64), 80000);
    auto _binaries = ArrayView<uint64_t>(binaries);

    for (size_t i = 0; i < 10000; i++) {
      for (size_t j = 0; j < 8; j++) {
        _binaries[i * 8 + j] = (public_bits[i] >> j) & 1;
      }
    }

    if (comm->getRank() == 0) {
      for (int i = 0; i < 5; i++) {
        std::cout << _binaries[i] << ", ";
      }
      std::cout << std::endl;
    }

    ArrayRef bshares = obj->call("p2b", binaries);
    ArrayRef ashares = obj->call("b2a", bshares);
    ArrayRef number = obj->call("a2p", ashares);

    // (void)number;

    if (comm->getRank() == 0) {
      auto _number = ArrayView<uint64_t>(number);
      for (int i = 0; i < 80000; i++) {
        SPU_ENFORCE(_number[i] == _binaries[i], "i = {}, {} != {}", i,
                    _number[i], _binaries[i]);
      }
      std::cout << std::endl;
    }
  });
}

}  // namespace spu::mpc::test