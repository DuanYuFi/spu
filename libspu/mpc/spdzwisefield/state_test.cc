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

#include "libspu/mpc/spdzwisefield/state.h"

#include "gtest/gtest.h"

#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/spdzwisefield/protocol.h"
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
class StateTest : public ::testing::TestWithParam<CACOpTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    Beaver, StateTest,
    testing::Combine(testing::Values(1000),            //
                     testing::Values(PtType::PT_U64),  //
                     testing::Values(3)),              //
    [](const testing::TestParamInfo<StateTest::ParamType>& p) {
      return fmt::format("{}x{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(StateTest, BinaryCAC) {
  using spu::mpc::beaver::BinaryTriple;
  using spu::mpc::beaver::BTDataType;

  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  const int batch_size = 20000;
  const int bucket_size = 4;

  PtType type = std::get<1>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    auto* state = obj->getState<BeaverState>();
    auto* comm = obj->getState<Communicator>();

    const int test_size = 1e3;

    DISPATCH_UINT_PT_TYPES(type, "_", [&]() {
      using T = ScalarT;

      auto before = comm->getStats();

      auto ret = state->gen_bin_triples(obj.get(), type, test_size, batch_size,
                                        bucket_size);

      auto cost = comm->getStats() - before;

      if (comm->getRank() == 0) {
        std::cout << "cost: " << cost.comm / (double)(1024 * 1024) << std::endl;
        std::cout << "Left: " << state->trusted_triples_bin()->size()
                  << std::endl;
      }

      auto _ret = ArrayView<std::array<std::array<T, 2>, 3>>(ret);
      std::vector<T> send_buffer(test_size * 3);

      for (int i = 0; i < test_size; i++) {
        send_buffer[i * 3] = _ret[i][0][1];
        send_buffer[i * 3 + 1] = _ret[i][1][1];
        send_buffer[i * 3 + 2] = _ret[i][2][1];
      }

      auto recv_buffer = comm->rotate<T>(send_buffer, "b2p");

      for (int i = 0; i < test_size; i++) {
        auto a = (recv_buffer[i * 3] ^ _ret[i][0][0] ^ _ret[i][0][1]);
        auto b = (recv_buffer[i * 3 + 1] ^ _ret[i][1][0] ^ _ret[i][1][1]);
        auto c = (recv_buffer[i * 3 + 2] ^ _ret[i][2][0] ^ _ret[i][2][1]);

        SPU_ENFORCE((a & b) == c);
      }
    });
  });
}

TEST_P(StateTest, TruncPairTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  const int test_size = std::get<0>(GetParam());

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = factory(conf, lctx);

    auto* spdzwisefield_state = ctx->getState<SpdzWiseFieldState>();

    auto pairs = spdzwisefield_state->gen_trunc_pairs(ctx.get(), test_size, 20);

    ArrayRef r(makeType<aby3::AShrTy>(FM64), test_size);
    ArrayRef r_prime(makeType<aby3::AShrTy>(FM64), test_size);

    auto _r = ArrayView<std::array<uint64_t, 2>>(r);
    auto _r_prime = ArrayView<std::array<uint64_t, 2>>(r_prime);

    pforeach(0, test_size, [&](uint64_t idx) {
      _r[idx] = pairs[idx][0];
      _r_prime[idx] = pairs[idx][1];
    });

    ArrayRef open_r = ctx->call("a2psh", r);
    ArrayRef open_r_prime = ctx->call("a2psh", r_prime);

    auto _open_r = ArrayView<uint64_t>(open_r);
    auto _open_r_prime = ArrayView<uint64_t>(open_r_prime);

    pforeach(0, test_size, [&](uint64_t idx) {
      SPU_ENFORCE(_open_r[idx] == _open_r_prime[idx] >> 20, "Truncation error");
    });
  });
}

}  // namespace spu::mpc::test