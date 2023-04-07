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

#include "libspu/mpc/beaver/state.h"

#include "gtest/gtest.h"

#include "libspu/mpc/beaver/protocol.h"
#include "libspu/mpc/common/communicator.h"
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

TEST(CutAndChooseTest, BinaryCAC) {
  using spu::mpc::beaver::BinaryTriple;
  using spu::mpc::beaver::BTDataType;

  const auto factory = makeBeaverProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  const int batch_size = 5;
  const int bucket_size = 5;

  const int test_size = 5;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);

    auto* state = obj->getState<BeaverState>();
    auto* comm = obj->getState<Communicator>();

    PtType type = PT_U32;
    auto ret = state->gen_bin_triples(obj.get(), type, test_size, batch_size,
                                      bucket_size);
    auto _ret = ArrayView<std::array<std::array<uint32_t, 2>, 3>>(ret);
    std::vector<uint32_t> send_buffer(test_size * 3);

    for (int i = 0; i < test_size; i++) {
      send_buffer[i * 3] = _ret[i][0][1];
      send_buffer[i * 3 + 1] = _ret[i][1][1];
      send_buffer[i * 3 + 2] = _ret[i][2][1];
    }

    auto recv_buffer = comm->rotate<uint32_t>(send_buffer, "b2p");

    for (int i = 0; i < test_size; i++) {
      auto a = (recv_buffer[i * 3] ^ _ret[i][0][0] ^ _ret[i][0][1]);
      auto b = (recv_buffer[i * 3 + 1] ^ _ret[i][1][0] ^ _ret[i][1][1]);
      auto c = (recv_buffer[i * 3 + 2] ^ _ret[i][2][0] ^ _ret[i][2][1]);

      EXPECT_EQ(a & b, c);
    }
  });
}

}  // namespace spu::mpc::test