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
    // auto* prg_state = obj->getState<PrgState>();
    auto* comm = obj->getState<Communicator>();

    (void)comm;

    using T = uint64_t;
    const size_t bit_length = 61;
    const uint128_t mod = 1ULL << bit_length;

    const size_t test_times = 3;

    std::vector<conversion::BitStream> _lhs(test_times);
    std::vector<conversion::BitStream> _rhs(test_times);

    std::vector<T> data(2);
    // prg_state->fillPubl(absl::MakeSpan(data));
    data[0] = 1577476313703811090;
    data[1] = 1839456394423608633;

    data[0] %= mod;
    data[1] %= mod;

    for (size_t _ = 0; _ < test_times; _++) {
      ArrayRef array_data(makeType<Pub2kTy>(FM64), 2);
      auto _array_data = ArrayView<T>(array_data);

      _array_data[0] = data[0];
      _array_data[1] = data[1];

      ArrayRef share = obj->call("p2b", array_data);
      auto _share = ArrayView<std::array<T, 2>>(share);

      conversion::BitStream lhs;
      conversion::BitStream rhs;

      for (size_t i = 0; i < bit_length; i++) {
        std::array<bool, 2> dataA;
        std::array<bool, 2> dataB;
        dataA[0] = (_share[0][0] >> i) & 1;
        dataA[1] = (_share[0][1] >> i) & 1;

        dataB[0] = (_share[1][0] >> i) & 1;
        dataB[1] = (_share[1][1] >> i) & 1;

        lhs.push_back(dataA);
        rhs.push_back(dataB);
      }

      _lhs[_] = lhs;
      _rhs[_] = rhs;
    }

    auto result = full_adder(obj.get(), _lhs, _rhs, false);

    ArrayRef result_share(
        makeType<spdzwisefield::BShrTy>(PT_U64, bit_length + 1), test_times);
    auto _result_share = ArrayView<std::array<uint64_t, 2>>(result_share);

    for (size_t _ = 0; _ < test_times; _++) {
      _result_share[_][0] = _result_share[_][1] = 0;

      for (size_t i = 0; i < bit_length + 1; i++) {
        _result_share[_][0] |= (static_cast<uint64_t>(result[_][i][0]) << i);
        _result_share[_][1] |= (static_cast<uint64_t>(result[_][i][1]) << i);
      }
    }

    ArrayRef result_pub = obj->call("b2p", result_share);
    auto _result_pub = ArrayView<uint64_t>(result_pub);

    pforeach(0, test_times, [&](uint64_t idx) {
      SPU_ENFORCE(data[0] + data[1] == _result_pub[idx],
                  "{} + {} != {} where idx = {}", data[0], data[1],
                  _result_pub[idx], idx);
    });
  });
}

TEST_P(CircuitTest, TwosComplement) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    // auto* prg_state = obj->getState<PrgState>();
    auto* comm = obj->getState<Communicator>();

    (void)comm;

    using T = uint64_t;
    const size_t bit_length = 61;
    const uint128_t mod = 1ULL << bit_length;

    const size_t test_times = 3;

    std::vector<conversion::BitStream> _lhs(test_times);
    std::vector<conversion::BitStream> _rhs(test_times);

    std::vector<T> data(2);
    // prg_state->fillPubl(absl::MakeSpan(data));
    data[1] = 1839456394423608633;
    data[0] = 1577476313703811090;

    data[0] %= mod;
    data[1] %= mod;

    for (size_t _ = 0; _ < test_times; _++) {
      ArrayRef array_data(makeType<Pub2kTy>(FM64), 2);
      auto _array_data = ArrayView<T>(array_data);

      _array_data[0] = data[0];
      _array_data[1] = data[1];

      ArrayRef share = obj->call("p2b", array_data);
      auto _share = ArrayView<std::array<T, 2>>(share);

      conversion::BitStream lhs;
      conversion::BitStream rhs;

      for (size_t i = 0; i < bit_length; i++) {
        std::array<bool, 2> dataA;
        std::array<bool, 2> dataB;
        dataA[0] = (_share[0][0] >> i) & 1;
        dataA[1] = (_share[0][1] >> i) & 1;

        dataB[0] = (_share[1][0] >> i) & 1;
        dataB[1] = (_share[1][1] >> i) & 1;

        lhs.push_back(dataA);
        rhs.push_back(dataB);
      }

      _lhs[_] = lhs;
      _rhs[_] = rhs;
    }

    auto neg_rhs = twos_complement(obj.get(), _rhs, 64);

    auto opened = open_bits(obj.get(), neg_rhs[0]);
    if (comm->getRank() == 0) {
      for (auto it = opened.rbegin(); it != opened.rend(); it++) {
        std::cout << *it;
      }
      std::cout << std::endl;
    }

    auto result = full_adder(obj.get(), _lhs, neg_rhs, false);

    ArrayRef result_share(
        makeType<spdzwisefield::BShrTy>(PT_U64, bit_length + 1), test_times);
    auto _result_share = ArrayView<std::array<uint64_t, 2>>(result_share);

    for (size_t _ = 0; _ < test_times; _++) {
      _result_share[_][0] = _result_share[_][1] = 0;

      for (size_t i = 0; i < 64; i++) {
        _result_share[_][0] |= (static_cast<uint64_t>(result[_][i][0]) << i);
        _result_share[_][1] |= (static_cast<uint64_t>(result[_][i][1]) << i);
      }
    }

    ArrayRef result_pub = obj->call("b2p", result_share);
    auto _result_pub = ArrayView<uint64_t>(result_pub);

    pforeach(0, test_times, [&](uint64_t idx) {
      SPU_ENFORCE(data[0] - data[1] == _result_pub[idx],
                  "{} - {} != {} where idx = {}", data[0], data[1],
                  _result_pub[idx], idx);
    });
  });
}

}  // namespace spu::mpc::test