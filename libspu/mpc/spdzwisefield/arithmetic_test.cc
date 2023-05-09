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
    testing::Combine(testing::Values(1e6),             //
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

    const size_t test_size = std::get<0>(GetParam());

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

TEST_P(ArithmeticTest, AddAPTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();
    auto* spdzwisefield_state = obj->getState<SpdzWiseFieldState>();

    using Field = SpdzWiseFieldState::Field;

    const size_t test_size = std::get<0>(GetParam());

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

    ArrayRef mul_share = obj->call("add_ap", lhs_share, rhs);

    ArrayRef mul = obj->call("a2p", mul_share);
    auto _mul = ArrayView<uint64_t>(mul);

    pforeach(0, test_size, [&](uint64_t idx) {
      uint64_t expected = Field::add(a[idx], b[idx]);
      SPU_ENFORCE(_mul[idx] == expected,
                  "failed at index = {}, expect = {}, actual = {}", idx,
                  expected, _mul[idx]);
    });

    spdzwisefield_state->verification(obj.get(), true);
  });
}

TEST_P(ArithmeticTest, AddAATest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();
    auto* spdzwisefield_state = obj->getState<SpdzWiseFieldState>();

    using Field = SpdzWiseFieldState::Field;

    const size_t test_size = std::get<0>(GetParam());

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

    ArrayRef mul_share = obj->call("add_aa", lhs_share, rhs_share);

    ArrayRef mul = obj->call("a2p", mul_share);
    auto _mul = ArrayView<uint64_t>(mul);

    pforeach(0, test_size, [&](uint64_t idx) {
      uint64_t expected = Field::add(a[idx], b[idx]);
      SPU_ENFORCE(_mul[idx] == expected,
                  "failed at index = {}, expect = {}, actual = {}", idx,
                  expected, _mul[idx]);
    });

    spdzwisefield_state->verification(obj.get(), true);
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

    const size_t test_size = std::get<0>(GetParam());

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

    const size_t test_size = std::get<0>(GetParam());

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

TEST_P(ArithmeticTest, MatMulAATest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* spdzwisefield_state = obj->getState<SpdzWiseFieldState>();

    const uint64_t matrix_a[] = {
        1079583176452641360, 180754367733816053,  848745186230829153,
        351791535330704421,  1930673091373594868, 314989138156701656,
        984205072758443953,  1648192178479043852, 926059857974546788,
        146456306863591677,  348165077317371359,  78546175977447744,
        1183438504980212268, 392091420824419584,  1044255942638452475,
        154811681376816547,  1840420200535589133, 1953597950198923407,
        822249507322191589,  1101014171232592281, 1683551398161597054,
        1859705210407578946, 2244132132745264923, 297189170258959568,
        2061298896427897622};

    const uint64_t matrix_b[] = {
        2286516452113769160, 4910382206552007,    2160347790131638430,
        1151847004418116212, 1302852968549487020, 119258200989584541,
        372804635845517007,  1767412808076642825, 2226742194325163354,
        613943209425475330,  233280672834133487,  125255693428653781,
        455855903918796755,  481358482893818156,  1356598191107128210,
        612141070190163902,  1132785658965829542, 1651816859319630094,
        343244891895428772,  2241391972631427159, 668426678546011281,
        1147149813720298112, 1305746300549790101, 1531694638791264899,
        785103388556697382};

    const uint64_t result[] = {
        1362767983112357614, 402900770777594719,  2089853018883549943,
        2163690078770831662, 788007446502100414,  553785767000749724,
        1862376823171663923, 1088604069833081461, 1546073856572057317,
        995114313735986687,  1421190187809524114, 1864886267607672780,
        2237404812213236321, 2288281571230733450, 1987788482391661756,
        242956791223001763,  1622907505127743518, 431676021020792048,
        706254558412287083,  844998320185472057,  861779922845940051,
        2032620195108637900, 2274570691820489461, 1720093284130106776,
        916164667825289421};

    // const uint64_t matrix_a[] = {275761082052000538, 2086419664119139214,
    //                              1476717424976793514, 994428730426061134};
    // const uint64_t matrix_b[] = {443336659926363265, 1737423453127494916,
    //                              1901689795835461538, 2266008188322129936};
    // const uint64_t result[] = {56169408535563823, 2056063599658010211,
    //                            2245586495268885100, 729313073338085576};

    size_t n;
    size_t m;
    size_t k;
    n = m = k = 5;

    ArrayRef lhs(makeType<Pub2kTy>(FM64), n * k);
    ArrayRef rhs(makeType<Pub2kTy>(FM64), k * m);

    auto _lhs = ArrayView<uint64_t>(lhs);
    auto _rhs = ArrayView<uint64_t>(rhs);

    for (size_t i = 0; i < n * k; ++i) {
      _lhs[i] = matrix_a[i];
    }
    for (size_t i = 0; i < k * m; ++i) {
      _rhs[i] = matrix_b[i];
    }

    ArrayRef lhs_share = obj->call("p2a", lhs);
    ArrayRef rhs_share = obj->call("p2a", rhs);

    ArrayRef mul_share = obj->call("mmul_aa", lhs_share, rhs_share, n, m, k);

    ArrayRef mul = obj->call("a2p", mul_share);
    auto _mul = ArrayView<uint64_t>(mul);

    pforeach(0, n * m, [&](uint64_t idx) {
      uint64_t expected = result[idx];
      SPU_ENFORCE(_mul[idx] == expected,
                  "failed at index = {}, expect = {}, actual = {}", idx,
                  expected, _mul[idx]);
    });

    spdzwisefield_state->verification(obj.get(), true);
  });
}

TEST_P(ArithmeticTest, MatMulAPTest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* spdzwisefield_state = obj->getState<SpdzWiseFieldState>();

    const uint64_t matrix_a[] = {
        1079583176452641360, 180754367733816053,  848745186230829153,
        351791535330704421,  1930673091373594868, 314989138156701656,
        984205072758443953,  1648192178479043852, 926059857974546788,
        146456306863591677,  348165077317371359,  78546175977447744,
        1183438504980212268, 392091420824419584,  1044255942638452475,
        154811681376816547,  1840420200535589133, 1953597950198923407,
        822249507322191589,  1101014171232592281, 1683551398161597054,
        1859705210407578946, 2244132132745264923, 297189170258959568,
        2061298896427897622};

    const uint64_t matrix_b[] = {
        2286516452113769160, 4910382206552007,    2160347790131638430,
        1151847004418116212, 1302852968549487020, 119258200989584541,
        372804635845517007,  1767412808076642825, 2226742194325163354,
        613943209425475330,  233280672834133487,  125255693428653781,
        455855903918796755,  481358482893818156,  1356598191107128210,
        612141070190163902,  1132785658965829542, 1651816859319630094,
        343244891895428772,  2241391972631427159, 668426678546011281,
        1147149813720298112, 1305746300549790101, 1531694638791264899,
        785103388556697382};

    const uint64_t result[] = {
        1362767983112357614, 402900770777594719,  2089853018883549943,
        2163690078770831662, 788007446502100414,  553785767000749724,
        1862376823171663923, 1088604069833081461, 1546073856572057317,
        995114313735986687,  1421190187809524114, 1864886267607672780,
        2237404812213236321, 2288281571230733450, 1987788482391661756,
        242956791223001763,  1622907505127743518, 431676021020792048,
        706254558412287083,  844998320185472057,  861779922845940051,
        2032620195108637900, 2274570691820489461, 1720093284130106776,
        916164667825289421};

    // const uint64_t matrix_a[] = {275761082052000538, 2086419664119139214,
    //                              1476717424976793514, 994428730426061134};
    // const uint64_t matrix_b[] = {443336659926363265, 1737423453127494916,
    //                              1901689795835461538, 2266008188322129936};
    // const uint64_t result[] = {56169408535563823, 2056063599658010211,
    //                            2245586495268885100, 729313073338085576};

    size_t n;
    size_t m;
    size_t k;
    n = m = k = 5;

    ArrayRef lhs(makeType<Pub2kTy>(FM64), n * k);
    ArrayRef rhs(makeType<Pub2kTy>(FM64), k * m);

    auto _lhs = ArrayView<uint64_t>(lhs);
    auto _rhs = ArrayView<uint64_t>(rhs);

    for (size_t i = 0; i < n * k; ++i) {
      _lhs[i] = matrix_a[i];
    }
    for (size_t i = 0; i < k * m; ++i) {
      _rhs[i] = matrix_b[i];
    }

    ArrayRef lhs_share = obj->call("p2a", lhs);

    ArrayRef mul_share = obj->call("mmul_ap", lhs_share, rhs, n, m, k);

    ArrayRef mul = obj->call("a2p", mul_share);
    auto _mul = ArrayView<uint64_t>(mul);

    pforeach(0, n * m, [&](uint64_t idx) {
      uint64_t expected = result[idx];
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

    const size_t test_size = std::get<0>(GetParam());

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

TEST_P(ArithmeticTest, NotATest) {
  const auto factory = makeSpdzWiseFieldProtocol;
  const RuntimeConfig& conf = makeConfig(FieldType::FM64);
  const int npc = 3;

  utils::simulate(npc, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto obj = factory(conf, lctx);
    auto* prg_state = obj->getState<PrgState>();

    using T = uint64_t;

    const size_t test_size = std::get<0>(GetParam());

    std::vector<conversion::BitStream> _lhs(test_size);
    std::vector<conversion::BitStream> _rhs(test_size);

    std::vector<T> data(test_size);

    prg_state->fillPubl(absl::MakeSpan(data));

    pforeach(0, test_size, [&](uint64_t idx) {
      data[idx] = MersennePrimeField::modp(data[idx]);
    });

    ArrayRef in(makeType<Pub2kTy>(FM64), test_size);
    auto _in = ArrayView<uint64_t>(in);

    pforeach(0, test_size, [&](uint64_t idx) { _in[idx] = data[idx]; });

    ArrayRef in_share = obj->call("p2a", in);
    ArrayRef truncated = obj->call("not_a", in_share);
    ArrayRef out = obj->call("a2p", truncated);

    auto _out = ArrayView<uint64_t>(out);

    pforeach(0, test_size, [&](uint64_t idx) {
      SPU_ENFORCE(_out[idx] == MersennePrimeField::sub(
                                   MersennePrimeField::neg(data[idx]), 1),
                  "Failed");
    });
  });
}

}  // namespace spu::mpc::test