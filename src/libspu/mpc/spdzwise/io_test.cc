// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/spdzwise/io.h"

#include "gtest/gtest.h"

#include "libspu/core/value.h"
#include "libspu/mpc/spdzwise/type.h"

namespace spu::mpc::spdzwise {
namespace {

NdArrayRef Input() {
  NdArrayRef x(makeType<RingTy>(FM64), {2, 3});
  NdArrayView<uint64_t> v(x);
  const uint64_t values[] = {
      0, 1, uint64_t{0} - 1, kHalfPrime, uint64_t{0} - kHalfPrime, 123456};
  for (int64_t i = 0; i < x.numel(); ++i) {
    v[i] = values[i];
  }
  return x;
}

TEST(SpdzWiseIo, CenteredRoundTrip) {
  SpdzWiseIo io(FM64, 3);
  const auto input = Input();
  auto shares = io.toShares(input, VIS_SECRET);
  const auto result = io.fromShares(shares);
  const NdArrayView<uint64_t> a(input), b(result);
  for (int64_t i = 0; i < input.numel(); ++i) {
    EXPECT_EQ(a[i], b[i]);
  }
}

TEST(SpdzWiseIo, EveryReplicatedCopyIsChecked) {
  SpdzWiseIo io(FM64, 3);
  for (size_t rank = 0; rank < 3; ++rank) {
    for (size_t part = 0; part < 2; ++part) {
      auto shares = io.toShares(Input(), VIS_SECRET);
      NdArrayView<RepShare> v(shares[rank]);
      v[0][part] = Add(v[0][part], 1);
      EXPECT_ANY_THROW(io.fromShares(shares));
    }
  }
}

TEST(SpdzWiseIo, RejectsMetadataAndNoncanonicalShares) {
  SpdzWiseIo io(FM64, 3);
  auto shares = io.toShares(Input(), VIS_SECRET);
  std::swap(shares[0], shares[1]);
  EXPECT_ANY_THROW(io.fromShares(shares));
  shares = io.toShares(Input(), VIS_SECRET);
  NdArrayView<RepShare>{shares[0]}[0][0] = kPrime;
  NdArrayView<RepShare>{shares[2]}[0][1] = kPrime;
  EXPECT_ANY_THROW(io.fromShares(shares));
  shares.pop_back();
  EXPECT_ANY_THROW(io.fromShares(shares));
}

TEST(SpdzWiseIo, BindsOutputsToExecution) {
  SpdzWiseIo io(FM64, 3);
  auto shares = io.toShares(Input(), VIS_SECRET);
  for (size_t rank = 0; rank < 3; ++rank) {
    shares[rank] =
        shares[rank].as(makeType<OutputTy>(rank, std::string(64, 'a')));
  }
  EXPECT_NO_THROW(io.fromShares(shares));
  shares[1] = shares[1].as(makeType<OutputTy>(1, std::string(64, 'b')));
  EXPECT_ANY_THROW(io.fromShares(shares));
}

TEST(SpdzWiseIo, RejectsUnsupportedConfigurationsAndRanges) {
  EXPECT_ANY_THROW(SpdzWiseIo(FM32, 3));
  EXPECT_ANY_THROW(SpdzWiseIo(FM64, 2));
  SpdzWiseIo io(FM64, 3);
  EXPECT_ANY_THROW(io.toShares(Input(), VIS_SECRET, 0));
  auto x = Input();
  NdArrayView<uint64_t>{x}[0] = kHalfPrime + 1;
  EXPECT_ANY_THROW(io.toShares(x, VIS_SECRET));
  NdArrayView<uint64_t>{x}[0] = uint64_t{0} - kHalfPrime - 1;
  EXPECT_ANY_THROW(io.toShares(x, VIS_SECRET));
}

TEST(SpdzWiseIo, ExternalTypesRoundTripButAuthenticationCannotBeForged) {
  registerTypes();
  for (size_t rank = 0; rank < 3; ++rank) {
    const auto input = makeType<InputTy>(rank);
    EXPECT_EQ(Type::fromString(input.toString()), input);
    const auto output = makeType<OutputTy>(rank, std::string(64, 'c'));
    EXPECT_EQ(Type::fromString(output.toString()), output);
  }
  EXPECT_ANY_THROW(Type::fromString("spdzwise.Input<0junk>"));
  EXPECT_ANY_THROW(Type::fromString("spdzwise.Input<3>"));
  EXPECT_ANY_THROW(Type::fromString("spdzwise.AShr<forged>"));
  const auto internal = makeType<AShrTy>("session");
  EXPECT_FALSE(internal.isSerializable());
  EXPECT_NO_THROW(internal.toString());
  Value value(NdArrayRef(internal, {1}), DT_I64);
  EXPECT_ANY_THROW(value.toMetaProto());
  EXPECT_ANY_THROW(value.toProto(1024));
}

TEST(SpdzWiseIo, EmptyAndNoncontiguousBuffers) {
  SpdzWiseIo io(FM64, 3);
  const NdArrayRef empty(makeType<RingTy>(FM64), {0, 3});
  EXPECT_EQ(io.fromShares(io.toShares(empty, VIS_SECRET)).shape(),
            empty.shape());
  const auto input = Input().transpose({1, 0});
  const auto output = io.fromShares(io.toShares(input, VIS_SECRET));
  const NdArrayView<uint64_t> a(input), b(output);
  for (int64_t i = 0; i < input.numel(); ++i) {
    EXPECT_EQ(a[i], b[i]);
  }
}

}  // namespace
}  // namespace spu::mpc::spdzwise
