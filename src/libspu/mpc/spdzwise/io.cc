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

#include <limits>

#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/type.h"

namespace spu::mpc::spdzwise {
namespace {

uint64_t RandomResidue() {
  uint64_t result;
  do {
    result = static_cast<uint64_t>(yacl::crypto::SecureRandSeed()) & kPrime;
  } while (result == kPrime);
  return result;
}

}  // namespace

SpdzWiseIo::SpdzWiseIo(FieldType field, size_t parties)
    : BaseIo(field, parties) {
  SPU_ENFORCE(field == FM64, "SPDZWISE v1 requires FM64 / p=2^61-1");
  SPU_ENFORCE(parties == 3, "SPDZWISE requires exactly three parties");
  registerTypes();
}

Type SpdzWiseIo::getShareType(Visibility vis, int owner_rank) const {
  SPU_ENFORCE(owner_rank == -1, "SPDZWISE v1 does not support colocated input");
  if (vis == VIS_PUBLIC) {
    return makeType<Pub2kTy>(FM64);
  }
  SPU_ENFORCE(vis == VIS_SECRET, "unsupported SPDZWISE visibility");
  // Used for sizing; toShares sets the actual destination rank.
  return makeType<InputTy>(0);
}

std::vector<NdArrayRef> SpdzWiseIo::toShares(const NdArrayRef& raw,
                                             Visibility vis,
                                             int owner_rank) const {
  const auto type = getShareType(vis, owner_rank);
  SPU_ENFORCE(
      raw.eltype().isa<RingTy>() && raw.eltype().as<Ring2k>()->field() == FM64,
      "SPDZWISE input must be an FM64 ring-encoded buffer");
  if (vis == VIS_PUBLIC) {
    return std::vector<NdArrayRef>(3, raw.as(type));
  }
  std::vector<NdArrayRef> shares;
  for (size_t rank = 0; rank < 3; ++rank) {
    shares.emplace_back(makeType<InputTy>(rank), raw.shape());
  }
  const NdArrayView<uint64_t> input(raw);
  std::array<NdArrayView<RepShare>, 3> out = {NdArrayView<RepShare>(shares[0]),
                                              NdArrayView<RepShare>(shares[1]),
                                              NdArrayView<RepShare>(shares[2])};
  for (int64_t i = 0; i < raw.numel(); ++i) {
    SPU_ENFORCE(IsBoundedRing(input[i]),
                "SPDZWISE input exceeds the centered F61 range");
    const uint64_t a = RandomResidue();
    const uint64_t b = RandomResidue();
    const uint64_t c = Sub(Sub(FromRing(input[i]), a), b);
    out[0][i] = {a, b};
    out[1][i] = {b, c};
    out[2][i] = {c, a};
  }
  return shares;
}

std::vector<NdArrayRef> SpdzWiseIo::toBooleanShares(const NdArrayRef& raw,
                                                    size_t nbits) const {
  const auto mask = BooleanMask(nbits);
  SPU_ENFORCE(
      raw.eltype().isa<RingTy>() && raw.eltype().as<Ring2k>()->field() == FM64,
      "Boolean input requires FM64 storage");
  std::vector<NdArrayRef> shares;
  for (size_t rank = 0; rank < 3; ++rank) {
    shares.emplace_back(makeType<BInputTy>(rank, nbits), raw.shape());
  }
  const NdArrayView<uint64_t> input(raw);
  for (int64_t i = 0; i < raw.numel(); ++i) {
    SPU_ENFORCE((input[i] & ~mask) == 0, "Boolean input exceeds its bit width");
    const auto a = uint64_t(yacl::crypto::SecureRandSeed()) & mask;
    const auto b = uint64_t(yacl::crypto::SecureRandSeed()) & mask;
    const auto c = input[i] ^ a ^ b;
    shares[0].data<RepShare>()[i] = {a, b};
    shares[1].data<RepShare>()[i] = {b, c};
    shares[2].data<RepShare>()[i] = {c, a};
  }
  return shares;
}
size_t SpdzWiseIo::getBitSecretShareSize(size_t numel) const {
  SPU_ENFORCE(numel <= std::numeric_limits<size_t>::max() / sizeof(RepShare),
              "Boolean share size overflow");
  return numel * sizeof(RepShare);
}
std::vector<NdArrayRef> SpdzWiseIo::makeBitSecret(
    const PtBufferView& raw) const {
  NdArrayRef words(makeType<RingTy>(FM64), raw.shape);
  auto* out = words.data<uint64_t>();
  if (raw.isBitSet()) {
    for (int64_t i = 0; i < words.numel(); ++i) {
      out[i] = raw.getBit(i);
    }
  } else {
    DISPATCH_INT_PT_TYPES(raw.pt_type, [&]() {
      for (int64_t i = 0; i < words.numel(); ++i) {
        out[i] = static_cast<uint64_t>(raw.get<ScalarT>(i)) & 1;
      }
    });
  }
  return toBooleanShares(words, 1);
}

NdArrayRef SpdzWiseIo::fromShares(const std::vector<NdArrayRef>& shares) const {
  SPU_ENFORCE(shares.size() == 3,
              "SPDZWISE reconstruction requires all parties");
  const auto& first = shares[0];
  if (first.eltype().isa<BInputTy>() || first.eltype().isa<BOutputTy>()) {
    const bool input = first.eltype().isa<BInputTy>();
    const auto width = first.eltype().as<BShare>()->nbits();
    const auto mask = BooleanMask(width);
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& type = shares[rank].eltype();
      SPU_ENFORCE(shares[rank].shape() == first.shape(),
                  "Boolean shape mismatch");
      if (input) {
        SPU_ENFORCE(type.isa<BInputTy>() && type.as<BInputTy>()->rank() == rank,
                    "Boolean input rank/type mismatch");
      } else {
        SPU_ENFORCE(type.isa<BOutputTy>() &&
                        type.as<BOutputTy>()->rank() == rank &&
                        type.as<BOutputTy>()->binding() ==
                            first.eltype().as<BOutputTy>()->binding(),
                    "Boolean output binding/rank/type mismatch");
      }
      SPU_ENFORCE(type.as<BShare>()->nbits() == width,
                  "Boolean width mismatch");
    }
    NdArrayRef result(makeType<RingTy>(FM64), first.shape());
    const std::array<NdArrayView<RepShare>, 3> in = {
        NdArrayView<RepShare>(shares[0]), NdArrayView<RepShare>(shares[1]),
        NdArrayView<RepShare>(shares[2])};
    auto* out = result.data<uint64_t>();
    for (int64_t i = 0; i < first.numel(); ++i) {
      out[i] = 0;
      for (size_t rank = 0; rank < 3; ++rank) {
        SPU_ENFORCE(((in[rank][i][0] | in[rank][i][1]) & ~mask) == 0,
                    "noncanonical Boolean share");
        SPU_ENFORCE(in[rank][i][1] == in[(rank + 1) % 3][i][0],
                    "Boolean copies disagree");
        out[i] ^= in[rank][i][0];
      }
    }
    return result;
  }
  const bool is_public = first.eltype().isa<Pub2kTy>();
  const bool is_input = first.eltype().isa<InputTy>();
  SPU_ENFORCE(is_public || is_input || first.eltype().isa<OutputTy>(),
              "SPDZWISE cannot reconstruct internal MAC shares");
  for (size_t rank = 0; rank < 3; ++rank) {
    SPU_ENFORCE(shares[rank].shape() == first.shape(), "share shape mismatch");
    const auto& type = shares[rank].eltype();
    if (is_public) {
      SPU_ENFORCE(type == makeType<Pub2kTy>(FM64),
                  "public share type mismatch");
    } else if (is_input) {
      SPU_ENFORCE(type.isa<InputTy>() && type.as<InputTy>()->rank() == rank,
                  "input share rank/type mismatch");
    } else {
      SPU_ENFORCE(type.isa<OutputTy>() && type.as<OutputTy>()->rank() == rank &&
                      type.as<OutputTy>()->binding() ==
                          first.eltype().as<OutputTy>()->binding(),
                  "output share rank or execution binding mismatch");
    }
  }
  NdArrayRef result(makeType<RingTy>(FM64), first.shape());
  NdArrayView<uint64_t> out(result);
  if (is_public) {
    const NdArrayView<uint64_t> a(shares[0]), b(shares[1]), c(shares[2]);
    for (int64_t i = 0; i < first.numel(); ++i) {
      SPU_ENFORCE(a[i] == b[i] && a[i] == c[i], "public inputs disagree");
      out[i] = a[i];
    }
  } else {
    const std::array<NdArrayView<RepShare>, 3> in = {
        NdArrayView<RepShare>(shares[0]), NdArrayView<RepShare>(shares[1]),
        NdArrayView<RepShare>(shares[2])};
    for (int64_t i = 0; i < first.numel(); ++i) {
      for (size_t rank = 0; rank < 3; ++rank) {
        SPU_ENFORCE(in[rank][i][0] < kPrime && in[rank][i][1] < kPrime,
                    "noncanonical SPDZWISE share");
        SPU_ENFORCE(in[rank][i][1] == in[(rank + 1) % 3][i][0],
                    "SPDZWISE replicated copies disagree");
      }
      out[i] = ToRing(Add(Add(in[0][i][0], in[1][i][0]), in[2][i][0]));
    }
  }
  return result;
}

std::unique_ptr<SpdzWiseIo> makeSpdzWiseIo(FieldType field, size_t parties) {
  return std::make_unique<SpdzWiseIo>(field, parties);
}

}  // namespace spu::mpc::spdzwise
