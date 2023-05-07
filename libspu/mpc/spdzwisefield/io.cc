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

#include "libspu/mpc/spdzwisefield/io.h"

#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/utils/rand.h"

#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/spdzwisefield/utils.h"
#include "libspu/mpc/spdzwisefield/value.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdzwisefield {

std::vector<ArrayRef> SpdzWiseFieldIo::toShares(const ArrayRef& raw,
                                                Visibility vis,
                                                int owner_rank) const {
  SPU_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
              raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
              field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<ArrayRef>(world_size_, share);
  } else if (vis == VIS_SECRET) {
    // by default, make as arithmetic share.

    auto _raw = ArrayView<uint64_t>(raw);
    std::cout << "raw[0]: " << _raw[0] << std::endl;

    auto splits =
        MersennePrimeField::field_rand_additive_splits(raw, world_size_);

    std::vector<ArrayView<uint64_t>> _splits;

    for (size_t i = 0; i < world_size_; i++) {
      _splits.emplace_back(splits[i]);
    }

    std::cout << "raw[0][0] = " << _splits[0][0]
              << ", raw[0][1] = " << _splits[1][0]
              << ", raw[0][2] = " << _splits[2][0] << std::endl;

    std::vector<ArrayRef> shares;
    for (std::size_t i = 0; i < 3; i++) {
      ArrayRef share(makeType<AShrTy>(FM64), raw.numel());
      auto _share = ArrayView<std::array<uint64_t, 4>>(share);

      pforeach(0, raw.numel(), [&](uint64_t idx) {
        _share[idx][0] = _splits[i][idx];
        _share[idx][1] = _splits[(i + 1) % world_size_][idx];
        _share[idx][2] = 0;
        _share[idx][3] = 0;
      });

      shares.push_back(share);
    }
    return shares;
  }

  SPU_THROW("unsupported vis type {}", vis);
}

std::vector<ArrayRef> SpdzWiseFieldIo::makeBitSecret(const ArrayRef& in) const {
  SPU_ENFORCE(in.eltype().isa<PtTy>(), "expected PtType, got {}", in.eltype());
  PtType in_pt_type = in.eltype().as<PtTy>()->pt_type();
  SPU_ENFORCE(in_pt_type == PT_BOOL);

  if (in_pt_type == PT_BOOL) {
    // we assume boolean is stored with byte array.
    in_pt_type = PT_U8;
  }

  const auto out_type = makeType<BShrTy>(PT_U8, /* out_nbits */ 1);
  const size_t numel = in.numel();

  std::vector<ArrayRef> shares{
      {out_type, numel}, {out_type, numel}, {out_type, numel}};
  return DISPATCH_UINT_PT_TYPES(in_pt_type, "_", [&]() {
    using InT = ScalarT;
    using BShrT = uint8_t;

    auto _in = ArrayView<InT>(in);

    std::vector<BShrT> r0(numel);
    std::vector<BShrT> r1(numel);

    yacl::crypto::PrgAesCtr(yacl::crypto::RandSeed(), absl::MakeSpan(r0));
    yacl::crypto::PrgAesCtr(yacl::crypto::RandSeed(), absl::MakeSpan(r1));

    auto _s0 = ArrayView<std::array<BShrT, 2>>(shares[0]);
    auto _s1 = ArrayView<std::array<BShrT, 2>>(shares[1]);
    auto _s2 = ArrayView<std::array<BShrT, 2>>(shares[2]);

    for (int64_t idx = 0; idx < in.numel(); idx++) {
      const BShrT r2 = static_cast<BShrT>(_in[idx]) - r0[idx] - r1[idx];

      _s0[idx][0] = r0[idx] & 0x1;
      _s0[idx][1] = r1[idx] & 0x1;

      _s1[idx][0] = r1[idx] & 0x1;
      _s1[idx][1] = r2 & 0x1;

      _s2[idx][0] = r2 & 0x1;
      _s2[idx][1] = r0[idx] & 0x1;
    }
    return shares;
  });
}

ArrayRef SpdzWiseFieldIo::fromShares(
    const std::vector<ArrayRef>& shares) const {
  const auto& eltype = shares.at(0).eltype();

  if (eltype.isa<Pub2kTy>()) {
    SPU_ENFORCE(field_ == eltype.as<Ring2k>()->field());
    return shares[0].as(makeType<RingTy>(field_));
  } else if (eltype.isa<AShrTy>()) {
    ArrayRef out(makeType<Pub2kTy>(FM64), shares[0].numel());
    auto _out = ArrayView<uint64_t>(out);
    for (size_t si = 0; si < shares.size(); si++) {
      auto _share = ArrayView<std::array<uint64_t, 4>>(shares[si]);
      for (auto idx = 0; idx < shares[0].numel(); idx++) {
        if (si == 0) {
          _out[idx] = 0;
        }
        _out[idx] = MersennePrimeField::add(_out[idx], _share[idx][0]);
      }
    }
    return out;
  } else if (eltype.isa<BShrTy>()) {
    ArrayRef out(makeType<Pub2kTy>(field_), shares[0].numel());

    DISPATCH_ALL_FIELDS(field_, "_", [&]() {
      using OutT = ring2k_t;
      auto _out = ArrayView<OutT>(out);
      DISPATCH_UINT_PT_TYPES(eltype.as<BShrTy>()->getBacktype(), "_", [&] {
        using BShrT = ScalarT;
        for (size_t si = 0; si < shares.size(); si++) {
          auto _share = ArrayView<std::array<BShrT, 2>>(shares[si]);
          for (auto idx = 0; idx < shares[0].numel(); idx++) {
            if (si == 0) {
              _out[idx] = 0;
            }
            _out[idx] ^= _share[idx][0];
          }
        }
      });
    });

    return out;
  }
  SPU_THROW("unsupported eltype {}", eltype);
}

std::unique_ptr<SpdzWiseFieldIo> makeSpdzWiseFieldIo(FieldType field,
                                                     size_t npc) {
  SPU_ENFORCE(npc == 3U, "spdzwisefield now only supports for 3pc.");
  registerTypes();
  return std::make_unique<SpdzWiseFieldIo>(field, npc);
}

}  // namespace spu::mpc::spdzwisefield
