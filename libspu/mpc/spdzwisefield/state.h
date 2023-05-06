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

#pragma once

#include <complex>
#include <vector>

#include "yacl/crypto/base/hash/blake3.h"
#include "yacl/link/link.h"

#include "libspu/core/array_ref.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/object.h"
#include "libspu/mpc/spdzwisefield/utils.h"

namespace spu::mpc {

/*
 * Beaver binary Related
 */

namespace beaver {

using BTDataType = uint128_t;  // binary triple data type

using BinRss3PC = std::array<BTDataType, 2>;
using BinaryTriple = std::array<BinRss3PC, 3>;

enum CutAndChooseType {
  ARITH_TRIPLE = 1,
  BIN_TRIPLE = 2,
  DABITS = 3,
  EDABITS = 4
};

}  // namespace beaver

class BeaverState : public State {
  std::shared_ptr<yacl::link::Context> lctx_;
  std::unique_ptr<std::vector<beaver::BinaryTriple>> trusted_triples_bin_;

  const FieldType field_ = FM128;

 public:
  const static size_t batch_size_ = 10000;
  const static size_t bucket_size_ = 4;

  static constexpr char kBindName[] = "Beaver";

  explicit BeaverState(std::shared_ptr<yacl::link::Context> lctx) {
    lctx_ = lctx;
    trusted_triples_bin_ =
        std::make_unique<std::vector<beaver::BinaryTriple>>();
  }

  static size_t batch_size() { return batch_size_; }
  static size_t bucket_size() { return bucket_size_; }

  std::vector<beaver::BinaryTriple>* trusted_triples_bin() {
    return trusted_triples_bin_.get();
  }

  ArrayRef gen_bin_triples(Object* ctx, PtType out_type, size_t size,
                           size_t batch_size = BeaverState::batch_size_,
                           size_t bucket_size = BeaverState::bucket_size_);

  std::vector<beaver::BinaryTriple> cut_and_choose(
      Object* ctx, std::vector<beaver::BinaryTriple>::iterator data,
      const std::shared_ptr<yacl::crypto::Blake3Hash>& hash_algo,
      const std::shared_ptr<yacl::crypto::Blake3Hash>& hash_algo2,
      size_t batch_size, size_t bucket_size, size_t C);
};

/*
 * SpdzWise field Related
 */

namespace spdzwisefield {

using StorageType = uint64_t;
using Share = std::array<StorageType, 2>;

using TruncPair = std::array<Share, 2>;

}  // namespace spdzwisefield

class SpdzWiseFieldState : public State {
  std::shared_ptr<yacl::link::Context> lctx_;
  // share of global mac key
  spdzwisefield::Share key_;

  // triples to be verified
  std::unique_ptr<std::vector<ArrayRef>> stored_triples_;

  // statistical security parameter
  const size_t s_ = 64;

  // default in FM64
  FieldType field_;

 public:
  using Field = MersennePrimeField;

  static constexpr char kBindName[] = "SpdzWiseFieldState";

  explicit SpdzWiseFieldState(std::shared_ptr<yacl::link::Context> lctx,
                              spdzwisefield::Share key, FieldType field) {
    lctx_ = lctx;
    stored_triples_ = std::make_unique<std::vector<ArrayRef>>();
    key_ = key;

    // Now SpdzwiseField only supports mersenne prime field with p = 2^61-1, so
    // the storage type only supports uint64_t.
    assert(field == FM64);

    field_ = field;
  }

  spdzwisefield::Share key() const { return key_; }

  size_t s() const { return s_; }

  FieldType field() const { return field_; }

  std::vector<ArrayRef>* stored_triples() { return stored_triples_.get(); }

  std::vector<spdzwisefield::TruncPair> gen_trunc_pairs(Object* ctx,
                                                        size_t size,
                                                        size_t nbits);
};

/*
 * Share conversion Related
 * By using edabits
 */

namespace conversion {

using AShareType = uint64_t;
using BShareType = bool;
using ArithmeticShare = std::array<AShareType, 2>;
using BinaryShare = std::array<BShareType, 2>;

using BitStream = std::vector<BinaryShare>;

struct Edabit {
  ArithmeticShare ashare;
  std::array<BinaryShare, 61> bshares;
};

struct PubEdabit {
  AShareType dataA;
  std::array<BShareType, 61> dataB;

  void print(bool ignore_first = false, bool ignore_second = false) {
    if (!ignore_first) {
      std::cout << "dataA: " << dataA << std::endl;
    }

    if (!ignore_second) {
      std::cout << "dataB: ";
      for (auto rit = dataB.rbegin(); rit != dataB.rend(); ++rit) {
        std::cout << *rit;
      }
      std::cout << std::endl;
    }
  }
};

}  // namespace conversion

class EdabitState : public State {
  std::shared_ptr<yacl::link::Context> lctx_;
  std::unique_ptr<std::vector<conversion::Edabit>> trusted_edabits_;

 public:
  // statistical security parameter
  const size_t s_ = 40;

  const static size_t nbits_ = 61;

  /*
   * Reference: https://eprint.iacr.org/2020/338.pdf
   * Table 1: Number of edaBits produced by CutNChoose for statistical security
   * 2^{-s} and bucket size B, with C = C' = B.
   */
  const static size_t batch_size_ = 20000;
  const static size_t bucket_size_ = 4;

  static constexpr char kBindName[] = "EdabitState";
  explicit EdabitState(std::shared_ptr<yacl::link::Context> lctx) {
    lctx_ = lctx;
    trusted_edabits_ = std::make_unique<std::vector<conversion::Edabit>>();
  }

  size_t s() const { return s_; }

  size_t nbits() const { return nbits_; }

  std::vector<conversion::Edabit> gen_edabits(
      Object* ctx, PtType out_type, size_t size,
      size_t batch_size = EdabitState::batch_size_,
      size_t bucket_size = EdabitState::bucket_size_);

  std::vector<conversion::Edabit> cut_and_choose(
      Object* ctx, typename std::vector<conversion::Edabit>::iterator data,
      size_t batch_size, size_t bucket_size, size_t C);
};

std::vector<conversion::PubEdabit> open_edabits(
    Object* ctx, typename std::vector<conversion::Edabit>::iterator edabits,
    size_t n);

std::vector<conversion::BitStream> full_adder(
    Object* ctx, std::vector<conversion::BitStream> lhs,
    std::vector<conversion::BitStream> rhs, bool with_check, bool drop = false);

std::vector<conversion::BitStream> twos_complement(
    Object* ctx, std::vector<conversion::BitStream> bits, size_t nbits,
    bool with_check = true);

ArrayRef semi_honest_and_bb(Object* ctx, const ArrayRef& lhs,
                            const ArrayRef& rhs);

std::vector<bool> open_bits(Object* ctx, const conversion::BitStream& bts);
bool check_edabits(const std::vector<conversion::PubEdabit>& edabits);

template <typename T>
std::vector<T> open_semi_honest(Object* ctx, std::vector<std::array<T, 2>> in) {
  auto* comm = ctx->getState<Communicator>();
  using Field = SpdzWiseFieldState::Field;

  std::vector<T> buf(in.size());
  pforeach(0, in.size(), [&](int64_t idx) { buf[idx] = in[idx][1]; });

  std::vector<T> result = comm->rotate<T>(buf, "open_semi_honest");

  pforeach(0, in.size(), [&](int64_t idx) {
    result[idx] = Field::add(result[idx], in[idx][0], in[idx][1]);
  });

  return result;
}

std::vector<std::array<uint64_t, 2>> open_pair(
    Object* ctx, const std::vector<spdzwisefield::TruncPair>& pairs);

}  // namespace spu::mpc