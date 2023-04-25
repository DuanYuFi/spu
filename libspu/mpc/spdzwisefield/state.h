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
      Object* ctx, typename std::vector<beaver::BinaryTriple>::iterator data,
      size_t batch_size, size_t bucket_size, size_t C);
};

/*
 * SpdzWise field Related
 */

namespace spdzwisefield {

using StorageType = uint64_t;
using Share = std::array<StorageType, 2>;

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
};

/*
 * Share conversion Related
 * By using edabits
 */

namespace conversion {

using AShareType = uint64_t;
using BShareType = bool;
using ArithmeticShare = std::array<uint64_t, 2>;
using BinaryShare = std::array<AShareType, 2>;

struct Edabit {
  ArithmeticShare ashare;
  std::array<BinaryShare, 61> bshares;
};

}  // namespace conversion

class EdabitState : public State {
  std::shared_ptr<yacl::link::Context> lctx_;
  std::unique_ptr<std::vector<conversion::Edabit>> trusted_edabits_;

 public:
  // statistical security parameter
  const size_t s_ = 40;

  const size_t nbits_ = 61;

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

  ArrayRef gen_edabits(Object* ctx, PtType out_type, size_t size,
                       size_t batch_size = EdabitState::batch_size_,
                       size_t bucket_size = EdabitState::bucket_size_);

  std::vector<conversion::Edabit> cut_and_choose(
      Object* ctx, typename std::vector<conversion::Edabit>::iterator data,
      size_t batch_size, size_t bucket_size, size_t C);
};

}  // namespace spu::mpc