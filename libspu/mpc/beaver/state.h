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

#include "yacl/crypto/utils/rand.h"
#include "yacl/link/link.h"

#include "libspu/core/array_ref.h"
// #include "libspu/mpc/beaver/utils.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/object.h"

namespace spu::mpc {

namespace beaver {

using ring2k_t = uint64_t;
using ScalarT = uint64_t;

typedef std::array<ring2k_t, 2> Rss3PC;
typedef std::array<Rss3PC, 3> ArithmeticTriple;

typedef std::array<ScalarT, 2> BinRss3PC;
typedef std::array<BinRss3PC, 3> BinaryTriple;

enum CutAndChooseType {
  ARITH_TRIPLE = 1,
  BIN_TRIPLE = 2,
  DABITS = 3,
  EDABITS = 4
};

}  // namespace beaver

class BeaverState : public State {
  std::shared_ptr<yacl::link::Context> lctx_;
  std::unique_ptr<std::vector<beaver::ArithmeticTriple>> trusted_triples_arith_;
  std::unique_ptr<std::vector<beaver::BinaryTriple>> trusted_triples_bin_;

  const FieldType field_ = FM128;

 public:
  const static size_t batch_size_ = 64000;
  const static size_t bucket_size_ = 4;

  static constexpr char kBindName[] = "Beaver";

  explicit BeaverState(std::shared_ptr<yacl::link::Context> lctx) {
    ;
    lctx_ = lctx;
    trusted_triples_arith_ =
        std::make_unique<std::vector<beaver::ArithmeticTriple>>();
    trusted_triples_bin_ =
        std::make_unique<std::vector<beaver::BinaryTriple>>();
  }

  size_t batch_size() const { return batch_size_; }
  size_t bucket_size() const { return bucket_size_; }

  std::vector<beaver::ArithmeticTriple>* trusted_triples_arith() {
    return trusted_triples_arith_.get();
  }

  std::vector<beaver::BinaryTriple>* trusted_triples_bin() {
    return trusted_triples_bin_.get();
  }

  std::vector<beaver::BinaryTriple> get_bin_triples(Object* ctx,
                                                    size_t num_triples) {
    if (trusted_triples_bin_->size() < num_triples) {
      gen_bin_triples(ctx, trusted_triples_bin_.get(), num_triples, batch_size_,
                      bucket_size_);
    }

    std::vector<beaver::BinaryTriple> ret(num_triples);
    std::copy_n(trusted_triples_bin_->begin(), num_triples, ret.begin());

    trusted_triples_bin_->erase(trusted_triples_bin_->begin(),
                                trusted_triples_bin_->begin() + num_triples);

    return ret;
  }

  void gen_arith_triples(Object* ctx,
                         std::vector<beaver::ArithmeticTriple>* triples,
                         size_t num_triples,
                         size_t batch_size = BeaverState::batch_size_,
                         size_t bucket_size = BeaverState::bucket_size_);

  void gen_bin_triples(Object* ctx, std::vector<beaver::BinaryTriple>* triples,
                       size_t num_triples,
                       size_t batch_size = BeaverState::batch_size_,
                       size_t bucket_size = BeaverState::bucket_size_);

  std::vector<beaver::ArithmeticTriple> cut_and_choose(
      Object* ctx,
      typename std::vector<beaver::ArithmeticTriple>::iterator data,
      size_t batch_size, size_t bucket_size, size_t C);

  std::vector<beaver::BinaryTriple> cut_and_choose(
      Object* ctx, typename std::vector<beaver::BinaryTriple>::iterator data,
      size_t batch_size, size_t bucket_size, size_t C);
};

}  // namespace spu::mpc