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
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/object.h"

namespace spu::mpc {

class SpdzWiseFieldState : public State {
  // aby3 for semi-honest protocol
  std::unique_ptr<Object> honest_;

  std::shared_ptr<yacl::link::Context> lctx_;
  // share of global mac key
  uint128_t key_;

  // triples to be verified
  std::unique_ptr<std::vector<ArrayRef>> stored_triples_;

  // plaintext ring size
  const size_t k_ = 64;

  // statistical security parameter
  const size_t s_ = 64;

  // default in FM128
  const FieldType field_ = FM128;

 public:
  static constexpr char kBindName[] = "SpdzWiseFieldState";
  explicit SpdzWiseFieldState(RuntimeConfig conf,
                              std::shared_ptr<yacl::link::Context> lctx) {
    lctx_ = lctx;
    stored_triples_ = std::make_unique<std::vector<ArrayRef>>();
    honest_ = makeAby3Protocol(conf, lctx_);
  }

  uint128_t key() const { return key_; }

  size_t k() const { return k_; }

  size_t s() const { return s_; }

  std::vector<ArrayRef>* stored_triples() { return stored_triples_.get(); }
};
}  // namespace spu::mpc