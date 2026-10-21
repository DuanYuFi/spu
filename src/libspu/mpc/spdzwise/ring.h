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

#pragma once

#include <functional>

#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/spdzwise/ring_type.h"

namespace spu::mpc::spdzwise::ring {

// Wire words are exactly 16 bytes, little endian, independent of host layout.
std::string Encode(const std::vector<Word>& words);
std::vector<Word> Decode(std::string_view bytes, size_t count);

// context must bind session, execution and a fresh batch/message identity.
void CheckCopies(const std::shared_ptr<yacl::link::Context>& link,
                 const std::vector<RepShare>& values, std::string_view context);
void CheckZero(const std::shared_ptr<yacl::link::Context>& link,
               const RepShare& value, std::string_view context);

class ArithmeticState {
  std::shared_ptr<yacl::link::Context> link_;
  std::string session_;
  std::function<std::string(std::string_view)> tag_;
  std::unique_ptr<PrgState> key_prg_, mask_prg_, check_prg_;
  RepShare key_{};
  std::vector<AuthShare> pending_;

  void exchange(const std::vector<Word>& local, std::vector<Word>& next,
                std::string_view operation);
  std::vector<RepShare> reshare(std::vector<Word> local,
                                std::string_view operation);
  std::vector<RepShare> multiplyRaw(const std::vector<RepShare>& x,
                                    const std::vector<RepShare>& y);
  void remember(const NdArrayRef& value);

 public:
  ArithmeticState(std::shared_ptr<yacl::link::Context> link,
                  std::string session,
                  std::function<std::string(std::string_view)> tag);
  void validateInternal(const NdArrayRef& value) const;
  NdArrayRef importValue(const NdArrayRef& value);
  NdArrayRef publicValue(const NdArrayRef& value) const;
  NdArrayRef add(const NdArrayRef& x, const NdArrayRef& y) const;
  NdArrayRef negate(const NdArrayRef& x) const;
  NdArrayRef multiplyPublic(const NdArrayRef& x, const NdArrayRef& y) const;
  NdArrayRef multiply(const NdArrayRef& x, const NdArrayRef& y);
  NdArrayRef matmul(const NdArrayRef& x, const NdArrayRef& y, bool public_rhs);
  NdArrayRef exportValue(const NdArrayRef& value, const Type& type) const;
  bool check();
  bool empty() const { return pending_.empty(); }
};
}  // namespace spu::mpc::spdzwise::ring
