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

#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/protocol_session.h"
#include "libspu/mpc/spdzwise/boolean.h"
#include "libspu/mpc/spdzwise/type.h"

namespace spu::mpc::spdzwise {

// Checked session for the current F61 arithmetic and shared ABFL prototype.
// F61 arithmetic kernels live in arithmetic.h/.cc, Boolean kernels in
// boolean.h/.cc. This state rejects RING configurations; ring arithmetic must
// provide its own state/checking implementation. The production factory stays
// closed until mixed protocols, source review and integration tests complete.
class SpdzWiseState final : public ProtocolSession {
  // The benchmark separates MAC generation from the final batched check.
  // Normal imports retain their mandatory immediate check.
  friend class SpdzWiseBenchmarkAccess;
  friend class BooleanTestAccess;
  std::unique_ptr<BooleanState> boolean_;
  std::shared_ptr<yacl::link::Context> link_;
  RuntimeConfig config_;
  std::string session_;
  RepShare key_{};
  // Independently seeded per-party streams. Key, multiplication masks and
  // secret checking coefficients never consume each other's PRG material.
  std::unique_ptr<PrgState> key_prg_, mask_prg_, check_prg_;
  uint64_t message_ = 0;
  uint64_t execution_ = 0;
  uint64_t arithmetic_checks_ = 0;
  std::vector<AuthShare> pending_;

  std::string tag(std::string_view operation);
  std::vector<uint64_t> exchange(const std::vector<uint64_t>& local,
                                 std::string_view operation);
  std::vector<RepShare> randomRep(size_t size);
  std::vector<RepShare> multiplyRaw(const std::vector<RepShare>& x,
                                    const std::vector<RepShare>& y);
  void checkCopies(const std::vector<RepShare>& x);
  uint64_t openRaw(const RepShare& value);
  void agree(const std::string& descriptor);
  NdArrayRef authenticateInput(const std::vector<RepShare>& values,
                               const Shape& shape);

 protected:
  void onBegin(const ExecutableProto& exec,
               const std::vector<Value>& inputs) override;
  Type exportType(const std::string& name, const Value& value) const override;
  void onAbort() noexcept override;

 public:
  explicit SpdzWiseState(const RuntimeConfig& config,
                         std::shared_ptr<yacl::link::Context> link);
  std::unique_ptr<State> fork() override;
  void validateExternalInput(const Value& value) const override;
  const std::string& session() const { return session_; }
  size_t rank() const { return link_->Rank(); }
  uint64_t arithmeticChecks() const { return arithmetic_checks_; }

  void validateInternal(const NdArrayRef& value) const;
  NdArrayRef importValue(const NdArrayRef& value);
  NdArrayRef publicValue(const NdArrayRef& value) const;
  NdArrayRef add(const NdArrayRef& x, const NdArrayRef& y) const;
  NdArrayRef negate(const NdArrayRef& x) const;
  NdArrayRef multiplyPublic(const NdArrayRef& x, const NdArrayRef& y) const;
  NdArrayRef multiply(const NdArrayRef& x, const NdArrayRef& y);
  NdArrayRef matmul(const NdArrayRef& x, const NdArrayRef& y, bool public_rhs);
  NdArrayRef exportValue(const NdArrayRef& value, const Type& type) const;
  void check();
  void finalize();
  void validateBoolean(const NdArrayRef& value) const;
  NdArrayRef publicBoolean(const NdArrayRef& value) const;
  NdArrayRef booleanBinary(const NdArrayRef& x, const NdArrayRef& y,
                           bool is_and, bool public_rhs);
  size_t pendingBooleanBits() const {
    return boolean_ ? boolean_->pendingBits() : 0;
  }
  uint64_t booleanChecks() const { return boolean_ ? boolean_->checks() : 0; }
};

}  // namespace spu::mpc::spdzwise
