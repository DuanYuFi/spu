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

#include <array>
#include <cstdint>
#include <string>

#include "libspu/core/type.h"
#include "libspu/mpc/spdzwise/field.h"

namespace spu::mpc::spdzwise {

using RepShare = std::array<uint64_t, 2>;
// x_i, x_{i+1}, gamma_i, gamma_{i+1}; one element is moved atomically by
// standard shape kernels.
using AuthShare = std::array<uint64_t, 4>;

// This is an external, UNAUTHENTICATED representation. No client-supplied flag
// can turn it into AShrTy. rank is part of the wire type to detect permutation.
class InputTy final : public TypeImpl<InputTy, TypeObject, Secret, AShare> {
  size_t rank_ = 0;

 public:
  InputTy() = default;
  explicit InputTy(size_t rank);
  static std::string_view getStaticId() { return "spdzwise.Input"; }
  size_t rank() const { return rank_; }
  size_t size() const override { return sizeof(RepShare); }
  std::string toString() const override;
  void fromString(std::string_view detail) override;
  bool equals(const TypeObject* other) const override;
};

// Arithmetic shares carry a local session identity, never a globally known MAC
// key. Their serialization is deliberately forbidden: exports must strip MACs.
class AShrTy final : public TypeImpl<AShrTy, TypeObject, Gfp, Secret, AShare> {
  std::string session_;

 public:
  AShrTy() { prime_ = kPrime; }
  explicit AShrTy(std::string session);
  static std::string_view getStaticId() { return "spdzwise.AShr"; }
  const std::string& session() const { return session_; }
  size_t size() const override { return sizeof(AuthShare); }
  bool isSerializable() const override { return false; }
  std::string toString() const override;
  void fromString(std::string_view detail) override;
  bool equals(const TypeObject* other) const override;
};

// The binding is a digest of session/execution/output name/shape/dtype. It is
// not a signature: authenticity comes from the three authenticated channels and
// matching replicated copies supplied by the honest servers.
class OutputTy final : public TypeImpl<OutputTy, TypeObject, Secret, AShare> {
  size_t rank_ = 0;
  std::string binding_;

 public:
  OutputTy() = default;
  OutputTy(size_t rank, std::string binding);
  static std::string_view getStaticId() { return "spdzwise.Output"; }
  size_t rank() const { return rank_; }
  const std::string& binding() const { return binding_; }
  size_t size() const override { return sizeof(RepShare); }
  std::string toString() const override;
  void fromString(std::string_view detail) override;
  bool equals(const TypeObject* other) const override;
};

// Register external types only. Internal authentication cannot be deserialized.
void registerTypes();

}  // namespace spu::mpc::spdzwise
