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

#include "libspu/mpc/spdzwise/ring_math.h"
#include "libspu/mpc/spdzwise/type.h"

namespace spu::mpc::spdzwise::ring {

// Reuse the external metadata grammar, but never the FIELD type identity.
class InputTy final : public TypeImpl<InputTy, TypeObject, Secret, AShare> {
  spdzwise::InputTy metadata_;

 public:
  InputTy() = default;
  explicit InputTy(size_t rank) : metadata_(rank) {}
  static std::string_view getStaticId() { return "spdzwise.ring.Input"; }
  size_t rank() const { return metadata_.rank(); }
  size_t size() const override { return sizeof(RepShare); }
  std::string toString() const override { return metadata_.toString(); }
  void fromString(std::string_view detail) override {
    metadata_.fromString(detail);
  }
  bool equals(const TypeObject* other) const override {
    const auto* typed = dynamic_cast<const InputTy*>(other);
    return typed && metadata_.equals(&typed->metadata_);
  }
};

// FM64 is the application width; all four internal components are 128 bits.
class AShrTy final
    : public TypeImpl<AShrTy, TypeObject, Ring2k, Secret, AShare> {
  std::string session_;

 public:
  AShrTy() { field_ = FM64; }
  explicit AShrTy(std::string session) : session_(std::move(session)) {
    SPU_ENFORCE(!session_.empty(), "SPDZWISE requires a session identity");
    field_ = FM64;
  }
  static std::string_view getStaticId() { return "spdzwise.ring.AShr"; }
  const std::string& session() const { return session_; }
  size_t size() const override { return sizeof(AuthShare); }
  bool isSerializable() const override { return false; }
  std::string toString() const override { return session_; }
  void fromString(std::string_view) override {
    SPU_THROW("SPDZWISE ring authentication cannot be imported from clients");
  }
  bool equals(const TypeObject* other) const override {
    const auto* typed = dynamic_cast<const AShrTy*>(other);
    return typed && session_ == typed->session_;
  }
};

// Only the low 64 bits of each value component leave the checked session.
class OutputTy final : public TypeImpl<OutputTy, TypeObject, Secret, AShare> {
  spdzwise::OutputTy metadata_;

 public:
  OutputTy() = default;
  OutputTy(size_t rank, std::string binding)
      : metadata_(rank, std::move(binding)) {}
  static std::string_view getStaticId() { return "spdzwise.ring.Output"; }
  size_t rank() const { return metadata_.rank(); }
  const std::string& binding() const { return metadata_.binding(); }
  size_t size() const override { return sizeof(spdzwise::RepShare); }
  std::string toString() const override { return metadata_.toString(); }
  void fromString(std::string_view detail) override {
    metadata_.fromString(detail);
  }
  bool equals(const TypeObject* other) const override {
    const auto* typed = dynamic_cast<const OutputTy*>(other);
    return typed && metadata_.equals(&typed->metadata_);
  }
};

}  // namespace spu::mpc::spdzwise::ring
