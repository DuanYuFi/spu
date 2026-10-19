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

#include "libspu/mpc/spdzwise/type.h"

#include <algorithm>
#include <mutex>
#include <utility>

#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::spdzwise {

InputTy::InputTy(size_t rank) : rank_(rank) {
  SPU_ENFORCE(rank < 3, "SPDZWISE requires rank in [0,3)");
}

std::string InputTy::toString() const { return std::to_string(rank_); }

void InputTy::fromString(std::string_view detail) {
  SPU_ENFORCE(detail.size() == 1 && detail[0] >= '0' && detail[0] <= '2',
              "invalid SPDZWISE input rank");
  rank_ = detail[0] - '0';
}

bool InputTy::equals(const TypeObject* other) const {
  const auto* typed = dynamic_cast<const InputTy*>(other);
  return typed != nullptr && rank_ == typed->rank_;
}

AShrTy::AShrTy(std::string session) : session_(std::move(session)) {
  SPU_ENFORCE(!session_.empty(), "SPDZWISE requires a session identity");
  prime_ = kPrime;
}

std::string AShrTy::toString() const { return session_; }

void AShrTy::fromString(std::string_view) {
  SPU_THROW("SPDZWISE internal authentication cannot be imported from clients");
}

bool AShrTy::equals(const TypeObject* other) const {
  const auto* typed = dynamic_cast<const AShrTy*>(other);
  return typed != nullptr && session_ == typed->session_;
}

OutputTy::OutputTy(size_t rank, std::string binding) {
  fromString(std::to_string(rank) + ":" + binding);
}

std::string OutputTy::toString() const {
  return std::to_string(rank_) + ":" + binding_;
}

void OutputTy::fromString(std::string_view detail) {
  SPU_ENFORCE(detail.size() == 66 && detail[0] >= '0' && detail[0] <= '2' &&
                  detail[1] == ':',
              "invalid SPDZWISE output binding");
  SPU_ENFORCE(std::all_of(detail.begin() + 2, detail.end(),
                          [](char c) {
                            return (c >= '0' && c <= '9') ||
                                   (c >= 'a' && c <= 'f');
                          }),
              "invalid SPDZWISE output digest");
  rank_ = detail[0] - '0';
  binding_ = detail.substr(2);
}

bool OutputTy::equals(const TypeObject* other) const {
  const auto* typed = dynamic_cast<const OutputTy*>(other);
  return typed != nullptr && rank_ == typed->rank_ &&
         binding_ == typed->binding_;
}

void registerTypes() {
  regPV2kTypes();
  static std::once_flag flag;
  std::call_once(flag, [] {
    TypeContext::getTypeContext()->addTypes<InputTy, OutputTy>();
  });
}

}  // namespace spu::mpc::spdzwise
