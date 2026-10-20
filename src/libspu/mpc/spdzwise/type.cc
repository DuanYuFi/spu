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
#include <charconv>
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

uint64_t BooleanMask(size_t nbits) {
  SPU_ENFORCE(nbits >= 1 && nbits <= 64, "Boolean width must be in [1,64]");
  return nbits == 64 ? ~uint64_t{0} : (uint64_t{1} << nbits) - 1;
}

namespace {
size_t ParseBooleanNumber(std::string_view text) {
  size_t result = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  SPU_ENFORCE(!text.empty() && parsed.ec == std::errc{} &&
                  parsed.ptr == text.data() + text.size(),
              "invalid Boolean type");
  return result;
}
}  // namespace

BInputTy::BInputTy(size_t rank, size_t nbits) : rank_(rank) {
  SPU_ENFORCE(rank < 3, "invalid Boolean rank");
  BooleanMask(nbits);
  nbits_ = nbits;
}
std::string BInputTy::toString() const {
  return std::to_string(rank_) + ":" + std::to_string(nbits_);
}
void BInputTy::fromString(std::string_view detail) {
  const auto split = detail.find(':');
  SPU_ENFORCE(split != std::string_view::npos, "invalid Boolean input type");
  *this = BInputTy(ParseBooleanNumber(detail.substr(0, split)),
                   ParseBooleanNumber(detail.substr(split + 1)));
}
bool BInputTy::equals(const TypeObject* other) const {
  const auto* t = dynamic_cast<const BInputTy*>(other);
  return t && rank_ == t->rank_ && nbits_ == t->nbits_;
}
BShrTy::BShrTy(std::string session, size_t nbits)
    : session_(std::move(session)) {
  SPU_ENFORCE(!session_.empty(), "Boolean shares require a session");
  BooleanMask(nbits);
  nbits_ = nbits;
}
std::string BShrTy::toString() const {
  return session_ + ":" + std::to_string(nbits_);
}
void BShrTy::fromString(std::string_view) {
  SPU_THROW("internal Boolean shares cannot be deserialized");
}
bool BShrTy::equals(const TypeObject* other) const {
  const auto* t = dynamic_cast<const BShrTy*>(other);
  return t && session_ == t->session_ && nbits_ == t->nbits_;
}
BOutputTy::BOutputTy(size_t rank, size_t nbits, std::string binding)
    : rank_(rank), binding_(std::move(binding)) {
  // Reuse the existing rank/digest syntax validation.
  OutputTy validated(rank_, binding_);
  BooleanMask(nbits);
  nbits_ = nbits;
}
std::string BOutputTy::toString() const {
  return std::to_string(rank_) + ":" + std::to_string(nbits_) + ":" + binding_;
}
void BOutputTy::fromString(std::string_view detail) {
  const auto first = detail.find(':');
  const auto second =
      detail.find(':', first == std::string_view::npos ? 0 : first + 1);
  SPU_ENFORCE(
      first != std::string_view::npos && second != std::string_view::npos,
      "invalid Boolean output type");
  *this = BOutputTy(
      ParseBooleanNumber(detail.substr(0, first)),
      ParseBooleanNumber(detail.substr(first + 1, second - first - 1)),
      std::string(detail.substr(second + 1)));
}
bool BOutputTy::equals(const TypeObject* other) const {
  const auto* t = dynamic_cast<const BOutputTy*>(other);
  return t && rank_ == t->rank_ && nbits_ == t->nbits_ &&
         binding_ == t->binding_;
}

void registerTypes() {
  regPV2kTypes();
  static std::once_flag flag;
  std::call_once(flag, [] {
    TypeContext::getTypeContext()
        ->addTypes<InputTy, OutputTy, BInputTy, BOutputTy>();
  });
}

}  // namespace spu::mpc::spdzwise
