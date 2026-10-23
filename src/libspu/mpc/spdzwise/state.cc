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

#include "libspu/mpc/spdzwise/state.h"

#include <cstring>
#include <limits>

#include "yacl/crypto/hash/blake3.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/link/algorithm/allgather.h"

#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/replicated.h"

namespace spu::mpc::spdzwise {
namespace {

std::string Digest(std::string_view bytes) {
  yacl::crypto::Blake3Hash hash;
  hash.Update(bytes);
  const auto digest = hash.CumulativeHash();
  std::string result;
  constexpr char hex[] = "0123456789abcdef";
  for (auto byte : digest) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}

void Append(std::string& out, std::string_view value) {
  out += std::to_string(value.size()) + ":";
  out.append(value);
}

RepShare Values(const AuthShare& x) { return {x[0], x[1]}; }

void RequirePublic(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype() == makeType<Pub2kTy>(FM64),
              "expected an FM64 public value");
}

}  // namespace

SpdzWiseState::SpdzWiseState(const RuntimeConfig& config,
                             std::shared_ptr<yacl::link::Context> link)
    : link_(std::move(link)), config_(config) {
  if (!config_.spdzwise_config.check_threshold) {
    config_.spdzwise_config.check_threshold =
        SpdzWiseConfig::kDefaultCheckThreshold;
  }
  SPU_ENFORCE(link_ && link_->WorldSize() == 3,
              "SPDZWISE requires three parties");
  SPU_ENFORCE(config.field == FM64, "SPDZWISE requires FM64");
  SPU_ENFORCE(config.spdzwise_config.domain == SpdzWiseConfig::FIELD ||
                  config.spdzwise_config.domain == SpdzWiseConfig::RING,
              "invalid SPDZWISE arithmetic domain");
  SPU_ENFORCE(!config.experimental_enable_inter_op_par &&
                  !config.experimental_enable_intra_op_par,
              "SPDZWISE context parallelism is not supported");
  SPU_ENFORCE(!config.experimental_enable_colocated_optimization &&
                  !config.enable_runtime_snapshot,
              "SPDZWISE colocated input and runtime snapshots are unsupported");
}

std::unique_ptr<State> SpdzWiseState::fork() {
  SPU_THROW("SPDZWISE cannot fork key, preprocessing or deferred-check state");
}

std::string SpdzWiseState::tag(std::string_view operation) {
  requireRunning();
  SPU_ENFORCE(message_ != std::numeric_limits<uint64_t>::max(),
              "message overflow");
  return fmt::format("spdzwise/{}/{}/{}", execution_, message_++, operation);
}

std::vector<uint64_t> SpdzWiseState::exchange(
    const std::vector<uint64_t>& local, std::string_view operation) {
  return ExchangeF61(link_, local, tag(operation));
}

std::vector<RepShare> SpdzWiseState::randomRep(size_t size) {
  SPU_ENFORCE(check_prg_, "checking PRSS is not initialized");
  return RandomF61Rep(check_prg_.get(), size);
}

void SpdzWiseState::checkCopies(const std::vector<RepShare>& x) {
  std::vector<uint64_t> first(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    SPU_ENFORCE(x[i][0] < kPrime && x[i][1] < kPrime, "noncanonical share");
    first[i] = x[i][0];
  }
  // Only send a component to the party that already holds its duplicate.
  const auto next = exchange(first, "copy-consistency");
  for (size_t i = 0; i < x.size(); ++i) {
    SPU_ENFORCE(next[i] == x[i][1],
                "SPDZWISE replicated input copies disagree");
  }
}

void SpdzWiseState::agree(const std::string& descriptor) {
  const auto digest = Digest(descriptor);
  const auto all = yacl::link::AllGather(link_, yacl::ByteContainerView(digest),
                                         tag("agreement"));
  for (const auto& item : all) {
    SPU_ENFORCE(std::string_view(item) == digest, "SPDZWISE parties disagree");
  }
}

NdArrayRef SpdzWiseState::randomPublic(const Shape& shape) {
  try {
    requireRunning();
    NdArrayRef out(makeType<Pub2kTy>(FM64), shape);
    if (!out.numel()) return out;
    // Separate public coins from all private masks and checking randomness.
    const auto draw =
        session_ + "/" + tag("public-random") + fmt::format("/{}", shape);
    const auto nonce = yacl::crypto::SecureRandSeed();
    const std::string bytes(reinterpret_cast<const char*>(&nonce), sizeof(nonce));
    const auto commitment = Digest(draw + bytes);
    const auto commitments = yacl::link::AllGather(
        link_, yacl::ByteContainerView(commitment), tag("public-random-commit"));
    std::string descriptor = draw;
    for (const auto& item : commitments) Append(descriptor, std::string_view(item));
    agree(descriptor);
    const auto nonces = yacl::link::AllGather(
        link_, yacl::ByteContainerView(bytes), tag("public-random-reveal"));
    uint128_t seed = 0;
    bool valid = true;
    for (size_t rank = 0; rank < nonces.size(); ++rank) {
      valid &= nonces[rank].size() == sizeof(nonce) &&
               Digest(draw + std::string(std::string_view(nonces[rank]))) ==
                   std::string_view(commitments[rank]);
      if (nonces[rank].size() == sizeof(nonce)) {
        uint128_t contribution;
        std::memcpy(&contribution, nonces[rank].data(), sizeof(contribution));
        seed ^= contribution;
      }
      Append(descriptor, std::string_view(nonces[rank]));
    }
    Append(descriptor, std::to_string(valid));
    agree(descriptor);
    SPU_ENFORCE(valid, "SPDZWISE public random commitment mismatch");
    yacl::crypto::FillPRand(
        PrgState::kAesType, seed, 0, 0,
        absl::MakeSpan(out.data<uint64_t>(), size_t(out.numel())));
    return out;
  } catch (...) {
    abort();
    throw;
  }
}

void SpdzWiseState::onBegin(const ExecutableProto& exec,
                            const std::vector<Value>& inputs) {
  SPU_ENFORCE(execution_ != std::numeric_limits<uint64_t>::max(),
              "execution overflow");
  ++execution_;
  SPU_ENFORCE(inputs.size() == exec.input_names.size(), "input count mismatch");
  std::string descriptor;
  Append(descriptor, exec.SerializeAsString());
  auto agreed_config = config_;
  // Simulators commonly profile only rank zero. These flags affect local
  // logging, not protocol messages or numerical semantics.
  agreed_config.enable_action_trace = false;
  agreed_config.enable_hal_profile = false;
  agreed_config.enable_pphlo_profile = false;
  agreed_config.enable_pphlo_trace = false;
  Append(descriptor, agreed_config.SerializeAsString());
  for (const auto& value : inputs) {
    SPU_ENFORCE(!value.isComplex() && !value.isPrivate(),
                "unsupported input kind");
    Append(descriptor, std::to_string(value.dtype()));
    Append(descriptor, fmt::format("{}", value.shape()));
    Append(descriptor, std::to_string(value.isPublic()));
    if (value.isPublic()) {
      RequirePublic(value.data());
      const NdArrayView<uint64_t> view(value.data());
      std::vector<uint64_t> words(value.numel());
      for (int64_t i = 0; i < value.numel(); ++i) {
        words[i] = view[i];
      }
      Append(descriptor, EncodeF61(words));
    } else if (value.storage_type().isa<BShrTy>()) {
      validateBoolean(value.data());
      Append(descriptor, value.storage_type().toString());
    } else if (value.storage_type().isa<ring::AShrTy>()) {
      validateInternal(value.data());
      Append(descriptor, "ring-authenticated");
      Append(descriptor, value.storage_type().as<ring::AShrTy>()->session());
    } else if (value.storage_type().isa<AShrTy>()) {
      validateInternal(value.data());
      Append(descriptor, "authenticated");
      Append(descriptor, value.storage_type().as<AShrTy>()->session());
    } else {
      validateExternalInput(value);
      // Rank is deliberately excluded: it differs across the three parties.
      // The kind and prior output binding must agree before any authentication
      // messages, otherwise parties can import different execution batches.
      if (value.storage_type().isa<BInputTy>()) {
        Append(descriptor, "boolean-input");
        Append(descriptor,
               std::to_string(value.storage_type().as<BInputTy>()->nbits()));
      } else if (value.storage_type().isa<BOutputTy>()) {
        Append(descriptor, "boolean-output");
        Append(descriptor,
               std::to_string(value.storage_type().as<BOutputTy>()->nbits()));
        Append(descriptor, value.storage_type().as<BOutputTy>()->binding());
      } else if (value.storage_type().isa<ring::OutputTy>()) {
        Append(descriptor, "ring-external-output");
        Append(descriptor,
               value.storage_type().as<ring::OutputTy>()->binding());
      } else if (value.storage_type().isa<OutputTy>()) {
        Append(descriptor, "external-output");
        Append(descriptor, value.storage_type().as<OutputTy>()->binding());
      } else {
        Append(descriptor, "external-input");
      }
    }
  }
  agree(descriptor);
  if (session_.empty()) {
    const auto nonce = yacl::crypto::SecureRandSeed();
    const auto nonces =
        yacl::link::AllGather(link_, {&nonce, sizeof(nonce)}, tag("session"));
    std::string combined;
    for (const auto& item : nonces) {
      SPU_ENFORCE(item.size() == sizeof(nonce), "invalid session nonce");
      Append(combined, std::string_view(item));
    }
    session_ = Digest(combined);
    agree(session_);
    if (isRing()) {
      ring_ = std::make_unique<ring::ArithmeticState>(
          link_, session_, [this](std::string_view operation) {
            return session_ + "/" + tag(operation);
          });
    } else {
      key_prg_ = std::make_unique<PrgState>(link_);
      mask_prg_ = std::make_unique<PrgState>(link_);
      check_prg_ = std::make_unique<PrgState>(link_);
    }
    boolean_ = std::make_unique<BooleanState>(
        link_, [this](std::string_view operation) {
          return session_ + "/" + tag(operation);
        });
    if (!isRing()) {
      key_ = RandomF61Rep(key_prg_.get(), 1)[0];
      checkCopies({key_});
    }
  }
}

Type SpdzWiseState::exportType(const std::string& name,
                               const Value& value) const {
  std::string descriptor;
  Append(descriptor, session_);
  Append(descriptor, std::to_string(execution_));
  Append(descriptor, name);
  Append(descriptor, fmt::format("{}", value.shape()));
  Append(descriptor, std::to_string(value.dtype()));
  if (value.storage_type().isa<BShrTy>()) {
    validateBoolean(value.data());
    const auto width = value.storage_type().as<BShrTy>()->nbits();
    Append(descriptor, "boolean");
    Append(descriptor, std::to_string(width));
    return makeType<BOutputTy>(rank(), width, Digest(descriptor));
  }
  if (isRing()) {
    Append(descriptor, "ring");
    return makeType<ring::OutputTy>(rank(), Digest(descriptor));
  }
  return makeType<OutputTy>(rank(), Digest(descriptor));
}

void SpdzWiseState::onAbort() noexcept {
  pending_.clear();
  boolean_.reset();
  ring_.reset();
  key_ = {};
  key_prg_.reset();
  mask_prg_.reset();
  check_prg_.reset();
}

void SpdzWiseState::validateExternalInput(const Value& value) const {
  requireHealthy();
  SPU_ENFORCE(!value.isComplex() && !value.isPrivate(),
              "unsupported input kind");
  const auto& type = value.storage_type();
  if (value.isPublic()) {
    RequirePublic(value.data());
  } else if (type.isa<BInputTy>() || type.isa<BOutputTy>()) {
    SPU_ENFORCE(type.isa<BInputTy>() ? type.as<BInputTy>()->rank() == rank()
                                     : type.as<BOutputTy>()->rank() == rank(),
                "Boolean external input rank mismatch");
    const auto mask = BooleanMask(type.as<BShare>()->nbits());
    const NdArrayView<RepShare> words(value.data());
    for (int64_t i = 0; i < value.numel(); ++i) {
      SPU_ENFORCE(((words[i][0] | words[i][1]) & ~mask) == 0,
                  "noncanonical Boolean input");
    }
  } else if (isRing()) {
    SPU_ENFORCE((type.isa<ring::InputTy>() &&
                 type.as<ring::InputTy>()->rank() == rank()) ||
                    (type.isa<ring::OutputTy>() &&
                     type.as<ring::OutputTy>()->rank() == rank()),
                "ring external input domain/rank mismatch");
  } else {
    SPU_ENFORCE(
        (type.isa<InputTy>() && type.as<InputTy>()->rank() == rank()) ||
            (type.isa<OutputTy>() && type.as<OutputTy>()->rank() == rank()),
        "external input cannot assert internal authentication");
  }
}

void SpdzWiseState::validateInternal(const NdArrayRef& value) const {
  requireRunning();
  if (isRing()) {
    SPU_ENFORCE(ring_, "ring state is not initialized");
    return ring_->validateInternal(value);
  }
  SPU_ENFORCE(value.eltype().isa<AShrTy>() &&
                  value.eltype().as<AShrTy>()->session() == session_,
              "SPDZWISE value belongs to a different session/type");
}

std::vector<RepShare> SpdzWiseState::multiplyRaw(
    const std::vector<RepShare>& x, const std::vector<RepShare>& y) {
  SPU_ENFORCE(x.size() == y.size(), "raw multiplication size mismatch");
  std::vector<RepShare> result(x.size());
  MultiplyF61(
      mask_prg_.get(), link_, x.size(), [&](int64_t i) { return x[i]; },
      [&](int64_t i) { return y[i]; },
      [&](int64_t i, const RepShare& value) { result[i] = value; },
      tag("multiply"));
  return result;
}

uint64_t SpdzWiseState::openRaw(const RepShare& value) {
  const auto bytes = EncodeF61({value[0], value[1]});
  const auto buffers = yacl::link::AllGather(
      link_, yacl::ByteContainerView(bytes), tag("check-open"));
  std::array<std::vector<uint64_t>, 3> shares;
  for (size_t i = 0; i < 3; ++i) {
    shares[i] = DecodeF61(std::string_view(buffers[i]), 2);
  }
  for (size_t i = 0; i < 3; ++i) {
    SPU_ENFORCE(shares[i][1] == shares[(i + 1) % 3][0], "inconsistent opening");
  }
  return Add(Add(shares[0][0], shares[1][0]), shares[2][0]);
}

void SpdzWiseState::maybeCheck() {
  requireRunning();
  const auto threshold = config_.spdzwise_config.check_threshold;
  const auto arithmetic = isRing() ? ring_->pendingRecords() : pending_.size();
  if (arithmetic >= threshold) checkMac();
  if (boolean_->pendingRecords() >= threshold) checkBoolean();
}

void SpdzWiseState::checkMac() {
  try {
    requireRunning();
    if (isRing()) {
      if (ring_->checkMac()) {
        SPU_ENFORCE(arithmetic_checks_ != std::numeric_limits<uint64_t>::max(),
                    "check counter overflow");
        ++arithmetic_checks_;
      }
      return;
    }
    if (pending_.empty()) {
      return;
    }
    // The secret-coefficient check follows SpdzWise::check / zero_check. Do not
    // replace the coefficients with public PRSS values or open u-alpha*w
    // itself.
    const auto coefficients = randomRep(pending_.size());
    // Read pending shares directly and accumulate value/MAC dot products in
    // one pass. For a replicated product, a0*r0+a0*r1+a1*r0 equals
    // a0*(r0+r1)+a1*r0. Both dot products reuse r0+r1, reducing six field
    // multiplications per record to four, without materializing values/macs.
    const auto sums = yacl::parallel_reduce<std::array<uint64_t, 2>>(
        0, pending_.size(), kMinTaskSize,
        [&](int64_t begin, int64_t end) {
          std::array<uint64_t, 2> partial{};
          for (int64_t i = begin; i < end; ++i) {
            const auto& value = pending_[i];
            const auto& coefficient = coefficients[i];
            const auto rsum = Add(coefficient[0], coefficient[1]);
            partial[0] = Add(partial[0], Add(Mul(value[0], rsum),
                                             Mul(value[1], coefficient[0])));
            partial[1] = Add(partial[1], Add(Mul(value[2], rsum),
                                             Mul(value[3], coefficient[0])));
          }
          return partial;
        },
        [](const auto& a, const auto& b) {
          return std::array<uint64_t, 2>{Add(a[0], b[0]), Add(a[1], b[1])};
        });
    // The two dot products are independent. Keep a separate zero-sharing
    // mask for each, but exchange the two masked local sums together.
    const auto masks = RandomF61Rep(mask_prg_.get(), 2);
    const std::vector<uint64_t> local = {
        Add(sums[0], Sub(masks[0][0], masks[0][1])),
        Add(sums[1], Sub(masks[1][0], masks[1][1]))};
    const auto next = exchange(local, "check-dots");
    const RepShare w = {local[0], next[0]};
    const RepShare u = {local[1], next[1]};
    const auto product = multiplyRaw({key_}, {w})[0];
    const RepShare t = {Sub(u[0], product[0]), Sub(u[1], product[1])};
    const auto blinded = multiplyRaw({t}, randomRep(1))[0];
    SPU_ENFORCE(openRaw(blinded) == 0,
                "SPDZWISE arithmetic authentication failed");
    SPU_ENFORCE(arithmetic_checks_ != std::numeric_limits<uint64_t>::max(),
                "check counter overflow");
    ++arithmetic_checks_;
    pending_.clear();

  } catch (...) {
    abort();
    throw;
  }
}

NdArrayRef SpdzWiseState::authenticateInput(const std::vector<RepShare>& values,
                                            const Shape& shape) {
  requireRunning();
  SPU_ENFORCE(static_cast<int64_t>(values.size()) == shape.numel(),
              "input shape mismatch");
  NdArrayRef result(makeType<AShrTy>(session_), shape);
  auto* out = result.data<AuthShare>();
  MultiplyF61(
      mask_prg_.get(), link_, values.size(),
      [&](int64_t i) { return values[i]; }, [&](int64_t) { return key_; },
      [&](int64_t i, const RepShare& mac) {
        out[i] = {values[i][0], values[i][1], mac[0], mac[1]};
      },
      tag("input-mac"));
  if (!values.empty()) {
    pending_.insert(pending_.end(), out, out + values.size());
  }
  return result;
}

NdArrayRef SpdzWiseState::importValue(const NdArrayRef& value) {
  try {
    requireRunning();
    if (value.eltype().isa<BShrTy>()) {
      validateBoolean(value);
      return value;
    }
    if (value.eltype().isa<BInputTy>() || value.eltype().isa<BOutputTy>()) {
      validateExternalInput(Value(value, DT_I64));
      const NdArrayView<RepShare> in(value);
      std::vector<RepShare> shares(value.numel());
      for (int64_t i = 0; i < value.numel(); ++i) {
        shares[i] = in[i];
      }
      boolean_->checkCopies(shares);
      return value.clone().as(
          makeType<BShrTy>(session_, value.eltype().as<BShare>()->nbits()));
    }
    if (isRing()) {
      if (value.eltype().isa<ring::AShrTy>()) {
        validateInternal(value);
        return value;
      }
      validateExternalInput(Value(value, DT_I64));
      auto result = ring_->importValue(value);
      checkMac();
      return result;
    }
    if (value.eltype().isa<AShrTy>()) {
      validateInternal(value);
      return value;
    }
    validateExternalInput(Value(value, DT_I64));
    const NdArrayView<RepShare> input(value);
    std::vector<RepShare> values(value.numel());
    for (int64_t i = 0; i < value.numel(); ++i) {
      values[i] = input[i];
    }
    checkCopies(values);
    auto result = authenticateInput(values, value.shape());
    checkMac();
    return result;

  } catch (...) {
    abort();
    throw;
  }
}

void SpdzWiseState::checkBoolean() {
  try {
    requireRunning();
    boolean_->check();
  } catch (...) {
    abort();
    throw;
  }
}

void SpdzWiseState::verify() {
  checkMac();
  checkBoolean();
}

void SpdzWiseState::finalize() { verify(); }

NdArrayRef SpdzWiseState::exportValue(const NdArrayRef& value,
                                      const Type& type) const {
  requireRunning();
  SPU_ENFORCE(pending_.empty() && (!ring_ || ring_->empty()) &&
                  pendingBooleanBits() == 0,
              "unchecked SPDZWISE results cannot be exported");
  if (value.eltype().isa<BShrTy>()) {
    validateBoolean(value);
    SPU_ENFORCE(type.isa<BOutputTy>() &&
                    type.as<BOutputTy>()->rank() == rank() &&
                    type.as<BOutputTy>()->nbits() ==
                        value.eltype().as<BShrTy>()->nbits(),
                "invalid Boolean export type");
    return value.clone().as(type);
  }
  if (isRing()) return ring_->exportValue(value, type);
  validateInternal(value);
  SPU_ENFORCE(type.isa<OutputTy>() && type.as<OutputTy>()->rank() == rank(),
              "invalid export type");
  NdArrayRef result(type, value.shape());
  const NdArrayView<AuthShare> in(value);
  NdArrayView<RepShare> out(result);
  for (int64_t i = 0; i < value.numel(); ++i) {
    out[i] = Values(in[i]);
  }
  return result;
}

}  // namespace spu::mpc::spdzwise
