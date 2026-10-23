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

#include "libspu/mpc/spdzwise/ring.h"

#include <cstring>
#include <limits>

#include "yacl/crypto/hash/blake3.h"
#include "yacl/link/algorithm/allgather.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/ring_arithmetic.h"
#include "libspu/mpc/spdzwise/state.h"

namespace spu::mpc::spdzwise::ring {
namespace {
void Append(std::string& bytes, std::string_view value) {
  bytes += std::to_string(value.size()) + ":";
  bytes.append(value);
}

template <typename GetWord>
std::string HashWords(std::string_view domain, std::string_view context,
                      size_t owner, size_t count, GetWord&& get) {
  SPU_ENFORCE(count <= std::numeric_limits<size_t>::max() / sizeof(Word),
              "ring hash size overflow");
  std::string bytes;
  Append(bytes, "SPDZWISE-RING-v1");
  Append(bytes, domain);
  Append(bytes, context);
  Append(bytes, std::to_string(owner));
  // Preserve exactly the original length-prefixed LE transcript. Stream the
  // last field in bounded chunks instead of materializing/copying whole
  // batches.
  bytes += std::to_string(count * sizeof(Word)) + ":";
  yacl::crypto::Blake3Hash hash;
  hash.Update(bytes);
  constexpr size_t kHashWords = 4096;
  std::vector<Word> chunk(std::min(count, kHashWords));
  for (size_t offset = 0; offset < count; offset += kHashWords) {
    chunk.resize(std::min(count - offset, kHashWords));
    for (size_t i = 0; i < chunk.size(); ++i) chunk[i] = get(offset + i);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    hash.Update(
        yacl::ByteContainerView(chunk.data(), chunk.size() * sizeof(Word)));
#else
    hash.Update(Encode(chunk));
#endif
  }
  const auto digest = hash.CumulativeHash();
  return {reinterpret_cast<const char*>(digest.data()), digest.size()};
}

std::string Hash(std::string_view domain, std::string_view context,
                 size_t owner, const std::vector<Word>& words) {
  return HashWords(domain, context, owner, words.size(),
                   [&](size_t i) { return words[i]; });
}

// Two rounds propagate an honest rejection to every honest party, even if a
// corrupt party equivocates about its own status in the first round.
void RequireAll(const std::shared_ptr<yacl::link::Context>& link, bool ok,
                const std::string& context) {
  for (size_t round = 0; round < 2; ++round) {
    const std::string status(1, ok ? '1' : '0');
    const auto statuses =
        yacl::link::AllGather(link, yacl::ByteContainerView(status),
                              context + "/status/" + std::to_string(round));
    for (const auto& item : statuses) {
      ok &= std::string_view(item) == "1";
    }
  }
  SPU_ENFORCE(ok, "SPDZWISE ring check failed");
}

template <typename GetShare>
void CheckCopiesImpl(const std::shared_ptr<yacl::link::Context>& link,
                     size_t count, GetShare&& get, std::string_view context) {
  const auto sent = HashWords("copies", context, link->Rank(), count,
                              [&](size_t i) { return get(i)[0]; });
  const std::string tag = std::string(context) + "/copies";
  // Component i is held by i and i-1. Only hashes reach its duplicate holder.
  link->SendAsync(link->PrevRank(), yacl::ByteContainerView(sent), tag);
  const auto received = link->Recv(link->NextRank(), tag);
  RequireAll(link,
             std::string_view(received) ==
                 HashWords("copies", context, link->NextRank(), count,
                           [&](size_t i) { return get(i)[1]; }),
             tag);
}

std::vector<RepShare> Random(PrgState* prg, size_t count) {
  std::vector<Word> a(count), b(count);
  if (count != 0) {
    prg->fillPrssPair(a.data(), b.data(), count, PrgState::GenPrssCtrl::Both);
  }
  std::vector<RepShare> out(count);
  for (size_t i = 0; i < count; ++i) out[i] = {a[i], b[i]};
  return out;
}

RepShare Values(const AuthShare& x) { return {x[0], x[1]}; }
RepShare Macs(const AuthShare& x) { return {x[2], x[3]}; }
void RequirePublic(const NdArrayRef& value) {
  SPU_ENFORCE(value.eltype() == makeType<Pub2kTy>(FM64),
              "expected FM64 public operand");
}
}  // namespace

std::string Encode(const std::vector<Word>& words) {
  SPU_ENFORCE(words.size() <= std::numeric_limits<size_t>::max() / 16,
              "ring wire size overflow");
  std::string bytes(words.size() * 16, '\0');
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (!words.empty()) std::memcpy(bytes.data(), words.data(), bytes.size());
#else
  for (size_t i = 0; i < words.size(); ++i) {
    for (size_t j = 0; j < 16; ++j) {
      bytes[16 * i + j] = static_cast<char>(words[i] >> (8 * j));
    }
  }
#endif
  return bytes;
}

std::vector<Word> Decode(std::string_view bytes, size_t count) {
  SPU_ENFORCE(count <= std::numeric_limits<size_t>::max() / 16 &&
                  bytes.size() == count * 16,
              "invalid ring wire length");
  std::vector<Word> words(count);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (count != 0) std::memcpy(words.data(), bytes.data(), bytes.size());
#else
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      words[i] |= Word(static_cast<uint8_t>(bytes[16 * i + j])) << (8 * j);
    }
  }
#endif
  return words;
}

void CheckCopies(const std::shared_ptr<yacl::link::Context>& link,
                 const std::vector<RepShare>& values,
                 std::string_view context) {
  CheckCopiesImpl(
      link, values.size(), [&](size_t i) { return values[i]; }, context);
}

void CheckCopies(const std::shared_ptr<yacl::link::Context>& link,
                 const NdArrayRef& values, std::string_view context,
                 bool include_macs) {
  const NdArrayView<AuthShare> data(values);
  if (include_macs) {
    CheckCopiesImpl(
        link, 2 * values.numel(),
        [&](size_t i) {
          return i % 2 == 0 ? Values(data[i / 2]) : Macs(data[i / 2]);
        },
        context);
  } else {
    CheckCopiesImpl(link, values.numel(),
                    [&](size_t i) { return Values(data[i]); }, context);
  }
}

void CheckZero(const std::shared_ptr<yacl::link::Context>& link,
               const RepShare& value, std::string_view context) {
  // Reference: Mark Abspoel, Anders Dalskov, Daniel Escudero, Ariel Nof,
  // "An Efficient Passive-to-Active Compiler for Honest-Majority MPC over
  // Rings", Section 6.3, Protocol 9.
  // https://ir.cwi.nl/pub/31033/31033.pdf#page=23 Pi holds (ti,t{i+1}). Send
  // H(-(ti+t{i+1})) to P{i+1}, receive from P{i-1}, compare with H(t{i+1}). The
  // sent target is component i+2; the received target is i+1. This direction is
  // opposite to resharing. Assumptions: three parties, <=1 static malicious
  // corruption, security with abort in the random-oracle model. BLAKE3-256
  // instantiates that oracle. Additional engineering checks: replicated-copy
  // consistency, unambiguous version/session/execution/batch/component domain
  // separation, and all-party failure propagation. Never open the residual or
  // its arithmetic multiple.
  CheckCopies(link, {value}, std::string(context) + "/residual");
  const std::string tag = std::string(context) + "/zero";
  const auto sent = Hash("zero", context, (link->Rank() + 2) % 3,
                         {Word(0) - value[0] - value[1]});
  link->SendAsync(link->NextRank(), yacl::ByteContainerView(sent), tag);
  const auto received = link->Recv(link->PrevRank(), tag);
  RequireAll(link,
             std::string_view(received) ==
                 Hash("zero", context, link->NextRank(), {value[1]}),
             tag);
}

ArithmeticState::ArithmeticState(
    std::shared_ptr<yacl::link::Context> link, std::string session,
    std::function<std::string(std::string_view)> tag)
    : link_(std::move(link)),
      session_(std::move(session)),
      tag_(std::move(tag)) {
  key_prg_ = std::make_unique<PrgState>(link_);
  mask_prg_ = std::make_unique<PrgState>(link_);
  check_prg_ = std::make_unique<PrgState>(link_);
  key_ = Random(key_prg_.get(), 1)[0];
  CheckCopies(link_, {key_}, tag_("ring-key"));
}

void ArithmeticState::validateInternal(const NdArrayRef& value) const {
  SPU_ENFORCE(value.eltype().isa<AShrTy>() &&
                  value.eltype().as<AShrTy>()->session() == session_,
              "ring value belongs to a different session/domain");
}

void ArithmeticState::exchange(const std::vector<Word>& local,
                               std::vector<Word>& next,
                               std::string_view operation) {
  const auto context = tag_(operation);
  if (local.empty()) {
    next.clear();
    return;
  }
  SPU_ENFORCE(local.size() <= std::numeric_limits<size_t>::max() / sizeof(Word),
              "ring wire size overflow");
  const size_t nbytes = local.size() * sizeof(Word);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // SendAsync copies this view into transport-owned storage before returning.
  // The native layout already is the canonical wire format on LE hosts.
  link_->SendAsync(link_->PrevRank(),
                   yacl::ByteContainerView(local.data(), nbytes), context);
#else
  const auto bytes = Encode(local);
  link_->SendAsync(link_->PrevRank(), yacl::ByteContainerView(bytes), context);
#endif
  const auto received = link_->Recv(link_->NextRank(), context);
  // Keep the same two-round failure propagation before reading the payload.
  RequireAll(link_, static_cast<size_t>(received.size()) == nbytes,
             context + "/wire");
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // The second PRSS mask has already been consumed. Reuse its allocation,
  // avoiding a new zero-initialized decoding vector on every multiplication.
  next.resize(local.size());
  std::memcpy(next.data(), received.data(), nbytes);
#else
  next = Decode(std::string_view(received), local.size());
#endif
}

std::vector<RepShare> ArithmeticState::reshare(std::vector<Word> local,
                                               std::string_view operation) {
  std::vector<Word> first(local.size()), second(local.size());
  if (!local.empty()) {
    mask_prg_->fillPrssPair(first.data(), second.data(), local.size(),
                            PrgState::GenPrssCtrl::Both);
  }
  pforeach(0, local.size(),
           [&](int64_t i) { local[i] += first[i] - second[i]; });
  exchange(local, second, operation);
  std::vector<RepShare> out(local.size());
  pforeach(0, local.size(), [&](int64_t i) { out[i] = {local[i], second[i]}; });
  return out;
}

std::vector<RepShare> ArithmeticState::multiplyRaw(
    const std::vector<RepShare>& x, const std::vector<RepShare>& y) {
  SPU_ENFORCE(x.size() == y.size(), "ring multiplication size mismatch");
  std::vector<Word> local(x.size());
  pforeach(0, x.size(), [&](int64_t i) { local[i] = LocalProduct(x[i], y[i]); });
  return reshare(std::move(local), "ring-multiply");
}

void ArithmeticState::remember(const NdArrayRef& value) {
  if (value.numel() == 0) return;
  if (value.isCompact()) {
    const auto* data = value.data<AuthShare>();
    // Own immutable copies; bulk insertion avoids per-element capacity checks.
    pending_.insert(pending_.end(), data, data + value.numel());
    return;
  }
  const NdArrayView<AuthShare> in(value);
  for (int64_t i = 0; i < value.numel(); ++i) pending_.push_back(in[i]);
}

NdArrayRef ArithmeticState::importValue(const NdArrayRef& value) {
  if (value.eltype().isa<AShrTy>()) {
    validateInternal(value);
    return value;
  }
  std::vector<RepShare> values(value.numel());
  if (value.eltype().isa<InputTy>()) {
    SPU_ENFORCE(value.eltype().as<InputTy>()->rank() == link_->Rank(),
                "ring input rank mismatch");
    const NdArrayView<RepShare> in(value);
    for (int64_t i = 0; i < value.numel(); ++i) values[i] = in[i];
  } else {
    SPU_ENFORCE(value.eltype().isa<OutputTy>() &&
                    value.eltype().as<OutputTy>()->rank() == link_->Rank(),
                "invalid ring external input");
    const NdArrayView<spdzwise::RepShare> in(value);
    for (int64_t i = 0; i < value.numel(); ++i) {
      // Each external component has independent modulo-2^64 semantics.
      values[i] = {Word(in[i][0]), Word(in[i][1])};
    }
  }
  CheckCopies(link_, values, tag_("ring-input"));
  const auto macs =
      multiplyRaw(values, std::vector<RepShare>(values.size(), key_));
  NdArrayRef out(makeType<AShrTy>(session_), value.shape());
  for (int64_t i = 0; i < value.numel(); ++i) {
    out.data<AuthShare>()[i] = {values[i][0], values[i][1], macs[i][0],
                                macs[i][1]};
  }
  remember(out);
  return out;
}

NdArrayRef ArithmeticState::randomAuthenticated(const Shape& shape) {
  NdArrayRef out(makeType<AShrTy>(session_), shape);
  if (!out.numel()) return out;
  // Fresh draws share the multiplication-mask stream, never the key or
  // checking-coefficient stream. Low 64 bits are uniform in the application
  // ring; retain all 128 bits for the authenticated representative.
  const auto values = Random(mask_prg_.get(), out.numel());
  const auto macs =
      multiplyRaw(values, std::vector<RepShare>(values.size(), key_));
  pforeach(0, out.numel(), [&](int64_t i) {
    out.data<AuthShare>()[i] = {values[i][0], values[i][1], macs[i][0],
                                macs[i][1]};
  });
  remember(out);
  return out;
}

bool ArithmeticState::checkMac() {
  const auto context = tag_("ring-check");
  const auto count = std::to_string(pending_.size());
  const auto counts = yacl::link::AllGather(
      link_, yacl::ByteContainerView(count), context + "/count");
  bool agree = true;
  for (const auto& item : counts) agree &= std::string_view(item) == count;
  RequireAll(link_, agree, context + "/count");
  if (pending_.empty()) return false;
  // Freeze records and check their copies BEFORE sampling fresh secret
  // coefficients. pending_ owns copies rather than aliases of mutable arrays.
  CheckCopiesImpl(
      link_, 2 * pending_.size(),
      [&](size_t i) {
        return i % 2 == 0 ? Values(pending_[i / 2]) : Macs(pending_[i / 2]);
      },
      context + "/records");
  std::vector<Word> first(pending_.size()), second(pending_.size());
  // Consume the same separated PRSS streams, on the calling thread and only
  // AFTER copy agreement. Parallel workers only access disjoint/read-only data.
  check_prg_->fillPrssPair(first.data(), second.data(), pending_.size(),
                           PrgState::GenPrssCtrl::Both);
  const auto sums = yacl::parallel_reduce<std::array<Word, 2>>(
      0, pending_.size(), kMinTaskSize,
      [&](int64_t begin, int64_t end) {
        std::array<Word, 2> partial{};
        for (int64_t i = begin; i < end; ++i) {
          const auto& x = pending_[i];
          const Word rsum = first[i] + second[i];
          partial[0] += x[0] * rsum + x[1] * first[i];
          partial[1] += x[2] * rsum + x[3] * first[i];
        }
        return partial;
      },
      [](const auto& a, const auto& b) {
        return std::array<Word, 2>{a[0] + b[0], a[1] + b[1]};
      });
  const auto dots = reshare({sums[0], sums[1]}, "ring-check-dots");
  const auto product = multiplyRaw({key_}, {dots[0]})[0];
  const RepShare residual = {dots[1][0] - product[0], dots[1][1] - product[1]};
  CheckZero(link_, residual, context);
  pending_.clear();
  return true;
}

NdArrayRef ArithmeticState::exportValue(const NdArrayRef& value,
                                        const Type& type) const {
  validateInternal(value);
  SPU_ENFORCE(empty() && type.isa<OutputTy>() &&
                  type.as<OutputTy>()->rank() == link_->Rank(),
              "invalid or unchecked ring export");
  NdArrayRef out(type, value.shape());
  const NdArrayView<AuthShare> in(value);
  for (int64_t i = 0; i < value.numel(); ++i) {
    out.data<spdzwise::RepShare>()[i] = {static_cast<uint64_t>(in[i][0]),
                                         static_cast<uint64_t>(in[i][1])};
  }
  return out;
}

NdArrayRef ArithmeticState::publicValue(const NdArrayRef& value) const {
  RequirePublic(value);
  NdArrayRef result(makeType<AShrTy>(session_), value.shape());
  const NdArrayView<uint64_t> in(value);
  auto* out = result.data<AuthShare>();
  pforeach(0, value.numel(), [&](int64_t i) {
    const Word x = in[i];
    out[i] = {link_->Rank() == 0 ? x : 0, link_->Rank() == 2 ? x : 0,
              key_[0] * x, key_[1] * x};
  });
  return result;
}

NdArrayRef ArithmeticState::add(const NdArrayRef& x,
                                const NdArrayRef& y) const {
  validateInternal(x);
  validateInternal(y);
  SPU_ENFORCE(x.shape() == y.shape(), "ring add shape mismatch");
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> a(x), b(y);
  auto* out = result.data<AuthShare>();
  pforeach(0, x.numel(), [&](int64_t i) {
    for (size_t c = 0; c < 4; ++c) out[i][c] = a[i][c] + b[i][c];
  });
  return result;
}

NdArrayRef ArithmeticState::negate(const NdArrayRef& x) const {
  validateInternal(x);
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> a(x);
  auto* out = result.data<AuthShare>();
  pforeach(0, x.numel(), [&](int64_t i) {
    for (size_t c = 0; c < 4; ++c) out[i][c] = Word(0) - a[i][c];
  });
  return result;
}

NdArrayRef ArithmeticState::leftShift(const NdArrayRef& x,
                                      const Sizes& bits) const {
  validateInternal(x);
  SPU_ENFORCE(bits.size() == 1 || int64_t(bits.size()) == x.numel(),
              "shift count shape mismatch");
  for (auto bit : bits) {
    SPU_ENFORCE(bit >= 0 && bit < 64, "shift count must be in [0,63]");
  }
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> in(x);
  auto* out = result.data<AuthShare>();
  pforeach(0, x.numel(), [&](int64_t i) {
    const auto shift = bits.size() == 1 ? bits[0] : bits[i];
    // Preserve the complete authenticated representative in Z/(2^128).
    for (size_t c = 0; c < 4; ++c) out[i][c] = in[i][c] << shift;
  });
  return result;
}

NdArrayRef ArithmeticState::multiplyPublic(const NdArrayRef& x,
                                           const NdArrayRef& y) const {
  validateInternal(x);
  RequirePublic(y);
  SPU_ENFORCE(x.shape() == y.shape(), "ring multiply shape mismatch");
  NdArrayRef result(x.eltype(), x.shape());
  const NdArrayView<AuthShare> a(x);
  const NdArrayView<uint64_t> b(y);
  auto* out = result.data<AuthShare>();
  pforeach(0, x.numel(), [&](int64_t i) {
    for (size_t c = 0; c < 4; ++c) out[i][c] = a[i][c] * Word(b[i]);
  });
  return result;
}

NdArrayRef ArithmeticState::multiply(const NdArrayRef& x, const NdArrayRef& y) {
  validateInternal(x);
  validateInternal(y);
  SPU_ENFORCE(x.shape() == y.shape(), "ring multiply shape mismatch");
  const NdArrayView<AuthShare> a(x), b(y);
  std::vector<Word> first(2 * x.numel()), second(2 * x.numel());
  if (x.numel() != 0) {
    mask_prg_->fillPrssPair(first.data(), second.data(), first.size(),
                            PrgState::GenPrssCtrl::Both);
  }
  // Fuse mask application and local products, reusing the PRSS buffers. Avoid
  // constructing a replicated mask vector and a replicated product vector.
  pforeach(0, x.numel(), [&](int64_t i) {
    const auto& lhs = a[i];
    const auto& rhs = b[i];
    const Word ysum = rhs[0] + rhs[1];
    first[2 * i] += lhs[0] * ysum + lhs[1] * rhs[0] - second[2 * i];
    first[2 * i + 1] += lhs[2] * ysum + lhs[3] * rhs[0] - second[2 * i + 1];
  });
  exchange(first, second, "ring-multiply");
  NdArrayRef result(x.eltype(), x.shape());
  auto* out = result.data<AuthShare>();
  pforeach(0, x.numel(), [&](int64_t i) {
    out[i] = {first[2 * i], second[2 * i], first[2 * i + 1], second[2 * i + 1]};
  });
  remember(result);
  return result;
}

NdArrayRef ArithmeticState::matmul(const NdArrayRef& x, const NdArrayRef& y,
                                   bool public_rhs) {
  validateInternal(x);
  if (public_rhs)
    RequirePublic(y);
  else
    validateInternal(y);
  SPU_ENFORCE(x.ndim() == 2 && y.ndim() == 2 && x.shape()[1] == y.shape()[0],
              "ring matmul requires compatible matrices");
  const int64_t m = x.shape()[0], k = x.shape()[1], n = y.shape()[1];
  NdArrayRef result(x.eltype(), {m, n});
  // Pack strided inputs once, rather than resolving NdArrayView indices for
  // every scalar product. RHS columns are contiguous in the inner dot loop.
  const auto lhs = x.isCompact() ? x : x.clone();
  const auto* a = lhs.data<AuthShare>();
  // A cell costs O(k), so schedule by scalar work rather than output size.
  const int64_t grain = std::max<int64_t>(1, kMinTaskSize / std::max<int64_t>(k, 1));
  auto* out = result.data<AuthShare>();
  if (public_rhs) {
    const NdArrayView<uint64_t> b(y);
    std::vector<uint64_t> columns(y.numel());
    pforeach(0, n, [&](int64_t j) {
      for (int64_t t = 0; t < k; ++t) columns[j * k + t] = b[t * n + j];
    });
    yacl::parallel_for(0, result.numel(), grain, [&](int64_t begin, int64_t end) {
      for (int64_t cell = begin; cell < end; ++cell) {
        const int64_t row = (cell / n) * k, col = (cell % n) * k;
        AuthShare sum{};
        for (int64_t t = 0; t < k; ++t) {
          const Word rhs = columns[col + t];
          for (size_t c = 0; c < 4; ++c) {
            sum[c] += a[row + t][c] * rhs;
          }
        }
        out[cell] = sum;
      }
    });
    return result;
  }
  const NdArrayView<AuthShare> b(y);
  std::vector<RepShare> columns(y.numel());
  pforeach(0, n, [&](int64_t j) {
    for (int64_t t = 0; t < k; ++t) {
      const auto& rhs = b[t * n + j];
      columns[j * k + t] = {rhs[0] + rhs[1], rhs[0]};
    }
  });
  std::vector<Word> local(2 * result.numel());
  yacl::parallel_for(0, result.numel(), grain, [&](int64_t begin, int64_t end) {
    for (int64_t cell = begin; cell < end; ++cell) {
      const int64_t row = (cell / n) * k, col = (cell % n) * k;
      Word value = 0, mac = 0;
      for (int64_t t = 0; t < k; ++t) {
        const auto& lhs = a[row + t];
        const auto& rhs = columns[col + t];
        value += lhs[0] * rhs[0] + lhs[1] * rhs[1];
        mac += lhs[2] * rhs[0] + lhs[3] * rhs[1];
      }
      local[2 * cell] = value;
      local[2 * cell + 1] = mac;
    }
  });
  // Exchange exactly two words per output, independent of inner dimension k.
  const auto products = reshare(std::move(local), "ring-matmul");
  for (int64_t i = 0; i < result.numel(); ++i) {
    out[i] = {products[2 * i][0], products[2 * i][1], products[2 * i + 1][0],
              products[2 * i + 1][1]};
  }
  remember(result);
  return result;
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->publicValue(in);
}

NdArrayRef Negate::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  return ctx->getState<SpdzWiseState>()->negate(in);
}

NdArrayRef LShiftA::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                         const Sizes& bits) const {
  return ctx->getState<SpdzWiseState>()->leftShift(in, bits);
}

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  return ctx->getState<SpdzWiseState>()->randomArithmetic(shape);
}

NdArrayRef AddAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->add(x, y);
}

NdArrayRef AddAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  auto* state = ctx->getState<SpdzWiseState>();
  return state->add(x, state->publicValue(y));
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->multiply(x, y);
}

NdArrayRef MulAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->multiplyPublic(x, y);
}

NdArrayRef MatmulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->matmul(x, y, false);
}

NdArrayRef MatmulAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ctx->getState<SpdzWiseState>()->matmul(x, y, true);
}

void registerArithmeticKernels(Object* object) {
  object->regKernel<P2A, Negate, AddAA, AddAP, MulAA, MulAP, MatmulAA, MatmulAP,
                    LShiftA, RandA>();
}

}  // namespace spu::mpc::spdzwise::ring
