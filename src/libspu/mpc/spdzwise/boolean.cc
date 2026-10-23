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

#include "libspu/mpc/spdzwise/boolean.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>

#include "yacl/crypto/hash/blake3.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/link/algorithm/allgather.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/state.h"

namespace spu::mpc::spdzwise {
namespace {
void RequirePublic(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype() == makeType<Pub2kTy>(FM64),
              "expected an FM64 public value");
}
constexpr int64_t kVerifyGrain = 131072;
constexpr size_t kSacrificeTile = 65536;
constexpr int64_t kTileGrain = 16384;
std::string Hash(std::string_view bytes) {
  yacl::crypto::Blake3Hash hash;
  hash.Update(bytes);
  const auto digest = hash.CumulativeHash();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}
std::string Encode(absl::Span<const uint64_t> words) {
  SPU_ENFORCE(words.size() <= std::numeric_limits<size_t>::max() / 8,
              "Boolean wire length overflow");
  std::string bytes(words.size() * 8, '\0');
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (!words.empty()) std::memcpy(bytes.data(), words.data(), bytes.size());
#else
  for (size_t i = 0; i < words.size(); ++i) {
    for (size_t b = 0; b < 8; ++b) {
      bytes[8 * i + b] = char(words[i] >> (8 * b));
    }
  }
#endif
  return bytes;
}
// Wire buffers need not be aligned. Validate their length before reading.
uint64_t ReadWord(const char* bytes) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint64_t word;
  std::memcpy(&word, bytes, sizeof(word));
  return word;
#else
  uint64_t word = 0;
  for (size_t b = 0; b < 8; ++b) {
    word |= uint64_t(static_cast<unsigned char>(bytes[b])) << (8 * b);
  }
  return word;
#endif
}
void WriteWord(char* bytes, uint64_t word) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  std::memcpy(bytes, &word, sizeof(word));
#else
  for (size_t b = 0; b < 8; ++b) bytes[b] = char(word >> (8 * b));
#endif
}
void SendWords(const std::shared_ptr<yacl::link::Context>& link,
               absl::Span<const uint64_t> words, const std::string& tag) {
  SPU_ENFORCE(words.size() <= std::numeric_limits<size_t>::max() / 8,
              "Boolean wire length overflow");
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  link->Send(link->PrevRank(),
             yacl::ByteContainerView(words.data(), words.size() * 8), tag);
#else
  const auto bytes = Encode(words);
  link->Send(link->PrevRank(), yacl::ByteContainerView(bytes), tag);
#endif
}
uint64_t LocalAnd(uint64_t x0, uint64_t x1, uint64_t y0, uint64_t y1) {
  return (x0 & (y0 ^ y1)) ^ (x1 & y0);
}
std::vector<uint64_t> Decode(std::string_view bytes, size_t count) {
  SPU_ENFORCE(count <= std::numeric_limits<size_t>::max() / 8 &&
                  bytes.size() == count * 8,
              "Boolean wire length mismatch");
  std::vector<uint64_t> words(count);
  for (size_t i = 0; i < count; ++i) {
    words[i] = ReadWord(bytes.data() + 8 * i);
  }
  return words;
}
size_t Words(size_t bits) { return bits / 64 + (bits % 64 != 0); }
size_t PoolWords(size_t n, size_t bucket) {
  SPU_ENFORCE(n <= (std::numeric_limits<size_t>::max() - 3) / (bucket - 1),
              "ABFL pool size overflow");
  return (bucket - 1) * n + 3;
}
}  // namespace

size_t AbflBucketSize(size_t words) {
  if (words == 0) {
    return 0;
  }
  if (words >= 1048576) {
    return 3;
  }
  if (words >= 10368) {
    return 4;
  }
  SPU_ENFORCE(words >= 1024,
              "ABFL needs at least 1024 packed AND words, got {}", words);
  return 5;
}

BooleanState::BooleanState(std::shared_ptr<yacl::link::Context> link,
                           std::function<std::string(std::string_view)> tag)
    : link_(std::move(link)),
      tag_(std::move(tag)),
      random_prg_(link_),
      mask_prg_(link_) {}

std::vector<yacl::Buffer> BooleanState::gather(std::string_view bytes,
                                               std::string_view name,
                                               bool borrow_local) {
  const auto tag = tag_(name);
  std::vector<yacl::Buffer> buffers;
  if (borrow_local) {
    // Same all-to-all exchange as AllGather, without its local Buffer copy.
    // Send completes at the peer's transport buffer, without requiring its
    // application to Recv. Avoid the extra asynchronous staging copy.
    buffers.resize(3);
    for (size_t rank = 0; rank < 3; ++rank) {
      if (rank != link_->Rank()) {
        link_->Send(rank, yacl::ByteContainerView(bytes), tag);
      }
    }
    for (size_t rank = 0; rank < 3; ++rank) {
      if (rank != link_->Rank()) buffers[rank] = link_->Recv(rank, tag);
    }
  } else {
    buffers = yacl::link::AllGather(link_, yacl::ByteContainerView(bytes), tag);
  }
  std::vector<std::string> payload_hashes;
  if (borrow_local) {
    payload_hashes.resize(buffers.size());
    // Independent payload hashes can run concurrently. The outer hash binds
    // their lengths, rank order and opening domain; no view check is omitted.
    yacl::parallel_for(0, buffers.size(), bytes.size() >= 65536 ? 1 : 4,
                       [&](int64_t begin, int64_t end) {
                         for (auto rank = begin; rank < end; ++rank) {
                           payload_hashes[rank] =
                               Hash(size_t(rank) == link_->Rank()
                                        ? bytes
                                        : std::string_view(buffers[rank]));
                         }
                       });
  }
  yacl::crypto::Blake3Hash hash;
  if (borrow_local) hash.Update(tag + ":opening-views:");
  for (size_t rank = 0; rank < buffers.size(); ++rank) {
    const auto view = borrow_local && rank == link_->Rank()
                          ? bytes
                          : std::string_view(buffers[rank]);
    hash.Update(std::to_string(view.size()) + ":");
    hash.Update(borrow_local ? std::string_view(payload_hashes[rank]) : view);
  }
  // AllGather alone is not a malicious broadcast. Compare the ordered views
  // before using any opened value or moving to the next protocol phase.
  const auto digest = hash.CumulativeHash();
  const auto views = yacl::link::AllGather(
      link_, yacl::ByteContainerView(digest), tag_("boolean-views"));
  for (const auto& view : views) {
    SPU_ENFORCE(
        std::string_view(view) ==
            std::string_view(reinterpret_cast<const char*>(digest.data()),
                             digest.size()),
        "Boolean transcript disagreement");
  }
  return buffers;
}
void BooleanState::agree(std::string_view bytes) {
  const auto digest = Hash(bytes);
  for (const auto& peer : gather(digest, "boolean-agree")) {
    SPU_ENFORCE(std::string_view(peer) == digest, "Boolean batch disagreement");
  }
}
void BooleanState::requireAll(bool valid, std::string_view reason) {
  for (const auto& peer : gather(valid ? "1" : "0", "boolean-status")) {
    SPU_ENFORCE(std::string_view(peer) == "1", "{}", reason);
  }
}

std::vector<RepShare> BooleanState::andRaw(const std::vector<RepShare>& x,
                                           const std::vector<RepShare>& y) {
  SPU_ENFORCE(x.size() == y.size(), "Boolean AND shape mismatch");
  if (x.empty()) {
    return {};
  }
  std::vector<uint64_t> a(x.size()), b(x.size());
  mask_prg_.fillPrssPair(a.data(), b.data(), a.size(),
                         PrgState::GenPrssCtrl::Both);
  for (size_t i = 0; i < x.size(); ++i) {
    a[i] ^= b[i] ^ LocalAnd(x[i][0], x[i][1], y[i][0], y[i][1]);
  }
  const auto tag = tag_("boolean-and");
  SendWords(link_, a, tag);
  const auto received = link_->Recv(link_->NextRank(), tag);
  SPU_ENFORCE(size_t(received.size()) == a.size() * 8,
              "Boolean wire length mismatch");
  std::vector<RepShare> out;
  out.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    out.push_back({a[i], ReadWord(received.data<char>() + i * 8)});
  }
  return out;
}
void BooleanState::checkCopies(const std::vector<RepShare>& shares) {
  if (shares.empty()) {
    return;
  }
  std::vector<uint64_t> first;
  first.reserve(shares.size());
  for (const auto& share : shares) first.push_back(share[0]);
  const auto tag = tag_("boolean-copies");
  SendWords(link_, first, tag);
  const auto received = link_->Recv(link_->NextRank(), tag);
  bool valid = static_cast<size_t>(received.size()) == first.size() * 8;
  if (valid) {
    for (size_t i = 0; i < shares.size(); ++i) {
      valid &= ReadWord(received.data<char>() + i * 8) == shares[i][1];
    }
  }
  requireAll(valid, "Boolean replicated copies disagree");
}
std::vector<uint64_t> BooleanState::open(absl::Span<const RepShare> shares,
                                         std::string_view operation) {
  std::vector<uint64_t> result(shares.size());
  open(shares, absl::MakeSpan(result), operation);
  return result;
}
void BooleanState::open(absl::Span<const RepShare> shares,
                        absl::Span<uint64_t> result,
                        std::string_view operation) {
  SPU_ENFORCE(result.size() == shares.size(), "Boolean opening shape mismatch");
  SPU_ENFORCE(shares.size() <= std::numeric_limits<size_t>::max() / 16,
              "Boolean opening length overflow");
  static_assert(sizeof(RepShare) == 2 * sizeof(uint64_t));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const auto bytes =
      shares.empty()
          ? std::string_view()
          : std::string_view(reinterpret_cast<const char*>(shares.data()),
                             shares.size() * sizeof(RepShare));
#else
  std::vector<uint64_t> words;
  words.reserve(shares.size() * 2);
  for (const auto& share : shares) {
    words.push_back(share[0]);
    words.push_back(share[1]);
  }
  const auto encoded = Encode(words);
  const auto bytes = std::string_view(encoded);
#endif
  const auto buffers = gather(bytes, operation, true);
  std::array<const char*, 3> data;
  for (size_t rank = 0; rank < 3; ++rank) {
    if (rank == link_->Rank()) {
      data[rank] = bytes.data();
      continue;
    }
    SPU_ENFORCE(size_t(buffers[rank].size()) == shares.size() * 16,
                "Boolean wire length mismatch");
    data[rank] = buffers[rank].data<char>();
  }
  // Do not use parallel_reduce<bool>: its vector<bool> partials share bits.
  const bool valid =
      shares.empty() || yacl::parallel_reduce<uint64_t>(
                            0, shares.size(), kTileGrain,
                            [&](int64_t begin, int64_t end) {
                              bool ok = true;
                              for (int64_t i = begin; i < end; ++i) {
                                const auto x0 = ReadWord(data[0] + 16 * i);
                                const auto x1 = ReadWord(data[1] + 16 * i);
                                const auto x2 = ReadWord(data[2] + 16 * i);
                                ok &= ReadWord(data[0] + 16 * i + 8) == x1;
                                ok &= ReadWord(data[1] + 16 * i + 8) == x2;
                                ok &= ReadWord(data[2] + 16 * i + 8) == x0;
                                result[i] = x0 ^ x1 ^ x2;
                              }
                              return uint64_t(ok);
                            },
                            [](uint64_t a, uint64_t b) { return a & b; });
  SPU_ENFORCE(valid, "Boolean opening copies disagree");
}
void BooleanState::reservePending(size_t full_words) {
  SPU_ENFORCE(full_words <= pending_.max_size(), "ABFL queue size overflow");
  if (full_words > pending_.capacity()) {
    // One capacity decision per batch; preserve amortized linear growth.
    const auto capacity = pending_.capacity();
    const auto grown = capacity <= pending_.max_size() / 2
                           ? 2 * capacity
                           : pending_.max_size();
    pending_.reserve(std::max(full_words, grown));
  }
}
void BooleanState::append(const std::vector<RepShare>& a,
                          const std::vector<RepShare>& b,
                          const std::vector<RepShare>& c, size_t width) {
  SPU_ENFORCE(
      width >= 1 && width <= 64 && a.size() == b.size() && a.size() == c.size(),
      "ABFL invalid packing batch");
  SPU_ENFORCE(a.size() <= (std::numeric_limits<size_t>::max() - bits_) / width,
              "ABFL bit count overflow");
  const auto new_bits = bits_ + a.size() * width;
  const auto full_words =
      pending_.size() + (overflow_bits_ + a.size() * width) / 64;
  reservePending(full_words);
  if (width == 64) {
    // Independent records may be reordered. Keep any old partial word out of
    // the full-width path, preserving the same mapping for all six components.
    for (size_t i = 0; i < a.size(); ++i) {
      pending_.push_back({a[i], b[i], c[i]});
    }
  } else {
    auto acc = overflow_;
    auto used = overflow_bits_;
    for (size_t i = 0; i < a.size(); ++i) {
      const Triple t{a[i], b[i], c[i]};
      for (size_t j = 0; j < 3; ++j) {
        for (size_t s = 0; s < 2; ++s) {
          acc[j][s] |= t[j][s] << used;
        }
      }
      const auto total = used + width;
      if (total >= 64) {
        pending_.push_back(acc);
        if (total > 64) {
          // Here used > 0, so the shift is strictly between 0 and 64.
          for (size_t j = 0; j < 3; ++j) {
            for (size_t s = 0; s < 2; ++s) {
              acc[j][s] = t[j][s] >> (64 - used);
            }
          }
        } else {
          acc = {};
        }
        used = total - 64;
      } else {
        used = total;
      }
    }
    overflow_ = acc;
    overflow_bits_ = used;
  }
  bits_ = new_bits;
}
NdArrayRef BooleanState::andBB(const NdArrayRef& x, const NdArrayRef& y,
                               const std::string& session) {
  const auto width = std::min(x.eltype().as<BShrTy>()->nbits(),
                              y.eltype().as<BShrTy>()->nbits());
  const auto mask = BooleanMask(width);
  SPU_ENFORCE(x.shape() == y.shape(), "Boolean AND shape mismatch");
  SPU_ENFORCE(
      size_t(x.numel()) <= (std::numeric_limits<size_t>::max() - bits_) / width,
      "ABFL bit count overflow");
  if (width == 64) {
    const auto count = size_t(x.numel());
    const auto new_bits = bits_ + count * 64;
    std::vector<Triple> snapshots;
    SPU_ENFORCE(count <= snapshots.max_size(), "ABFL queue size overflow");
    snapshots.reserve(count);
    NdArrayRef out(makeType<BShrTy>(session, 64), x.shape());
    if (count) {
      // Normalize a strided view once instead of indexing it repeatedly in the
      // product and snapshot loops. Compact inputs are borrowed, not copied.
      const auto lhs = x.isCompact() ? x : x.clone();
      const auto rhs = y.isCompact() ? y : y.clone();
      const auto* xp = lhs.data<RepShare>();
      const auto* yp = rhs.data<RepShare>();
      auto* zp = out.data<RepShare>();
      std::vector<uint64_t> first(count), second(count);
      mask_prg_.fillPrssPair(first.data(), second.data(), count,
                             PrgState::GenPrssCtrl::Both);
      for (size_t i = 0; i < count; ++i) {
        first[i] ^= second[i] ^ (xp[i][0] & yp[i][0]) ^ (xp[i][0] & yp[i][1]) ^
                    (xp[i][1] & yp[i][0]);
      }
      const auto tag = tag_("boolean-and");
      // Preserve the little-endian wire format. On little-endian machines
      // send the PRSS/product buffer directly, as in native ABY3 rotate().
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      link_->SendAsync(link_->PrevRank(),
                       yacl::ByteContainerView(first.data(), count * 8), tag);
#else
      const auto bytes = Encode(first);
      link_->SendAsync(link_->PrevRank(), yacl::ByteContainerView(bytes), tag);
#endif
      const auto received = link_->Recv(link_->NextRank(), tag);
      SPU_ENFORCE(size_t(received.size()) == count * 8,
                  "Boolean wire length mismatch");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
      second = Decode(std::string_view(received), count);
#endif
      for (size_t i = 0; i < count; ++i) {
        uint64_t next;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // memcpy avoids any alignment assumption about transport buffers.
        std::memcpy(&next, received.data<char>() + i * 8, 8);
#else
        next = second[i];
#endif
        const RepShare z{first[i], next};
        zp[i] = z;
        snapshots.push_back({xp[i], yp[i], z});
      }
    }
    if (count) pending_batches_.push_back(std::move(snapshots));
    bits_ = new_bits;
    layout_ += fmt::format("{}:{};", x.shape(), width);
    return out;
  }
  std::vector<RepShare> a(x.numel()), b(x.numel());
  const NdArrayView<RepShare> xv(x), yv(y);
  for (int64_t i = 0; i < x.numel(); ++i) {
    a[i] = {xv[i][0] & mask, xv[i][1] & mask};
    b[i] = {yv[i][0] & mask, yv[i][1] & mask};
  }
  auto c = andRaw(a, b);
  NdArrayRef out(makeType<BShrTy>(session, width), x.shape());
  NdArrayView<RepShare> ov(out);
  for (size_t i = 0; i < a.size(); ++i) {
    c[i][0] &= mask;
    c[i][1] &= mask;
    ov[i] = c[i];
  }
  append(a, b, c, width);
  layout_ += fmt::format("{}:{};", x.shape(), width);
  return out;
}
BooleanState::TriplePool BooleanState::generate(size_t words) {
  SPU_ENFORCE(words <= std::vector<Triple>().max_size(),
              "ABFL random pool allocation overflow");
  if (!words) return TriplePool(0);
  SPU_ENFORCE(words <= std::numeric_limits<size_t>::max() / (6 * 8),
              "ABFL random buffer size overflow");
  // Generate complete tiles using bounded scratch. Every full tile contains
  // an even number of uint64_t words, so splitting an AES-CTR stream discards
  // no partial block except at the same final tail as a single large fill.
  const auto capacity = std::min(words, kSacrificeTile);
  std::unique_ptr<uint64_t[]> random(new uint64_t[6 * capacity]);
  auto* a0 = random.get();
  auto* a1 = a0 + capacity;
  auto* b0 = a1 + capacity;
  auto* b1 = b0 + capacity;
  auto* r0 = b1 + capacity;
  auto* r1 = r0 + capacity;
  TriplePool out(words);
  for (size_t first = 0; first < words; first += capacity) {
    const auto size = std::min(capacity, words - first);
    // A fixed worker owns each entire PRSS stream. First/Second mutate
    // disjoint counters; within each stream a precedes b for every tile.
    yacl::parallel_for(
        0, 4, size >= kTileGrain ? 1 : 5, [&](int64_t begin, int64_t end) {
          for (auto task = begin; task < end; ++task) {
            if (task < 2) {
              const auto ctrl = task == 0 ? PrgState::GenPrssCtrl::First
                                          : PrgState::GenPrssCtrl::Second;
              random_prg_.fillPrssPair(a0, a1, size, ctrl);
              random_prg_.fillPrssPair(b0, b1, size, ctrl);
            } else {
              mask_prg_.fillPrssPair(r0, r1, size,
                                     task == 2 ? PrgState::GenPrssCtrl::First
                                               : PrgState::GenPrssCtrl::Second);
            }
          }
        });
    yacl::parallel_for(0, size, kTileGrain, [&](int64_t begin, int64_t end) {
      for (auto i = begin; i < end; ++i)
        r0[i] ^= r1[i] ^ LocalAnd(a0[i], a1[i], b0[i], b1[i]);
    });
    const auto tag = tag_("boolean-and");
    SendWords(link_, absl::Span<const uint64_t>(r0, size), tag);
    const auto received = link_->Recv(link_->NextRank(), tag);
    SPU_ENFORCE(size_t(received.size()) == size * 8,
                "Boolean wire length mismatch");
    yacl::parallel_for(0, size, kTileGrain, [&](int64_t begin, int64_t end) {
      for (auto i = begin; i < end; ++i) {
        out[first + i] = {
            RepShare{a0[i], a1[i]}, RepShare{b0[i], b1[i]},
            RepShare{r0[i], ReadWord(received.data<char>() + 8 * i)}};
      }
    });
  }

  return out;
}
std::vector<size_t> BooleanState::permutation(size_t count) {
  const auto domain = tag_("boolean-coin");
  const auto seed = yacl::crypto::SecureRandSeed();
  const auto bytes = Encode({uint64_t(seed), uint64_t(seed >> 64)});
  const auto commits =
      gather(Hash(domain + ":" + std::to_string(link_->Rank()) + ":" + bytes),
             "boolean-commit");
  const auto reveals = gather(bytes, "boolean-reveal");
  uint128_t common = 0;
  for (size_t rank = 0; rank < 3; ++rank) {
    SPU_ENFORCE(std::string_view(commits[rank]) ==
                    Hash(domain + ":" + std::to_string(rank) + ":" +
                         std::string(std::string_view(reveals[rank]))),
                "Boolean random commitment mismatch");
    const auto words = Decode(reveals[rank], 2);
    common ^= uint128_t(words[0]) | (uint128_t(words[1]) << 64);
  }
  return shuffledIndices(count, common);
}
std::vector<size_t> BooleanState::shuffledIndices(size_t count,
                                                  uint128_t seed) {
  // Amortize cipher-context setup over 32 KiB rather than the default 128 B.
  // AES counter order and rejection-sampled Fisher–Yates remain unchanged.
  yacl::crypto::Prg<uint64_t, 4096> prg(seed);
  std::vector<size_t> perm(count);
  std::iota(perm.begin(), perm.end(), 0);
  // Draw in the original order, then apply swaps in the original order.
  // Batching only exposes future public addresses for prefetch; it does not
  // parallelize dependent swaps or change the Fisher-Yates distribution.
  std::array<size_t, 64> swaps;
  for (size_t first = 0; first < count; first += swaps.size()) {
    const auto size = std::min(swaps.size(), count - first);
    for (size_t j = 0; j < size; ++j) {
      const auto i = first + j;
      const uint64_t range = count - i;
      uint64_t draw = prg();
      // threshold < range: the overwhelmingly common case needs no division
      // to calculate threshold. Rejected draws are identical to the reference.
      if (draw < range) {
        const uint64_t threshold = -range % range;
        while (draw < threshold) draw = prg();
      }
      swaps[j] = i + draw % range;
      __builtin_prefetch(&perm[swaps[j]], 1, 1);
    }
    for (size_t j = 0; j < size; ++j)
      std::swap(perm[first + j], perm[swaps[j]]);
  }
  return perm;
}
size_t BooleanState::checkWords() const {
  return bits_ ? std::max(size_t{1024}, Words(bits_)) : 0;
}
void BooleanState::checkPrepared(TriplePool pool) {
  const auto real_words = Words(bits_);
  const auto words = checkWords();
  const auto bucket = AbflBucketSize(words);
  const auto count = PoolWords(words, bucket);
  SPU_ENFORCE(pool.size() == count, "ABFL random pool length mismatch");
  // Build only a small table of views. Records remain in their original owned
  // blocks. Validate the total before writing any fixed-size scratch buffer.
  std::vector<absl::Span<const Triple>> blocks;
  blocks.emplace_back(pending_);
  for (const auto& batch : pending_batches_) blocks.emplace_back(batch);
  if (overflow_bits_) blocks.emplace_back(&overflow_, 1);
  size_t total = 0;
  for (const auto& block : blocks) {
    SPU_ENFORCE(block.size() <= real_words - total,
                "ABFL online pool length mismatch");
    total += block.size();
  }
  SPU_ENFORCE(total == real_words, "ABFL online pool length mismatch");
  // Public, correct online anchors need no AND or PRSS generation. Keep them
  // separate from real snapshots so pendingBits() and circuit metrics remain
  // unchanged. Include every dummy in copy checks and all sacrifice columns;
  // random companions must remain fresh, secret and used exactly once.
  const std::vector<Triple> padding(words - real_words, Triple{});
  blocks.emplace_back(padding);
  // Each worker owns an output range, irrespective of call-block boundaries.
  // No task mutates snapshots, counters, PRGs, or communication state.
  auto visitRange = [&](size_t first, size_t last, auto&& fn) {
    yacl::parallel_for(
        first, last, last - first <= kSacrificeTile ? kTileGrain : kVerifyGrain,
        [&](int64_t begin, int64_t end) {
          size_t offset = 0;
          for (const auto& block : blocks) {
            const auto limit = offset + block.size();
            for (size_t i = std::max(size_t(begin), offset);
                 i < std::min(size_t(end), limit); ++i) {
              fn(i, block[i - offset]);
            }
            offset = limit;
            if (offset >= size_t(end)) break;
          }
        });
  };
  // Freeze the complete online and random pools before the public coin.
  {
    constexpr size_t max_triples = std::numeric_limits<int64_t>::max() / 24;
    SPU_ENFORCE(words <= max_triples && count <= max_triples - words,
                "ABFL copy buffer size overflow");
    const auto total = words + count;
    const auto tiles = total / kSacrificeTile + (total % kSacrificeTile != 0);
    using Digest = std::array<uint8_t, 32>;
    std::vector<Digest> first_hashes(tiles), second_hashes(tiles);
    const auto tag = tag_("boolean-copies");
    // A digest goes only to the party that already holds this exact share.
    // Hash both local replicas with identical owner/batch/tile domains. This
    // uses the same BLAKE3 collision resistance as transcript agreement.
    // Bound concurrent scratch space independently of the global pool size.
    const auto tasks = std::min(size_t{6}, tiles);
    yacl::parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
      const auto capacity = std::min(kSacrificeTile, total);
      std::unique_ptr<char[]> scratch(new char[48 * capacity]);
      auto* first_bytes = scratch.get();
      auto* second_bytes = first_bytes + 24 * capacity;
      for (auto task = begin; task < end; ++task) {
        for (size_t tile = task; tile < tiles; tile += tasks) {
          const auto first = tile * kSacrificeTile;
          const auto last = first + std::min(kSacrificeTile, total - first);
          const auto bytes = 24 * (last - first);
          auto pack = [&](size_t i, const Triple& t) {
            for (size_t j = 0; j < 3; ++j) {
              WriteWord(first_bytes + 24 * (i - first) + 8 * j, t[j][0]);
              WriteWord(second_bytes + 24 * (i - first) + 8 * j, t[j][1]);
            }
          };
          size_t offset = 0;
          for (const auto& block : blocks) {
            const auto limit = offset + block.size();
            for (size_t i = std::max(first, offset); i < std::min(last, limit);
                 ++i)
              pack(i, block[i - offset]);
            offset = limit;
            if (offset >= last) break;
          }
          for (size_t i = std::max(first, words); i < last; ++i)
            pack(i, pool[i - words]);
          for (size_t replica = 0; replica < 2; ++replica) {
            yacl::crypto::Blake3Hash hash;
            hash.Update(fmt::format("{}:{}:{}:{}:", tag,
                                    (link_->Rank() + replica) % 3, first,
                                    last));
            hash.Update(
                std::string_view(replica ? second_bytes : first_bytes, bytes));
            const auto digest = hash.CumulativeHash();
            std::memcpy((replica ? second_hashes : first_hashes)[tile].data(),
                        digest.data(), sizeof(Digest));
          }
        }
      }
    });
    const auto bytes = tiles * sizeof(Digest);
    link_->Send(link_->PrevRank(),
                yacl::ByteContainerView(first_hashes.data(), bytes), tag);
    const auto received = link_->Recv(link_->NextRank(), tag);
    const bool valid =
        size_t(received.size()) == bytes &&
        std::memcmp(received.data(), second_hashes.data(), bytes) == 0;
    requireAll(valid, "Boolean replicated copies disagree");
  }

  const auto perm = permutation(count);
  std::vector<RepShare> sample;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) sample.push_back(pool[perm[i]][j]);
  }
  const auto opened = open(sample);
  for (size_t i = 0; i < 3; ++i) {
    SPU_ENFORCE((opened[3 * i] & opened[3 * i + 1]) == opened[3 * i + 2],
                "ABFL cut-and-choose failed");
  }

  SPU_ENFORCE(words <= std::vector<RepShare>().max_size() / 2,
              "ABFL opening buffer size overflow");
  // All scratch elements are overwritten by disjoint parallel ranges before
  // their first read. Reuse the allocation across all bucket columns.
  const auto capacity = std::min(words, kSacrificeTile);
  std::unique_ptr<Triple[]> companions(new Triple[capacity]);
  std::unique_ptr<RepShare[]> masked(new RepShare[2 * capacity]);
  std::unique_ptr<RepShare[]> checks(new RepShare[capacity]);
  std::unique_ptr<uint64_t[]> differences(new uint64_t[2 * capacity]);
  std::unique_ptr<uint64_t[]> zeros(new uint64_t[capacity]);
  const bool public0 = link_->Rank() == 0;
  const bool public1 = link_->Rank() == 2;
  for (size_t column = 0; column < bucket - 1; ++column) {
    for (size_t first = 0; first < words; first += capacity) {
      const auto size = std::min(capacity, words - first);
      visitRange(first, first + size, [&](size_t index, const Triple& t) {
        const auto i = index - first;
        // The permutation is public. Prefetch future random records, including
        // their last component when a 48-byte triple straddles cache lines.
        if (index + 16 < first + size) {
          const auto& next = pool[perm[3 + column * words + index + 16]];
          __builtin_prefetch(&next[0], 0, 1);
          __builtin_prefetch(&next[2], 0, 1);
        }
        const auto& companion = pool[perm[3 + column * words + index]];
        companions[i] = companion;
        for (size_t j = 0; j < 2; ++j) {
          masked[2 * i + j] = {t[j][0] ^ companion[j][0],
                               t[j][1] ^ companion[j][1]};
        }
      });
      open({masked.get(), 2 * size}, {differences.get(), 2 * size});
      visitRange(first, first + size, [&](size_t index, const Triple& t) {
        const auto i = index - first;
        const auto rho = differences[2 * i], sigma = differences[2 * i + 1];
        for (size_t share = 0; share < 2; ++share) {
          checks[i][share] = t[2][share] ^ companions[i][2][share] ^
                             (sigma & companions[i][0][share]) ^
                             (rho & companions[i][1][share]);
        }
        if (public0) checks[i][0] ^= rho & sigma;
        if (public1) checks[i][1] ^= rho & sigma;
      });
      open({checks.get(), size}, {zeros.get(), size});
      const auto nonzero = yacl::parallel_reduce<uint64_t>(
          0, size, kVerifyGrain,
          [&](int64_t begin, int64_t end) {
            uint64_t any = 0;
            for (auto i = begin; i < end; ++i) any |= zeros[i];
            return any;
          },
          [](uint64_t a, uint64_t b) { return a | b; });
      SPU_ENFORCE(nonzero == 0, "ABFL sacrifice failed");
    }
  }
  SPU_ENFORCE(checks_ != std::numeric_limits<uint64_t>::max(),
              "ABFL check count overflow");
  ++checks_;
  pending_.clear();
  pending_batches_.clear();
  overflow_ = {};
  overflow_bits_ = 0;
  bits_ = 0;
  layout_.clear();
}
void BooleanState::check() {
  if (!bits_) {
    layout_.clear();
    return;
  }
  const auto words = checkWords();
  const auto bucket = AbflBucketSize(words);
  agree(fmt::format("word64-v9-zero-padding:{}:{}:{}:{}:{}:{}:{}", bits_,
                    Words(bits_), words, bucket, overflow_bits_, kSacrificeTile,
                    layout_));
  checkPrepared(generate(PoolWords(words, bucket)));
}

NdArrayRef BooleanState::randomBit(const Shape& shape,
                                   const std::string& session) {
  NdArrayRef out(makeType<BShrTy>(session, 1), shape);
  if (!out.numel()) return out;
  std::vector<uint64_t> first(out.numel()), second(out.numel());
  random_prg_.fillPrssPair(first.data(), second.data(), first.size(),
                           PrgState::GenPrssCtrl::Both);
  auto* data = out.data<RepShare>();
  pforeach(0, out.numel(),
           [&](int64_t i) { data[i] = {first[i] & 1, second[i] & 1}; });
  return out;
}

NdArrayRef SpdzWiseState::randomBoolean(const Shape& shape) {
  try {
    requireRunning();
    return boolean_->randomBit(shape, session_);
  } catch (...) {
    abort();
    throw;
  }
}

Type SpdzWiseState::commonBooleanType(const Type& x, const Type& y) const {
  requireRunning();
  for (const auto* type : {&x, &y}) {
    SPU_ENFORCE(
        type->isa<BShrTy>() && type->as<BShrTy>()->session() == session_,
        "Boolean type belongs to a different session/type");
    BooleanMask(type->as<BShrTy>()->nbits());
  }
  return makeType<BShrTy>(
      session_, std::max(x.as<BShrTy>()->nbits(), y.as<BShrTy>()->nbits()));
}

void SpdzWiseState::validateBoolean(const NdArrayRef& value) const {
  requireRunning();
  SPU_ENFORCE(value.eltype().isa<BShrTy>() &&
                  value.eltype().as<BShrTy>()->session() == session_,
              "Boolean value belongs to a different session/type");
  const auto width = value.eltype().as<BShrTy>()->nbits();
  // Every uint64_t is canonical at full width. Keep session/type validation,
  // but avoid a vacuous whole-array scan on the packed AND hot path.
  if (width == 64) {
    return;
  }
  const auto mask = BooleanMask(width);
  const NdArrayView<RepShare> view(value);
  for (int64_t i = 0; i < value.numel(); ++i) {
    SPU_ENFORCE(((view[i][0] | view[i][1]) & ~mask) == 0,
                "noncanonical Boolean share");
  }
}
NdArrayRef SpdzWiseState::publicBoolean(const NdArrayRef& value) const {
  requireRunning();
  RequirePublic(value);
  NdArrayRef out(makeType<BShrTy>(session_, 64), value.shape());
  const NdArrayView<uint64_t> in(value);
  for (int64_t i = 0; i < value.numel(); ++i) {
    out.data<RepShare>()[i] = {rank() == 0 ? in[i] : 0,
                               rank() == 2 ? in[i] : 0};
  }
  return out;
}
NdArrayRef SpdzWiseState::booleanBinary(const NdArrayRef& x,
                                        const NdArrayRef& y, bool is_and,
                                        bool public_rhs) {
  try {
    validateBoolean(x);
    SPU_ENFORCE(x.shape() == y.shape(), "Boolean shape mismatch");
    if (public_rhs) {
      RequirePublic(y);
    } else {
      validateBoolean(y);
    }
    if (is_and && !public_rhs) {
      return boolean_->andBB(x, y, session_);
    }
    const size_t xbits = x.eltype().as<BShrTy>()->nbits();
    const size_t ybits = public_rhs ? 64 : y.eltype().as<BShrTy>()->nbits();
    const size_t width =
        is_and ? std::min(xbits, ybits) : std::max(xbits, ybits);
    const auto mask = BooleanMask(width);
    NdArrayRef out(makeType<BShrTy>(session_, width), x.shape());
    const NdArrayView<RepShare> xv(x);
    for (int64_t i = 0; i < x.numel(); ++i) {
      RepShare rhs;
      if (public_rhs) {
        const auto word = NdArrayView<uint64_t>(y)[i];
        rhs = is_and ? RepShare{word, word}
                     : RepShare{rank() == 0 ? word : 0, rank() == 2 ? word : 0};
      } else {
        rhs = NdArrayView<RepShare>(y)[i];
      }
      for (size_t j = 0; j < 2; ++j) {
        out.data<RepShare>()[i][j] =
            (is_and ? xv[i][j] & rhs[j] : xv[i][j] ^ rhs[j]) & mask;
      }
    }
    return out;
  } catch (...) {
    abort();
    throw;
  }
}

namespace {

class CommonTypeB final : public Kernel {
 public:
  static constexpr const char* kBindName() { return "common_type_b"; }
  void evaluate(KernelEvalContext* ctx) const override {
    ctx->pushOutput(ctx->getState<SpdzWiseState>()->commonBooleanType(
        ctx->getParam<Type>(0), ctx->getParam<Type>(1)));
  }
};

class CastTypeB final : public CastTypeKernel {
 public:
  static constexpr const char* kBindName() { return "cast_type_b"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Type& to_type) const override {
    auto* state = ctx->getState<SpdzWiseState>();
    state->validateBoolean(in);
    state->commonBooleanType(in.eltype(), to_type);
    const auto mask = BooleanMask(to_type.as<BShrTy>()->nbits());
    NdArrayRef out(to_type, in.shape());
    const NdArrayView<RepShare> input(in);
    auto* output = out.data<RepShare>();
    pforeach(0, in.numel(), [&](int64_t i) {
      output[i] = {input[i][0] & mask, input[i][1] & mask};
    });
    return out;
  }
};

enum class ShiftKind { Left, LogicalRight, ArithmeticRight };

template <ShiftKind Op>
class BooleanShift final : public ShiftKernel {
 public:
  static constexpr const char* kBindName() {
    if constexpr (Op == ShiftKind::Left) return "lshift_b";
    if constexpr (Op == ShiftKind::LogicalRight) return "rshift_b";
    return "arshift_b";
  }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in,
                  const Sizes& bits) const override {
    auto* state = ctx->getState<SpdzWiseState>();
    state->validateBoolean(in);
    SPU_ENFORCE(bits.size() == 1 || int64_t(bits.size()) == in.numel(),
                "shift count shape mismatch");
    for (auto bit : bits) {
      SPU_ENFORCE(bit >= 0 && bit < 64, "shift count must be in [0,63]");
    }
    const auto width = in.eltype().as<BShrTy>()->nbits();
    size_t out_width = width;
    if constexpr (Op == ShiftKind::ArithmeticRight) {
      SPU_ENFORCE(width == 64, "arshift_b requires 64-bit Boolean input");
    } else if (in.numel()) {
      if constexpr (Op == ShiftKind::Left) {
        out_width = std::min<size_t>(
            64, width + *std::max_element(bits.begin(), bits.end()));
      } else {
        const auto shift = size_t(*std::min_element(bits.begin(), bits.end()));
        out_width = std::max<size_t>(1, width - std::min(width, shift));
      }
    }
    NdArrayRef out(makeType<BShrTy>(state->session(), out_width), in.shape());
    const NdArrayView<RepShare> input(in);
    auto* output = out.data<RepShare>();
    pforeach(0, in.numel(), [&](int64_t i) {
      const auto shift = bits.size() == 1 ? bits[0] : bits[i];
      for (size_t j = 0; j < 2; ++j) {
        const auto word = input[i][j];
        if constexpr (Op == ShiftKind::Left) {
          output[i][j] = word << shift;
        } else if constexpr (Op == ShiftKind::LogicalRight) {
          output[i][j] = word >> shift;
        } else {
          // Copy each share's bit 63. This map is XOR-linear, including
          // sign extension, and never needs the reconstructed sign.
          output[i][j] =
              shift == 0 ? word
                         : (word >> shift) |
                               ((uint64_t{0} - (word >> 63)) << (64 - shift));
        }
      }
    });
    return out;
  }
};

class BitrevB final : public BitrevKernel {
 public:
  static constexpr const char* kBindName() { return "bitrev_b"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in, size_t start,
                  size_t end) const override {
    auto* state = ctx->getState<SpdzWiseState>();
    state->validateBoolean(in);
    SPU_ENFORCE(start <= end && end <= 64, "invalid bit reversal interval");
    const auto width = std::max(in.eltype().as<BShrTy>()->nbits(), end);
    NdArrayRef out(makeType<BShrTy>(state->session(), width), in.shape());
    const NdArrayView<RepShare> input(in);
    auto* output = out.data<RepShare>();
    pforeach(0, in.numel(), [&](int64_t i) {
      output[i] = input[i];
      for (size_t bit = start; bit < end; ++bit) {
        const auto target = end - 1 - bit + start;
        for (size_t j = 0; j < 2; ++j) {
          output[i][j] = (output[i][j] & ~(uint64_t{1} << target)) |
                         (((input[i][j] >> bit) & 1) << target);
        }
      }
    });
    return out;
  }
};

class RandB final : public RandKernel {
 public:
  static constexpr const char* kBindName() { return "rand_b"; }
  NdArrayRef proc(KernelEvalContext* ctx, const Shape& shape) const override {
    return ctx->getState<SpdzWiseState>()->randomBoolean(shape);
  }
};

class P2B final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2b"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override {
    return ctx->getState<SpdzWiseState>()->publicBoolean(in);
  }
};
template <bool IsAnd, bool PublicRhs>
class BooleanBinary final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() {
    return IsAnd ? (PublicRhs ? "and_bp" : "and_bb")
                 : (PublicRhs ? "xor_bp" : "xor_bb");
  }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override {
    return ctx->getState<SpdzWiseState>()->booleanBinary(x, y, IsAnd,
                                                         PublicRhs);
  }
};

}  // namespace

void registerBooleanKernels(Object* object) {
  object->regKernel<P2B, BooleanBinary<true, false>, BooleanBinary<true, true>,
                    BooleanBinary<false, false>, BooleanBinary<false, true>,
                    CommonTypeB, CastTypeB, BooleanShift<ShiftKind::Left>,
                    BooleanShift<ShiftKind::LogicalRight>,
                    BooleanShift<ShiftKind::ArithmeticRight>, BitrevB, RandB>();
}

}  // namespace spu::mpc::spdzwise
