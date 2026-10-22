// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Adapted from MP-SPDZ Protocols/{ReplicatedPrep,Spdz2kPrep}.hpp
// (see MP-SPDZ-LICENSE.txt).

#include "libspu/mpc/spdzwise/correlated_randomness.h"

#include <cstring>

#include "yacl/crypto/hash/blake3.h"
#include "yacl/link/algorithm/allgather.h"

#include "libspu/core/parallel_utils.h"

namespace spu::mpc::spdzwise::ring {
namespace {
void RequireAll(const std::shared_ptr<yacl::link::Context>& link, bool ok,
                const std::string& context) {
  for (size_t round = 0; round < 2; ++round) {
    const std::string status(1, ok ? '1' : '0');
    const auto all =
        yacl::link::AllGather(link, yacl::ByteContainerView(status),
                              context + "/status/" + std::to_string(round));
    for (const auto& item : all) ok &= std::string_view(item) == "1";
  }
  SPU_ENFORCE(ok, "SPDZWISE ring check failed (truncation)");
}

std::string Hash(std::string_view context, std::string_view bytes) {
  yacl::crypto::Blake3Hash hash;
  hash.Update("SPDZWISE-RING-TRUNC-v1/" + std::to_string(context.size()) + ":" +
              std::string(context));
  hash.Update(bytes);
  const auto digest = hash.CumulativeHash();
  return {reinterpret_cast<const char*>(digest.data()), digest.size()};
}

// Unlike a setup-time public PRG, this coin is unpredictable until the input
// records have been frozen. Commit/reveal and transcript agreement also bind
// a malicious sender that equivocates between recipients.
Word PublicCoin(const std::shared_ptr<yacl::link::Context>& link,
                const std::string& context) {
  const auto bytes = Encode({yacl::crypto::SecureRandSeed()});
  const auto commit = Hash(context + "/" + std::to_string(link->Rank()), bytes);
  const auto commits = yacl::link::AllGather(
      link, yacl::ByteContainerView(commit), context + "/commit");
  const auto reveals = yacl::link::AllGather(
      link, yacl::ByteContainerView(bytes), context + "/reveal");
  bool ok = true;
  std::string transcript;
  for (size_t rank = 0; rank < 3; ++rank) {
    const std::string_view reveal(reveals[rank]);
    ok &= reveal.size() == 16 &&
          std::string_view(commits[rank]) ==
              Hash(context + "/" + std::to_string(rank), reveal);
    transcript.append(reveal);
  }
  const auto digest = Hash(context + "/transcript", transcript);
  const auto digests = yacl::link::AllGather(
      link, yacl::ByteContainerView(digest), context + "/agree");
  for (const auto& item : digests) ok &= std::string_view(item) == digest;
  RequireAll(link, ok, context);
  Word seed = 0;
  for (const auto& reveal : reveals)
    seed ^= Decode(std::string_view(reveal), 1)[0];
  return seed;
}

AuthShare Public(Word x, size_t rank, const RepShare& key) {
  return {rank == 0 ? x : 0, rank == 2 ? x : 0, key[0] * x, key[1] * x};
}
}  // namespace

ArithmeticState::TruncationBuffers::TruncationBuffers(const Type& type,
                                                     int64_t max_bits)
    : first(2 * max_bits),
      second(2 * max_bits),
      private_words((max_bits + 63) / 64),
      inputs(type, {2 * max_bits}) {}

NdArrayRef ArithmeticState::privateRandomBits(size_t count, size_t owner,
                                              TruncationBuffers& buffers) {
  auto& [first, second, private_words, coefficients, storage] = buffers;
  const size_t total = 2 * count;
  first.resize(total);
  second.resize(total);
  private_words.resize((count + 63) / 64);
  auto out = storage.slice({0}, {static_cast<int64_t>(total)}, {1});
  auto* data = out.data<AuthShare>();
  const size_t rank = link_->Rank();
  if (rank == owner || rank == (owner + 1) % 3)
    trunc_prg_->fillPriv(absl::MakeSpan(private_words));
  trunc_prg_->fillPrssPair(first.data(), second.data(), total,
                           PrgState::GenPrssCtrl::Both);

  // MP-SPDZ replicated private input, batched for the two rotating owners.
  // Owner o sets (s_o,s_{o+1},s_{o+2})=(r,b-r,0). The pairwise r is already
  // known to o-1; only b-r must be sent to o+1. Neither non-owner knows b.
  // Both PRSS streams consume the full batch at every rank, including unused
  // components, so subsequent calls remain aligned through owner rotation.
  pforeach(0, total, [&](int64_t i) {
    const size_t input_owner = (owner + size_t(i) / count) % 3;
    if (rank == input_owner) {
      const size_t bit = size_t(i) % count;
      const Word value = (private_words[bit / 64] >> (bit % 64)) & 1;
      data[i] = {first[i], value - first[i], 0, 0};
      first[i] = data[i][1];
    } else if (link_->NextRank() == input_owner) {
      data[i] = {0, second[i], 0, 0};
    } else {
      data[i] = {};
    }
  });
  const auto context = tag_("trunc-private-input");
  // Queue both independent inputs before receiving either one.
  for (size_t input = 0; input < 2; ++input) {
    if (rank != (owner + input) % 3) continue;
    const auto tag = context + "/" + std::to_string(input);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    link_->SendAsync(link_->NextRank(),
                     yacl::ByteContainerView(first.data() + input * count,
                                             count * sizeof(Word)), tag);
#else
    const auto bytes = Encode(std::vector<Word>(
        first.begin() + input * count, first.begin() + (input + 1) * count));
    link_->SendAsync(link_->NextRank(), yacl::ByteContainerView(bytes), tag);
#endif
  }
  bool valid = true;
  for (size_t input = 0; input < 2; ++input) {
    const size_t input_owner = (owner + input) % 3;
    if (link_->PrevRank() != input_owner) continue;
    const auto received = link_->Recv(
        input_owner, context + "/" + std::to_string(input));
    const bool length_ok = size_t(received.size()) == count * sizeof(Word);
    valid &= length_ok;
    if (!length_ok) continue;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(second.data(), received.data(), count * sizeof(Word));
#else
    const auto decoded = Decode(std::string_view(received), count);
    std::copy(decoded.begin(), decoded.end(), second.begin());
#endif
    pforeach(0, count, [&](int64_t i) {
      data[input * count + i][0] = second[i];
    });
  }
  RequireAll(link_, valid, context + "/wire");
  // Freeze/check both inputs before authenticating; both input MAC records
  // remain immutable until the same deferred session check as before.
  CheckCopies(link_, out, tag_("trunc-input-copies"), false);
  mask_prg_->fillPrssPair(first.data(), second.data(), total,
                          PrgState::GenPrssCtrl::Both);
  pforeach(0, total, [&](int64_t i) {
    first[i] += LocalProduct({data[i][0], data[i][1]}, key_) - second[i];
  });
  exchange(first, second, "trunc-input-mac");
  pforeach(0, total, [&](int64_t i) {
    data[i][2] = first[i];
    data[i][3] = second[i];
  });
  remember(out);
  return out;
}

void ArithmeticState::checkTruncBits(
    const NdArrayRef& bits, std::vector<uint64_t>& coefficients) {
  const NdArrayView<AuthShare> b(bits);
  const auto context = tag_("trunc-bitness");
  CheckCopies(link_, bits, context + "/freeze", true);
  const auto seed = PublicCoin(link_, session_ + "/" + context);
  coefficients.resize(bits.numel());
  yacl::crypto::FillPRand(PrgState::kAesType, seed, 0, 0,
                          absl::MakeSpan(coefficients));
  const auto one = Public(1, link_->Rank(), key_);
  // MP-SPDZ's randomized dot product sum_i r_i * (1-b_i) * b_i.
  // Accumulate both the value and MAC products, then remember the result.
  const auto sums = yacl::parallel_reduce<std::array<Word, 2>>(
      0, bits.numel(), kMinTaskSize,
      [&](int64_t begin, int64_t end) {
        std::array<Word, 2> sum{};
        for (auto i = begin; i < end; ++i) {
          const RepShare rhs{b[i][0] * coefficients[i],
                             b[i][1] * coefficients[i]};
          sum[0] += LocalProduct({one[0] - b[i][0], one[1] - b[i][1]}, rhs);
          sum[1] += LocalProduct({one[2] - b[i][2], one[3] - b[i][3]}, rhs);
        }
        return sum;
      },
      [](const auto& a, const auto& b) {
        return std::array<Word, 2>{a[0] + b[0], a[1] + b[1]};
      });
  const auto dot = reshare({sums[0], sums[1]}, "trunc-bitness-dot");
  pending_.push_back({dot[0][0], dot[0][1], dot[1][0], dot[1][1]});
  // Honest generated bits are bits in the full internal ring. Use the existing
  // full-128-bit zero test (stronger than checking only the low application
  // word), without ever opening the extension or the MAC residual.
  CheckZero(link_, dot[0], context + "/zero");
}

NdArrayRef ArithmeticState::randomTruncBits(size_t count,
                                            TruncationBuffers& buffers) {
  if (!trunc_prg_) trunc_prg_ = std::make_unique<PrgState>(link_);
  const auto inputs = privateRandomBits(count, trunc_input_owner_, buffers);
  const auto n = static_cast<int64_t>(count);
  auto a = inputs.slice({0}, {n}, {1});
  const auto b = inputs.slice({n}, {2 * n}, {1});
  trunc_input_owner_ = (trunc_input_owner_ + 1) % 3;
  const auto ab = multiply(a, b);
  auto* out = a.data<AuthShare>();
  const auto* bv = b.data<AuthShare>();
  const auto* product = ab.data<AuthShare>();
  pforeach(0, count, [&](int64_t i) {
    for (size_t c = 0; c < 4; ++c) out[i][c] += bv[i][c] - 2 * product[i][c];
  });
  checkTruncBits(a, buffers.coefficients);
  return a;
}

}  // namespace spu::mpc::spdzwise::ring
