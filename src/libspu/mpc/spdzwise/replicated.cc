// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "libspu/mpc/spdzwise/replicated.h"

#include <cstring>

namespace spu::mpc::spdzwise {

void FillF61PrssPair(PrgState* prg, std::vector<uint64_t>& first,
                     std::vector<uint64_t>& second) {
  SPU_ENFORCE(first.size() == second.size(), "PRSS size mismatch");
  if (first.empty()) {
    return;
  }
  prg->fillPrssPair(first.data(), second.data(), first.size(),
                    PrgState::GenPrssCtrl::Both);
  // Masking gives a uniform 61-bit word. Reject p rather than reducing it
  // to zero. Each edge consumes its own rejection draws, so adjacent holders
  // remain aligned even when the two streams reject different positions.
  auto reduce = [&](std::vector<uint64_t>& words, PrgState::GenPrssCtrl ctrl) {
    std::vector<size_t> rejected;
    for (size_t i = 0; i < words.size(); ++i) {
      words[i] &= kPrime;
      if (words[i] == kPrime) {
        rejected.push_back(i);
      }
    }
    while (!rejected.empty()) {
      std::vector<uint64_t> fresh(rejected.size());
      prg->fillPrssPair(
          ctrl == PrgState::GenPrssCtrl::First ? fresh.data() : nullptr,
          ctrl == PrgState::GenPrssCtrl::Second ? fresh.data() : nullptr,
          fresh.size(), ctrl);
      std::vector<size_t> again;
      for (size_t j = 0; j < rejected.size(); ++j) {
        words[rejected[j]] = fresh[j] & kPrime;
        if (words[rejected[j]] == kPrime) {
          again.push_back(rejected[j]);
        }
      }
      rejected.swap(again);
    }
  };
  reduce(first, PrgState::GenPrssCtrl::First);
  reduce(second, PrgState::GenPrssCtrl::Second);
}

std::vector<RepShare> RandomF61Rep(PrgState* prg, size_t size) {
  std::vector<uint64_t> first(size), second(size);
  FillF61PrssPair(prg, first, second);
  std::vector<RepShare> result(size);
  pforeach(0, size, [&](int64_t i) { result[i] = {first[i], second[i]}; });
  return result;
}

std::string EncodeF61(const std::vector<uint64_t>& words) {
  std::string bytes(words.size() * sizeof(uint64_t), '\0');
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (!words.empty()) {
    std::memcpy(bytes.data(), words.data(), bytes.size());
  }
#else
  for (size_t i = 0; i < words.size(); ++i) {
    for (size_t b = 0; b < 8; ++b) {
      bytes[i * 8 + b] = static_cast<char>(words[i] >> (8 * b));
    }
  }
#endif
  return bytes;
}

std::vector<uint64_t> DecodeF61(std::string_view bytes, size_t count) {
  SPU_ENFORCE(bytes.size() == count * sizeof(uint64_t),
              "F61 message size mismatch");
  std::vector<uint64_t> words(count);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (count != 0) {
    std::memcpy(words.data(), bytes.data(), bytes.size());
  }
#else
  for (size_t i = 0; i < count; ++i) {
    for (size_t b = 0; b < 8; ++b) {
      words[i] |= uint64_t{static_cast<unsigned char>(bytes[i * 8 + b])}
                  << (8 * b);
    }
  }
#endif
  // Throw on the calling thread; parallel worker exceptions are not portable.
  SPU_ENFORCE(std::all_of(words.begin(), words.end(),
                          [](uint64_t x) { return x < kPrime; }),
              "noncanonical F61 message");
  return words;
}

std::vector<uint64_t> ExchangeF61(
    const std::shared_ptr<yacl::link::Context>& link,
    const std::vector<uint64_t>& words, std::string_view tag) {
  const auto bytes = EncodeF61(words);
  link->SendAsync(link->PrevRank(), yacl::ByteContainerView(bytes), tag);
  const auto received = link->Recv(link->NextRank(), tag);
  return DecodeF61(std::string_view(received), words.size());
}

}  // namespace spu::mpc::spdzwise
