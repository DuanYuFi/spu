// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/spdzwise/type.h"

namespace spu::mpc::spdzwise {

// Pairwise secret PRG streams: second_i == first_(i+1). No online messages.
void FillF61PrssPair(PrgState* prg, std::vector<uint64_t>& first,
                     std::vector<uint64_t>& second);
std::vector<RepShare> RandomF61Rep(PrgState* prg, size_t size);

// Canonical little-endian wire format, with a memcpy fast path on LE hosts.
std::string EncodeF61(const std::vector<uint64_t>& words);
std::vector<uint64_t> DecodeF61(std::string_view bytes, size_t count);
std::vector<uint64_t> ExchangeF61(
    const std::shared_ptr<yacl::link::Context>& link,
    const std::vector<uint64_t>& words, std::string_view tag);

inline uint64_t LocalF61Product(const RepShare& x, const RepShare& y) {
  return Add(Add(Mul(x[0], y[0]), Mul(x[0], y[1])), Mul(x[1], y[0]));
}

// The same semi-honest F61 primitive serves SPDZwise and the ABY3-F61
// benchmark. Accessors avoid packing/copying authenticated operands. The
// callbacks touch disjoint elements; PRG state is consumed before parallelism.
template <typename Left, typename Right, typename Output>
void MultiplyF61(PrgState* masks,
                 const std::shared_ptr<yacl::link::Context>& link, size_t size,
                 Left&& left, Right&& right, Output&& output,
                 std::string_view tag) {
  std::vector<uint64_t> first(size), second(size);
  FillF61PrssPair(masks, first, second);
  pforeach(0, size, [&](int64_t i) {
    first[i] =
        Add(LocalF61Product(left(i), right(i)), Sub(first[i], second[i]));
  });
  second = ExchangeF61(link, first, tag);
  pforeach(0, size,
           [&](int64_t i) { output(i, RepShare{first[i], second[i]}); });
}

}  // namespace spu::mpc::spdzwise
