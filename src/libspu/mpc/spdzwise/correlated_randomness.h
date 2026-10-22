// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//

#pragma once

#include "libspu/mpc/spdzwise/ring.h"

namespace spu::mpc::spdzwise::ring {

// Scratch belongs to one truncation call. Reuse storage across tiles, but fill
// every live entry with fresh randomness before use. No PRG runs on workers.
struct ArithmeticState::TruncationBuffers {
  std::vector<Word> first, second;
  std::vector<uint64_t> private_words, coefficients;
  NdArrayRef inputs;

  TruncationBuffers(const Type& type, int64_t max_bits);
};

}  // namespace spu::mpc::spdzwise::ring
