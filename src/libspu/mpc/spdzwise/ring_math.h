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

#include <array>
#include <cstdint>

namespace spu::mpc::spdzwise::ring {

inline constexpr unsigned kValueBits = 64;
inline constexpr unsigned kExtensionBits = 64;
using Word = __uint128_t;
static_assert(sizeof(Word) * 8 == kValueBits + kExtensionBits);
using RepShare = std::array<Word, 2>;
using AuthShare = std::array<Word, 4>;

// Unsigned wraparound is reduction modulo 2^128, including multiplication.
inline constexpr Word LocalProduct(const RepShare& x, const RepShare& y) {
  return x[0] * (y[0] + y[1]) + x[1] * y[0];
}

}  // namespace spu::mpc::spdzwise::ring
