// Copyright 2026 The SPU Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <stdexcept>

namespace spu::mpc::spdzwise {

// Scalar arithmetic only. Secret-dependent decisions must be implemented by
// MPC kernels, never by decoding a share with these plaintext helpers.
inline constexpr uint64_t kPrime = (uint64_t{1} << 61) - 1;
inline constexpr uint64_t kHalfPrime = kPrime / 2;
inline constexpr unsigned kFieldBits = 61;
inline constexpr unsigned kStatisticalSecurity = 40;

inline constexpr uint64_t Add(uint64_t x, uint64_t y) {
  const uint64_t z = x + y;  // canonical operands: sum fits in 62 bits
  return z >= kPrime ? z - kPrime : z;
}

inline constexpr uint64_t Sub(uint64_t x, uint64_t y) {
  return x >= y ? x - y : kPrime - (y - x);
}

inline constexpr uint64_t Neg(uint64_t x) { return x == 0 ? 0 : kPrime - x; }

inline constexpr uint64_t Mul(uint64_t x, uint64_t y) {
  const __uint128_t z = static_cast<__uint128_t>(x) * y;
  const uint64_t folded = static_cast<uint64_t>(z & kPrime) +
                          static_cast<uint64_t>(z >> kFieldBits);
  return folded >= kPrime ? folded - kPrime : folded;
}

inline constexpr uint64_t Pow(uint64_t x, uint64_t n) {
  uint64_t result = 1;
  while (n != 0) {
    if (n & 1) {
      result = Mul(result, x);
    }
    x = Mul(x, x);
    n >>= 1;
  }
  return result;
}

inline uint64_t Inv(uint64_t x) {
  if (x == 0 || x >= kPrime) {
    throw std::invalid_argument("F61 inverse requires a nonzero residue");
  }
  return Pow(x, kPrime - 2);
}

// Map an arbitrary signed 64-bit integer to the field without signed overflow,
// including INT64_MIN. Input validation is a separate bounded-number contract.
inline constexpr uint64_t FromSigned(int64_t x) {
  if (x >= 0) {
    return static_cast<uint64_t>(x) % kPrime;
  }
  const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(x);
  return Neg(magnitude % kPrime);
}

// Interpret SPU's uint64_t buffer as two's complement, not as an unsigned field
// element. In particular, the SPU encoding of -1 must map to p-1, not 7.
inline constexpr uint64_t FromRing(uint64_t bits) {
  if ((bits >> 63) == 0) {
    return bits % kPrime;
  }
  return Neg((uint64_t{0} - bits) % kPrime);
}

inline int64_t ToSigned(uint64_t x) {
  if (x >= kPrime) {
    throw std::invalid_argument("noncanonical F61 residue");
  }
  return x <= kHalfPrime ? static_cast<int64_t>(x)
                         : -static_cast<int64_t>(kPrime - x);
}

inline uint64_t ToRing(uint64_t x) {
  return static_cast<uint64_t>(ToSigned(x));
}

inline bool IsBoundedRing(uint64_t bits) {
  return bits <= kHalfPrime || bits >= uint64_t{0} - kHalfPrime;
}

}  // namespace spu::mpc::spdzwise
