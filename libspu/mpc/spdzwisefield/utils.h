#pragma once

#include "libspu/core/type.h"

class MersennePrimeField {
 public:
  static const uint32_t PRIME_EXP = 61;
  static const uint128_t PR = (1llu << PRIME_EXP) - 1;
  static uint64_t modp(uint64_t);
  static uint64_t modp(uint128_t);
  static uint64_t neg(uint64_t);

  template <typename... Args>
  static uint64_t add(Args...);

  static uint64_t sub(uint64_t, uint64_t);
  static uint64_t mul(uint64_t, uint64_t);
  static uint64_t inverse(uint64_t a);
};