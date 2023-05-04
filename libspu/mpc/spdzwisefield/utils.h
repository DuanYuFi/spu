#pragma once

#include "libspu/core/type.h"

class MersennePrimeField {
 public:
  static const uint32_t PRIME_EXP = 61;
  static const uint128_t PR = (1LLU << PRIME_EXP) - 1;
  static uint64_t modp(uint64_t);
  static uint64_t modp128(uint128_t);
  static uint64_t neg(uint64_t);

  static uint64_t add(uint64_t, uint64_t);
  static uint64_t add(uint64_t, uint64_t, uint64_t);
  static uint64_t add(uint64_t, uint64_t, uint64_t, uint64_t);

  static uint64_t sub(uint64_t, uint64_t);
  static uint64_t mul(uint64_t, uint64_t);
  static uint64_t inverse(uint64_t a);
};

class MersennePrimeField127 {
 public:
  static const uint32_t PRIME_EXP = 127;
  static const uint128_t PR = (static_cast<uint128_t>(1) << PRIME_EXP) - 1;
  static uint128_t modp(uint128_t);
  static uint128_t neg(uint128_t);

  static uint128_t add(uint128_t, uint128_t);
  static uint128_t add(uint128_t, uint128_t, uint128_t);
  static uint128_t add(uint128_t, uint128_t, uint128_t, uint128_t);

  static uint128_t sub(uint128_t, uint128_t);
  static uint128_t mul(uint128_t, uint128_t);
  static uint128_t inverse(uint128_t a);
};