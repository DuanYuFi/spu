
#include "libspu/mpc/spdzwisefield/utils.h"

uint64_t MersennePrimeField::modp(uint64_t a) {
  uint64_t res = (a >> PRIME_EXP) + (a & PR);
  if (res >= PR) {
    res -= PR;
  }
  return res;
}

uint64_t MersennePrimeField::modp128(uint128_t a) {
  uint64_t higher, middle, lower;
  higher = (a >> (2 * PRIME_EXP));
  middle = (a >> PRIME_EXP) & PR;
  lower = a & PR;
  return modp(higher + middle + lower);
}

uint64_t MersennePrimeField::neg(uint64_t a) {
  assert(a < PR);
  if (a > 0) {
    return PR - a;
  } else {
    return 0;
  }
}

uint64_t MersennePrimeField::add(uint64_t a, uint64_t b) {
  uint64_t res = a + b;
  if (res >= PR) {
    res -= PR;
  }
  return res;
}

uint64_t MersennePrimeField::add(uint64_t a, uint64_t b, uint64_t c) {
  return add(a, add(b, c));
}

uint64_t MersennePrimeField::add(uint64_t a, uint64_t b, uint64_t c,
                                 uint64_t d) {
  return add(add(a, b), add(c, d));
}

uint64_t MersennePrimeField::sub(uint64_t a, uint64_t b) {
  if (a >= b) {
    return a - b;
  } else {
    return PR - b + a;
  }
}

uint64_t MersennePrimeField::mul(uint64_t a, uint64_t b) {
  uint128_t res = (static_cast<uint128_t>(a)) * (static_cast<uint128_t>(b));
  uint64_t higher = (res >> PRIME_EXP);
  uint64_t lower = res & PR;
  return add(higher, lower);
}

uint64_t MersennePrimeField::inverse(uint64_t a) {
  uint64_t left = a;
  uint64_t right = PR;
  uint64_t x = 1, y = 0, u = 0, v = 1;
  uint64_t w, z;
  while (left != 0) {
    w = right / left;
    z = right % left;
    right = left;
    left = z;

    z = u - w * x;
    u = x;
    x = z;

    z = v - w * y;
    v = y;
    y = z;
  }
  if (u >= PR) {
    u += PR;
  }
  return u;
}