"""Tests of the plaintext numerical contract; these are not MPC executions."""

import random
import unittest

from reference import (HALF, P, BoundedFixedPoint, a2b, b2a, centered,
                       encode_integer, residue_bits_from_mask, trunc)


class NumericContractTest(unittest.TestCase):
    def test_full_decomposition_exhaustive_small_primes(self):
        for p in [3, 7, 31, 127]:
            for value in range(p):
                for mask in range(p):
                    bits = residue_bits_from_mask(value, mask, p)
                    self.assertEqual(sum(b << i for i, b in enumerate(bits)), value)

    def test_masked_open_is_uniform_for_small_primes(self):
        for p in [7, 31, 127]:
            for value in range(p):
                self.assertEqual(sorted((value - r) % p for r in range(p)), list(range(p)))

    def test_centered_representation_and_round_trip(self):
        rng = random.Random(20261016)
        values = [-HALF, -HALF + 1, -1, 0, 1, HALF - 1, HALF]
        values += [rng.randrange(-HALF, HALF + 1) for _ in range(10000)]
        for value in values:
            residue = encode_integer(value)
            self.assertEqual(centered(residue), value)
            self.assertEqual(b2a(a2b(residue)), residue)
            self.assertEqual(a2b(residue)[63], int(value < 0))

    def test_floor_truncation_including_negative_boundaries(self):
        for shift in [0, 1, 8, 16, 31, 60, 63]:
            for value in [-HALF, -65537, -65536, -65535, -1, 0, 1, 65535, 65536, HALF]:
                self.assertEqual(centered(trunc(encode_integer(value), shift)), value // 2**shift)

    def test_short_bits_are_unsigned(self):
        self.assertEqual(b2a([1]), 1)
        self.assertEqual(b2a([1] * 8), 255)
        self.assertEqual(b2a([1] * 64), P - 1)

    def test_invalid_values_are_rejected(self):
        for value in [-HALF - 1, HALF + 1]:
            with self.assertRaises(ValueError):
                encode_integer(value)
        for bits in [[], [2], [1] * 65]:
            with self.assertRaises(ValueError):
                b2a(bits)
        with self.assertRaises(ValueError):
            residue_bits_from_mask(0, P)

    def test_range_tracking_detects_pretruncation_overflow(self):
        reference = BoundedFixedPoint()
        self.assertEqual(reference.multiply(65536, -98304), -98304)
        self.assertEqual(reference.dot([65536, 32768], [65536, -65536]), 32768)
        self.assertEqual(reference.max_product, 65536 * 98304)
        with self.assertRaises(ValueError):
            reference.multiply(1 << 40, 1 << 30)


if __name__ == "__main__":
    unittest.main()
