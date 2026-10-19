"""Check the actual C++ F61 helpers against Python's arbitrary-precision ints.

This test requires a C++17 compiler, but no SPU/Bazel dependencies. It does not
test a cryptographic protocol, network execution, or malicious security.
Run: python3 src/libspu/mpc/spdzwise/field_reference_test.py
"""

import os
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
import unittest


P = 2**61 - 1
HALF = P // 2
MASK = 2**64 - 1


class FieldReferenceTest(unittest.TestCase):
    def test_cpp_against_bigints(self):
        # Fixed seed is for test vectors only, never protocol randomness.
        rng = random.Random(20261016)
        edges = [0, 1, 2, HALF, HALF + 1, P - 2, P - 1]
        pairs = [(a, b) for a in edges for b in edges]
        pairs += [(rng.randrange(P), rng.randrange(P)) for _ in range(10000)]
        rings = [0, 1, HALF, HALF + 1, 2**63, 2**63 - 1, MASK]
        rings += [rng.getrandbits(64) for _ in range(10000)]
        source = r'''
#include <iostream>
#include "field.h"
using namespace spu::mpc::spdzwise;
int main() {
  char op;
  uint64_t a, b;
  while (std::cin >> op >> a) {
    if (op == 'a') {
      std::cin >> b;
      std::cout << Add(a,b) << ' ' << Sub(a,b) << ' ' << Mul(a,b)
                << ' ' << Neg(a) << ' ' << ToRing(a) << '\n';
    } else if (op == 'r') {
      std::cout << FromRing(a) << ' ' << IsBoundedRing(a) << '\n';
    } else if (op == 'i') {
      std::cout << Inv(a) << '\n';
    }
  }
  try { Inv(0); return 1; } catch (const std::invalid_argument&) {}
  try { ToSigned(kPrime); return 2; } catch (const std::invalid_argument&) {}
  if (FromSigned(INT64_MIN) != FromRing(uint64_t{1} << 63)) return 3;
}
'''
        cases = [f"a {a} {b}" for a, b in pairs]
        expected = [
            [(a + b) % P, (a - b) % P, a * b % P, -a % P,
             (a if a <= HALF else a - P) & MASK]
            for a, b in pairs
        ]
        for bits in rings:
            signed = bits if bits < 2**63 else bits - 2**64
            cases.append(f"r {bits}")
            expected.append([signed % P, int(-HALF <= signed <= HALF)])
        for a in edges[1:] + [rng.randrange(1, P) for _ in range(100)]:
            cases.append(f"i {a}")
            expected.append([pow(a, -1, P)])
        with tempfile.TemporaryDirectory(prefix="spdzwise-field-") as temp:
            path = Path(temp)
            (path / "main.cc").write_text(source)
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            subprocess.run(
                compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            "-I", str(Path(__file__).resolve().parent),
                            str(path / "main.cc"), "-o", str(path / "check")],
                check=True,
            )
            result = subprocess.run([str(path / "check")],
                                    input="\n".join(cases) + "\n",
                                    text=True, capture_output=True, check=True)
        actual = [[int(x) for x in line.split()] for line in result.stdout.splitlines()]
        self.assertEqual(len(actual), len(expected))
        for case, got, want in zip(cases, actual, expected):
            self.assertEqual(got, want, case)


if __name__ == "__main__":
    unittest.main()
