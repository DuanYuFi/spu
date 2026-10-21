"""Differentially test the actual native-128 primitive with Python big integers.

Runs a dependency-free C++ executable under UBSan. This checks arithmetic,
not malicious security. Run: python3 src/libspu/mpc/spdzwise/ring_reference_test.py
"""
import os
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
import unittest

MASK64 = 2**64 - 1
MASK128 = 2**128 - 1


class RingReferenceTest(unittest.TestCase):
    def test_cpp_against_bigints(self):
        rng = random.Random(20261021)  # Test vectors only.
        edges = [0, 1, 2**64 - 1, 2**64, 2**64 + 1,
                 2**127 - 1, 2**127, MASK128]
        cases = [(a, b, c, d) for a in edges for b in edges
                 for c in edges for d in edges]
        cases += [tuple(rng.getrandbits(128) for _ in range(4))
                  for _ in range(10000)]
        source = r'''
#include <iostream>
#include "ring_math.h"
using namespace spu::mpc::spdzwise::ring;
void print(Word x) {
  std::cout << uint64_t(x >> 64) << ' ' << uint64_t(x) << ' ';
}
int main() {
  uint64_t hi, lo;
  while (std::cin >> hi >> lo) {
    Word a = (Word(hi) << 64) | lo;
    std::cin >> hi >> lo; Word b = (Word(hi) << 64) | lo;
    std::cin >> hi >> lo; Word c = (Word(hi) << 64) | lo;
    std::cin >> hi >> lo; Word d = (Word(hi) << 64) | lo;
    print(a+b); print(a-b); print(a*b); print(Word(0)-a);
    print(LocalProduct({a,b},{c,d}));
    std::cout << '\n';
  }
}
'''
        wire = "\n".join(" ".join(f"{x >> 64} {x & MASK64}" for x in case)
                         for case in cases) + "\n"
        with tempfile.TemporaryDirectory(prefix="spdzwise-ring-") as temp:
            path = Path(temp)
            (path / "main.cc").write_text(source)
            subprocess.run(
                shlex.split(os.environ.get("CXX", "c++")) +
                ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=undefined", "-fno-sanitize-recover=all",
                 "-I", str(Path(__file__).resolve().parent),
                 str(path / "main.cc"), "-o", str(path / "check")],
                check=True)
            result = subprocess.run([str(path / "check")], input=wire,
                                    text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), len(cases))
        for (a, b, c, d), line in zip(cases, lines):
            words = [int(x) for x in line.split()]
            actual = [(words[i] << 64) | words[i+1] for i in range(0, 10, 2)]
            expected = [a+b, a-b, a*b, -a, a*c+a*d+b*c]
            self.assertEqual(actual, [x & MASK128 for x in expected])


if __name__ == "__main__":
    unittest.main()
