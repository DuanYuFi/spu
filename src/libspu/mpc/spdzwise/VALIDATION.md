# Development validation record

## Current Boolean validation — 2026-10-20

The current C++ sources previously passed 276 native cases: Boolean 16,
SPDZwise arithmetic 9, SPDZwise IO 7, ProtocolSession 7, ABY3 protocol 225 and
ABY3 IO 12. The four unaffected targets used valid Bazel caches. Boolean also
passed with YACL_NUM_THREADS=1 and 12 before a whitespace-only formatting step.
Coverage includes all bucket sizes, packing widths/overflow occupancies,
strided and empty arrays, tile boundaries and tails, serial PRSS and shuffle
oracles, random-stream continuation, all-party replica/relationship/digest/
commitment/opening faults, and session/output lifecycle rules. These tests do
not constitute a full malicious-security proof.

[Retained test evidence](benchmarks/boolean-c2-latest/validation/) includes XML,
commands/logs and source provenance. C++ and BUILD fingerprints still match;
this experiment refresh changed only the runner and documentation, and did
not rerun unit tests. Intermediate Boolean performance records were removed
at the user's request; arithmetic validation below is retained.

The [latest C2 experiment](benchmarks/boolean-c2-latest/RESULTS.md) rebuilt the
current benchmark once and reran native ABY3 and ABFL sequentially using the
same binary, inputs, FactoryMem transport and YACL_NUM_THREADS=12. Each ran one
warmup and seven measured repetitions, checked all 65536 outputs per iteration,
and passed gate-count, check-count and payload assertions. Source and binary
fingerprints match. No previous performance samples are pooled into the new
results. Framework PRGs, communication, thread pool and protocol code were
not changed by this refresh. LAN/WAN behavior was not measured.

## Earlier arithmetic validation

Date: 2026-10-19. This validation record covers the two-input F61 circuit, now
archived in
[`benchmarks/macbook-air-f61-2026-10-19/`](benchmarks/macbook-air-f61-2026-10-19/).
Commands for all checks below
are in the root [`readme-malicious-ppml.md`](../../../../readme-malicious-ppml.md).

## Current correctness checks

The optimized Bazel 7.7.1 build passed and all four selected native test targets
executed successfully (not cached test results):

| Target | Cases | Failures |
|---|---:|---:|
| `//libspu/mpc/spdzwise:protocol_test` | 9 | 0 |
| `//libspu/mpc/spdzwise:io_test` | 7 | 0 |
| `//libspu/mpc:protocol_session_test` | 7 | 0 |
| `//libspu/kernel/hal:ring_test` | 4 | 0 |

Counts were checked against Bazel `test.xml` reports. Coverage includes F61
wire canonicalization; PRSS duplicate alignment for empty/odd batches; input,
arithmetic and matrix-product correctness; authenticated reuse; IO range,
metadata, serialization and duplicate-share checks; transactional exports;
terminal failure; HAL comparison dispatch and legacy fallback.

MAC corruption tests cover all three server positions and different positions
in a 65,537-element batch. Verification asserts 5 sends and 64 application
payload bytes per party and no extra nonempty check during export.

Both Python correctness suites were rerun:

- `field_reference_test.py`: 1 test passed, compiling the actual C++ F61 header
  with UBSan and comparing 20,162 operation inputs with Python big integers.
- `reference_test.py`: 7 tests passed for plaintext decomposition, signed
  conversion, floor truncation and range accounting. These are numerical
  oracles, not secret conversion protocol executions.

## Current circuit validation

Built `//libspu/mpc/spdzwise:circuit_bench` with the current merged verification
dot-product implementation. Each of W×D = 65,536×16, 1,048,576×1 and 16,384×64
ran one warmup and seven measured repetitions for both protocols. All runs
passed final-vector reconstruction; every party/sample passed input count,
gate count, queue length, final-only check count and phase payload assertions.
The summarizer validated all three data sets. Results and source fingerprints
are retained in [`benchmarks/macbook-air-f61-2026-10-19/`](benchmarks/macbook-air-f61-2026-10-19/).
The target above is its historical name; its current build entry is
`//:circuit_bench` in the independent `benchmarks` workspace.

## Outstanding acceptance

- Full device execution, Python binding build and broader old-protocol regression.
- Input authentication, PRSS scheduling and arithmetic-check composition review.
- Client-issued request binding and rejection of complete old output bundles;
  authenticated confidential channel deployment and client authorization.
- ABFL security-bound/composition review, malicious daBits and dependency-safe conversion openings.
- Cryptographic A2B/B2A, exact truncation, comparison and nonlinear HAL kernels.
- LR, tree and CNN execution, numerical/range/fault tests and measurements.
- Per-procedure security bounds and whole-execution composition accounting.

The production factory continues to reject `SPDZWISE`. Correctness and local
performance validation do not establish complete malicious-security acceptance.
