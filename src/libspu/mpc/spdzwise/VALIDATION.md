# Development validation record

## Zero-padded ABFL and opening checkpoints — 2026-10-22

`SpdzWiseState::checkMac()` is the arithmetic MAC checkpoint;
`checkBoolean()` exposes the session-bound ABFL checkpoint with terminal
failure handling, and `verify()` flushes both queues without ending execution.
Finalization, masked B2A opening and masked probabilistic-truncation opening use
this combined interface. Truncation verifies newly generated mask records as
well as earlier work before sending its opening payload. Internal checking
openings retain their own protocol schedule. Public and specific-party reveal
kernels remain closed; regression tests exercise all six denied kernel names
with pending work in both arithmetic domains.

For a nonempty Boolean queue, the verifier appends public zero online anchors
to reach 1024 packed words when needed, chooses B=5, and still generates 4099
fresh random companions. Dummy anchors participate in pre-coin copy agreement
and every sacrifice column. The versioned batch agreement binds real bits,
real words, checking words, bucket size, overflow and call layout. Padding is
local to checking and does not change pending-bit or B2A gate-work statistics.
Empty ABFL checks remain no-ops. RING arithmetic retains its empty-queue count
agreement without repeating the cryptographic check.

All seven optimized-build regression targets passed (80 test cases):

| Target | Cases | Failures |
| --- | ---: | ---: |
| `//libspu/mpc/spdzwise:boolean_test` | 17 | 0 |
| `//libspu/mpc/spdzwise:protocol_test` | 13 | 0 |
| `//libspu/mpc/spdzwise:ring_conversion_test` | 11 | 0 |
| `//libspu/mpc/spdzwise:ring_truncation_test` | 9 | 0 |
| `//libspu/mpc/spdzwise:ring_test` | 15 | 0 |
| `//libspu/mpc/spdzwise:io_test` | 8 | 0 |
| `//libspu/mpc:protocol_session_test` | 7 | 0 |

New and expanded cases cover 1/7/32/64-bit inputs, single-element and
1023-word batches, partial-word padding, repeated checkpoints, no-output
executions, unchanged work counters, and online/random-pool faults at all three
party positions. B2A accepts standalone small inputs and rejects faulty MACs,
ANDs and tail lanes before masked opening. Truncation observes both queues empty
at every opening and rejects pending MAC/Boolean faults before that boundary.
The run used the command below from `src`; the last invocation reused six
passing target caches and executed the corrected protocol target.

```sh
libspu/mpc/spdzwise/benchmarks/tools/bin/bazel-7.7.1 \
  --batch --output_user_root=/tmp/malicious-spu-bazel test -c opt \
  //libspu/mpc/spdzwise:boolean_test //libspu/mpc/spdzwise:protocol_test \
  //libspu/mpc/spdzwise:ring_conversion_test \
  //libspu/mpc/spdzwise:ring_truncation_test //libspu/mpc/spdzwise:ring_test \
  //libspu/mpc/spdzwise:io_test //libspu/mpc:protocol_session_test \
  --lockfile_mode=off --jobs=16 --test_env=YACL_NUM_THREADS=4 \
  --test_timeout=180 \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain
```

This supersedes the small-final-batch rejection and deferred truncation-mask
authentication described in earlier records. Existing benchmark results are
historical and were not rerun. The prototype's outstanding whole-execution
security/composition review remains unchanged; functional and fault-injection
tests do not certify the complete protocol's statistical security bound.

## RING split-3 A2B — 2026-10-22

Registered value-only RING/FM64 `a2b`, preserving shape/dtype/session. Compared
CSA + bit-sliced RCA, pruned Brent–Kung, word-packed Kogge–Stone, and the native
ABY3 A2B on identical full-width 100K inputs. RCA won by more than the 3% tie
threshold; the other implementations and candidate switches were deleted.
Final optimized build and six regression targets passed 61 cases (the unchanged
session target was cached). Conversion tests cover shape/stride/tails, arithmetic
and Boolean composition, session rejection, small-batch accumulation and six
corrupt-triple executions across all parties. Both final benchmark protocols
verified every output in one warmup and seven measured iterations.

Final median total: SPDZwise 134.182 ms, native ABY3 146.386 ms; ratio 0.917x.
SPDZwise conversion/verification medians: 12.950/121.232 ms. Main timing includes
one final ABFL verification; input authentication and output reconstruction are
excluded. Three-party sent bytes: 187,567,008 vs 33,600,000 (5.582x). These are
FactoryMem results with 64 background workers; phase medians are independent.
The existing ABFL security/composition premises and small-final-batch threshold
remain unchanged.

[Selection, raw samples, final source fingerprints and test evidence](benchmarks/a2b-split3-100k-2026-10-22/RESULTS.md).


## Truncation selector — 2026-10-22

Renamed the arithmetic-bit implementation to `TruncPrRandomBit` and its state
methods to `truncPrRandomBit`. The registered `trunc_a` kernel is now
`TruncASelector`, which currently routes to that sole implementation. The
algorithm, checks, dtype preservation and timing logic are unchanged.

Four uncached regression targets passed 40 cases with `YACL_NUM_THREADS=65`:
`ring_truncation_test` (7), `ring_test` (15), `protocol_test` (11) and
`protocol_session_test` (7). The truncation tests assert the registered selector
and its rounding/error metadata while exercising the existing numerical,
probability, dtype/shape and authentication checks through `trunc_a`.
The benchmark rebuilt successfully with its selector type assertion, and a
1,000-input smoke run (one warmup plus one measured iteration) validated all
outputs and final authentication. No new 100K performance claim is made;
the retained 100K archive below keeps its original source fingerprints.


## Signed RING truncation optimization — 2026-10-22

The final optimization build passed all eight regression targets, 301 cases
without failures, errors, skips or disabled tests. This adds owner rotation
through a partial tile, immutable authentication queue counts, canonical
streamed authenticated-copy hashes, and 27 transport fault executions across
all three ranks. Numerical/probability, output MAC, subsequent authenticated
multiplication, IO, session lifecycle and native ABY3 regressions passed.

The retained final 100K/m=16 measurement used 64 background workers, one
warmup and seven measured iterations. Both protocols validated every output
and completed final authentication. SPDZwise median: 3431.204 ms; native ABY3
TruncAPr2: 8.728 ms. Three-party primary traffic: 1,436,027,195 and 2,800,000 bytes.
Fresh material and internal checks are timed; final authentication, export and
reconstruction are outside the primary interval. Phase medians cannot be summed.

Full measurements, test XML/logs, source and binary fingerprints, source.patch
and commands are in [the final archive](benchmarks/truncation-signed-100k-optimized-2026-10-22/RESULTS.md).
Earlier baseline and intermediate experiment archives have been removed.
The protocol checks, 4096-input tiles and deferred-record count are preserved;
this does not remove the remaining composition/production acceptance gates.

## RING / Protocol 9 implementation — 2026-10-21

Configuration stays FM64; internal arithmetic is unsigned Z/(2^128).
`ring_reference_test.py` compiled the actual C++ scalar primitive with
`-fsanitize=undefined -fno-sanitize-recover=all` and compared 14,096 edge/random
vectors against Python arbitrary-precision integers: passed. The existing
FIELD differential/UBSan test also passed. These checks cover scalar arithmetic,
not the full linked protocol or its cryptographic security.

Six optimized native targets passed, totaling **57 cases** (counts checked
against Bazel XML reports):

| Target | Cases | Failures |
|---|---:|---:|
| `//libspu/mpc/spdzwise:ring_test` | 11 | 0 |
| `//libspu/mpc/spdzwise:protocol_test` | 11 | 0 |
| `//libspu/mpc/spdzwise:io_test` | 8 | 0 |
| `//libspu/mpc/spdzwise:boolean_test` | 16 | 0 |
| `//libspu/mpc:protocol_session_test` | 7 | 0 |
| `//libspu/core:config_test` | 4 | 0 |

The RING cases include all 128 single-bit nonzero residuals, zero/carry/random
shares, full-range and strided IO, chained 128-bit arithmetic, secret/public
matrices with empty and nonempty inner dimensions, empty/no-output finalization,
internal reuse and external reauthentication, serialization and domain/session
rejection. For each of the three corrupt-party positions, tests cover input
replica errors, consistent value/MAC errors, wire value/MAC errors, forged and
truncated Protocol 9 digests, cross-batch replay and incorrect component IDs.
They assert actual injection, protocol rejection (rather than transport
timeout), terminal failure and unavailable exports. Repairing a returned buffer
does not erase the corrupt pending snapshot.

Run from `src`:

```sh
libspu/mpc/spdzwise/benchmarks/tools/bin/bazel-7.7.1 \
  --batch --output_user_root=/tmp/malicious-spu-bazel test -c opt \
  //libspu/mpc/spdzwise:ring_test //libspu/mpc/spdzwise:protocol_test \
  //libspu/mpc/spdzwise:io_test //libspu/mpc/spdzwise:boolean_test \
  //libspu/mpc:protocol_session_test //libspu/core:config_test \
  --lockfile_mode=off --jobs=16 --test_env=YACL_NUM_THREADS=4 \
  --test_timeout=180 \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain
python3 libspu/mpc/spdzwise/ring_reference_test.py
python3 libspu/mpc/spdzwise/field_reference_test.py
```

All six targets executed successfully during this implementation. The final
invocation executed the corrected RING tests and reused passing caches for the
five unchanged regression targets. No linked full-protocol UBSan run was
performed; the UBSan evidence above is for the actual scalar arithmetic header.

The implementation reference and security-premise mapping are in
[README](README.md#protocol-9-and-security-model) and next to
`ring::CheckZero`. In particular, Protocol 9 relies on a random oracle, while
reused secret keys and secret key-times-dot-product checking require a separate
composition argument before applying the paper's main-protocol bound to the
whole session. Tests are not a substitute for that argument. The factory
remains closed; PPML acceptance and performance benchmarks were not run.

## Domain routing and source separation — 2026-10-20

The FM64-only `spdzwise_config.domain` configuration passed default/legacy,
protobuf, binary, JSON and diagnostic-string checks for FIELD and RING.
FIELD routes to F61 arithmetic kernels and shared ABFL; RING registration and
IO remain unimplemented and reject without installing field kernels. A routed
multiplication crossing the centered F61 boundary confirms prime-field behavior.
Arithmetic and Boolean session operations were moved without algorithm changes.

Seven optimized native targets passed, totaling **283 cases**:

| Target | Cases | Failures |
|---|---:|---:|
| `//libspu/core:config_test` | 4 | 0 |
| `//libspu/mpc/spdzwise:protocol_test` | 11 | 0 |
| `//libspu/mpc/spdzwise:io_test` | 8 | 0 |
| `//libspu/mpc/spdzwise:boolean_test` | 16 | 0 |
| `//libspu/mpc:protocol_session_test` | 7 | 0 |
| `//libspu/mpc/aby3:protocol_test` | 225 | 0 |
| `//libspu/mpc/aby3:io_test` | 12 | 0 |

Run from `src` with Bazel 7.7.1, `test -c opt`, the seven targets above and
`--lockfile_mode=off --jobs=4 --test_env=YACL_NUM_THREADS=4 --test_timeout=180
--extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain`.
Five targets passed in the initial build; after removing a helper made unused
by the source split, the final run reused those results and executed the two
remaining SPDZWISE targets. Counts were checked against Bazel XML reports.
The updated `//libspu/mpc:factory` translation unit also compiled successfully
with `build -c opt --output_groups=compilation_outputs`.

Python bindings, exports and type stubs include `SpdzWiseConfig`; Python source
syntax checks passed. The full Python extension integration suite was not run.
The production factory remains gated. Earlier benchmark fingerprints below
describe their recorded source snapshots, preceding this configuration/layout
change; performance benchmarks were not rerun.

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
commands/logs and source provenance. C++ and BUILD fingerprints matched at that
experiment refresh, which changed only the runner and documentation and did
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

Final RING C1–C3 measurements used `YACL_NUM_THREADS=12`, three-party
FactoryMem, one warmup and seven measured repetitions per circuit. All 24
runs checked every output, immediate input authentication, one final check,
checked export and exact phase payloads. The six regression targets passed
59 cases: RING 13, FIELD protocol 11, IO 8, Boolean 16, ProtocolSession 7 and
config 4. Coverage includes canonical hash transcripts at chunk boundaries,
66,049-element noncompact arithmetic, corruption, immutable snapshots and
failure propagation. The 14,096-vector native scalar/Python bigint oracle
also passed with UBSan (scalar-only coverage).

Final raw measurements, parameters, build log and validation evidence are
retained in [`benchmarks/spdzwise-ring128-latest/`](benchmarks/spdzwise-ring128-latest/).
Results are summarized in the project
[Arithmetic Benchmark](../../../../readme-malicious-ppml.md#arithmetic-benchmark).

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

## RING masked B2A

`ring_conversion_test` additionally covers 1/5/8/16/32/63/64-bit Boolean
inputs, zero extension, full-width carries, multidimensional strided inputs,
empty and small accumulated calls, the 320/321-element standalone threshold,
A2B/B2A round trips, Boolean and arithmetic consumers, and fresh masks across
repeated conversions. It reconstructs the extended-ring value and MAC in the
test harness and checks `mac = alpha * value`, including high limbs.

For each party position, fault tests corrupt mask MAC records (one copy and
both copies), AND results (one copy and both copies, including a padding lane), and the share
encoded in the masked opening payload. A test-only channel decorator also
corrupts an actual `b2a-open` wire payload to just one recipient and checks
that ordered-view agreement rejects the equivocation. An observer on the existing message-tag
callback asserts that arithmetic/ABFL failures never reach `b2a-open`; corrupt
opening copies must be rejected before any output. Consistent two-copy
injections test the algebraic checker separately from copy checking; they do
not model two corrupt parties under the one-corruption threat model.

The 100K benchmark compares the registered B2A kernels, verifies every output
for one warmup and seven measured runs, asserts one arithmetic mask check and
one ABFL check, and records all 188 ANDs per padded input lane. Raw logs,
parameters, hashes and validation artifacts are archived in
`benchmarks/b2a-split3-100k-2026-10-22/`.
