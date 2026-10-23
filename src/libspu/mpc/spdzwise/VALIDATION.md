# Development validation record

## RING framework modification audit — 2026-10-23

The root README's **Original SPU Modification** section now records a
per-entry audit against original SPU commit
`826537d90d60c1a94286852c5ba13c8eb470a785`. The following original files are
restored byte-for-byte to that baseline:

- `kernel/hal/ring.cc` (the user's rollback) and its `ring_test.cc`.
- `core/config.cc`, `core/BUILD.bazel`, and `core/context.h`.
- `mpc/kernel.h`, `mpc/common/pv2k.h`, and `mpc/common/pv2k.cc`.
- `spu/tests/spu_runtime_test.py`.

The now-unreachable protocol-local `RingLess` kernels are removed. Comparison
uses the unchanged HAL `MSB(x-y)` path: bounded differences are tested against
signed `<`, while full-width random/overflow cases verify modular MSB
semantics. Exact signed floor truncation is unchanged, including full-width
inputs; its selector and implementation use existing plain `Kernel`
registration instead of extending the global rounding enum. Finalization
uses existing `KernelEvalContext` and `Kernel::evaluate()` without a zero-argument
parameter-binding extension.

SPDZwise-specific global configuration validation is removed because the
protocol/IO constructors already enforce storage/domain constraints. FM64
now uses SPU's default **18 fractional bits**; the example configuration
continues to request **16** explicitly. Invalid configurations are rejected
at protocol/IO creation rather than during generic configuration population.
The C++ configuration fixture moved to `spdzwise/config_test.cc`; the Python
domain round-trip fixture moved to `spdzwise_runtime_test.py`.

Optional session boundaries, checked Python exports, internal MAC
serialization guards, factory/configuration bindings and rank-aware IO remain
in use. The common public-kernel registration API is unchanged; SPDZwise
uses a local forwarding adapter and separately installs `rand_p`. Device
execution tests live in `spdzwise/execution_test.cc`.

Fresh validation after the rollback:

- **15 native targets / 477 cases passed** (14 targets executed, one unchanged
  target reused Bazel's passing cache). This includes four relocated config
  cases, seven device execution cases, all SPDZwise kernel/IO suites, generic
  device IO, ProtocolSession, original HAL tests and ABY3 protocol/IO regression.
- **6 Python cases passed in 1.486 seconds** against the freshly rebuilt
  optimized extension: configuration round-trips, rank-aware IO, repeated
  runtime/export accessors, LR inference/gradient, MLP/log-softmax/BatchNorm,
  and Conv/BatchNorm/ReLU. The LR fixture tests both default precision (18 after
  population) and explicit 16; other numerical fixtures use the restored default.
- Extension SHA-256:
  `402be3987c2785bf697e0fd7fca6f0bf8dfbc70f772550f093c070719b2b96b4`.
- All nine restored original files listed above match the baseline byte-for-byte;
  Python syntax and `git diff --check` pass.

The native command below also applies to this audit, adding
`//libspu/mpc/spdzwise:config_test`. Python runs with
`python -m unittest -v spu.tests.spdzwise_runtime_test` using the runtime-only
package staging described below. Full logs for this run are
`/tmp/spdzwise-framework-audit-native.log`,
`/tmp/spdzwise-framework-audit-python-build.log`, and
`/tmp/spdzwise-framework-audit-python-tests.log`.

## Historical runtime registration validation before the audit — 2026-10-23

The results in this section describe the earlier integration and its binary
hashes. Its optional full-width comparison dispatch has since been removed;
the audit above defines the current comparison and precision contracts.

`Factory::RegisterProtocol` now selects `regSpdzWiseProtocol` for SPDZWISE.
Only RING/FM64 is enabled; FIELD retains its explicit arithmetic-development
entrypoint. Python reconstruction delegates ordered rank/type/binding checks
to SPDZwise IO instead of rejecting the intentional rank-specific wire types.
Local tracing/profiling flags are excluded from execution agreement; numerical
and protocol configuration remain bound. Public random draws use a separate
commit/reveal coin exchange and do not require a generic PRG state. Full-width
signed `less_ss/sp/ps` kernels correct subtraction overflow and return
deferred-check Boolean bits.

All 14 native targets passed, totaling 475 cases after adding public random
coverage and the final MAC fault fixture. The seven actual device-execution
cases cover:

- Checked output serialization and consecutive executions reusing internal MACs.
- Full-width signed comparison, Boolean output and rank permutation rejection.
- Empty-output final checking and successful transaction completion.
- Fixed-point exp/reciprocal, rsqrt, ReLU and secret matrix multiplication,
  compared with cleartext values at absolute tolerance 0.01.
- Parsing failure revoking previous exports and poisoning subsequent execution.
- Incomplete FIELD execution, snapshots and inter-op parallelism rejection.
- MAC corruption at each server position preventing new environment publication
  and revoking checked exports.

The existing 1025-element HAL comparison fixture now checks signed comparisons
for all random full-width pairs, including INT64_MIN/1 and INT64_MAX/-1.
Other regression targets include all eight SPDZwise kernel/IO targets,
ProtocolSession, generic device IO, HAL comparison dispatch, ABY3 protocol and
ABY3 IO. This validates execution behavior, not a composed security proof or
full ResNet-50/full-dataset model accuracy.

Build/test from `src`, with the bundled Bazel 7.7.1 and
`--output_user_root=/tmp/malicious-spu-bazel`:

```sh
libspu/mpc/spdzwise/benchmarks/tools/bin/bazel-7.7.1 --batch \
  --output_user_root=/tmp/malicious-spu-bazel test -c opt \
  //libspu/mpc/spdzwise:execution_test //libspu/device:io_test \
  //libspu/mpc/spdzwise:basic_ops_test //libspu/mpc/spdzwise:boolean_test \
  //libspu/mpc/spdzwise:ring_test //libspu/mpc/spdzwise:protocol_test \
  //libspu/mpc/spdzwise:io_test //libspu/mpc/spdzwise:ring_conversion_test \
  //libspu/mpc/spdzwise:ring_truncation_test //libspu/mpc/spdzwise:ring_mula1b_test \
  //libspu/mpc:protocol_session_test //libspu/mpc/aby3:protocol_test \
  //libspu/mpc/aby3:io_test //libspu/kernel/hal:ring_test \
  --lockfile_mode=off --jobs=12 --test_env=YACL_NUM_THREADS=4 --test_timeout=180 \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain \
  --per_file_copt=libspu/dialect/pphlo/IR/type_inference.cc@-Wno-error=maybe-uninitialized
```

The per-file option handles an existing GCC/LLVM ArrayRef warning in type
inference without changing that unrelated source. After the complete run,
the device and protocol targets were rerun for the final public-random binding
and passed.

### Python binding and JAX execution

The new optimized `//spu:libspu` extension built successfully. Its SHA-256 is
`21b458ef7eac9663d0be4b004d5a5bd570fba7522e6f55ee1b3a19b8a13e5a0c`.
All five `spu.tests.spdzwise_runtime_test` cases passed in 1.764 seconds using
JAX 0.8.0 and NumPy 2.1.3:

- Rank-specific Python IO reconstruction, with swapped ranks rejected.
- Consecutive executions, metadata and chunk accessors, checked reconstruction.
- Secret LR inference plus one gradient update (absolute/relative tolerance 0.01).
- Secret MLP/ReLU, log-softmax and BatchNorm normalization (tolerance 0.02).
- Secret Conv/BatchNorm/ReLU inference (tolerance 0.02).

All inputs, weights and variance parameters in the JAX fixtures are secret.
The tested extension uses the actual compiler, factory, PPHLO executor and
checked IO. A temporary runtime-only package linked the repository's Python
sources and this new extension; unrelated PSI bindings were not rebuilt.
Python dependencies came from the existing TransformerInference virtualenv.
Full dataset training and the original full ResNet-50 model were not run.

Build from the repository root:

```sh
src/libspu/mpc/spdzwise/benchmarks/tools/bin/bazel-7.7.1 --batch \
  --output_user_root=/tmp/malicious-spu-bazel build -c opt //spu:libspu \
  --lockfile_mode=off --jobs=12 \
  --extra_toolchains=@@rules_foreign_cc~//toolchains:preinstalled_cmake_toolchain \
  --per_file_copt=libspu/dialect/pphlo/IR/type_inference.cc@-Wno-error=maybe-uninitialized
python -m unittest -v spu.tests.spdzwise_runtime_test
```

The root lockfile was preserved byte-for-byte. The upstream registry had
changed `ippcp/2021.8.bcr.2/source.json`; its original content was recovered
from registry commit `109bae5dff96b2502b7b720c5a450073dd24fab6` and cached only
after verifying its SHA-256 matched the locked
`4a4a758ddf4292e9af6f7c6b1cd014758982a54c2dd3c50fb1912ceec68a5b69`.
No dependency hash or integrity check was weakened.

## RING basic operators and random shares — 2026-10-23

Added the nine planned kernels: `lshift_a`, `lshift_b`, `rshift_b`,
`arshift_b`, `bitrev_b`, `common_type_b`, `cast_type_b`, `rand_a`, `rand_b`.
RandA preserves full-ring sampling and uses the existing deferred MAC path;
the signed-comparison overflow restriction remains unchanged.

All nine optimized-build targets passed, totaling 99 test cases:
`basic_ops_test` (6), `boolean_test` (17), `ring_test` (15),
`protocol_test` (13), `io_test` (8), `ring_conversion_test` (15),
`ring_truncation_test` (11), `ring_mula1b_test` (7), and
`protocol_session_test` (7).

The new tests cover public scalar/per-element shifts, all planned boundary
widths, empty and strided inputs, invalid parameters and sessions, checked
external reconstruction, unchanged local-operation communication/check queues,
extended-ring MAC composition, deterministic full-width PRSS draws and exact
stream continuation, and inconsistent/consistent random-MAC faults at all
three party positions. Test-only deterministic streams never enter production.

Run from `src`:

```sh
libspu/mpc/spdzwise/benchmarks/tools/bin/bazel-7.7.1 \
  --batch --output_user_root=/tmp/malicious-spu-bazel test -c opt \
  //libspu/mpc/spdzwise:basic_ops_test \
  //libspu/mpc/spdzwise:boolean_test //libspu/mpc/spdzwise:ring_test \
  //libspu/mpc/spdzwise:protocol_test //libspu/mpc/spdzwise:io_test \
  //libspu/mpc/spdzwise:ring_conversion_test \
  //libspu/mpc/spdzwise:ring_truncation_test \
  //libspu/mpc/spdzwise:ring_mula1b_test //libspu/mpc:protocol_session_test \
  --lockfile_mode=off --jobs=16 --test_env=YACL_NUM_THREADS=4 \
  --test_timeout=180 \
  --extra_toolchains=@rules_foreign_cc//toolchains:preinstalled_cmake_toolchain
```

After the full run, the PRSS continuation assertions were strengthened and
`basic_ops_test` was rerun successfully. No model benchmarks were run.

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


## Exact TruncNaive default — 2026-10-23

`TruncASelector` now defaults to `TruncNaive`: A2B, local sign-extending right
shift, and checked B2A. It computes exact signed floor without MSB error;
`TruncPrRandomBit` remains explicitly callable with its original rounding.

Nine regression targets passed **323 cases**. The 11 truncation cases include
all shifts 0–63 over full int64 edges and seeded random values, non-contiguous
views, partial bit-sliced blocks, singleton/empty inputs, sign hints, invalid
parameters/session bindings, replicated MAC relations and subsequent arithmetic.
Both algorithms reject corrupted pending arithmetic/Boolean work before masked
opening. The preserved RandomBit tests still check probabilistic rounding and
owner/tile handling. Eight targets ran uncached initially; the final command
reused their passing results and rebuilt/retested the corrected truncation test.
The test helper now materializes a zero-shift strided result before inspecting
its replicated components, since a direct kernel may return the original view.

The retained benchmark smoke records cover Naive and ABY3 at 1K inputs. Its Naive check
requires exact floor equality, not a ±1 error bound. The main timing includes
all required pre-opening MAC/ABFL checks; a later empty checkpoint is reported
separately. This corrects the old benchmark's assumption that RandomBit leaves
its MAC queue for post-call verification.

[100K report and source fingerprints](benchmarks/truncation-naive-100k-2026-10-23/RESULTS.md),
[raw test XML/logs and commands](benchmarks/truncation-naive-100k-2026-10-23/validation/).

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

RandomBit performance measurements have been removed. Regression XML/logs and
shared source fingerprints remain in the [archive](benchmarks/truncation-signed-100k-optimized-2026-10-22/).
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

## RING MSB A2B validation — 2026-10-22

The registered `msb_a2b` kernel reuses split-3 CSA and computes only the
carry into bit 63 with a balanced tree. It returns one-bit Boolean shares;
HAL integer less-than reaches it through the existing subtraction/MSB fallback.
There are 241 ANDs per padded input lane in eight online AND layers. The
kernel leaves the arithmetic/Boolean verification schedule deferred.

Six native targets pass: `ring_conversion_test` (15 cases), `ring_test` (15),
`boolean_test` (17), `protocol_test` (13), `io_test` (8), and
`ring_truncation_test` (9), for **77 cases, zero failures**. The final run
re-executed conversion tests after adding extended-ring and modular-overflow
coverage; the five unchanged targets reused passing Bazel results.

MSB coverage includes full-width boundary/random values, long carries,
alternating bits, 0/1/63/64/65/130/1025-element inputs, scalar/matrix shapes,
non-contiguous views, output dtype/session/bit width, Boolean consumers and
agreement with full A2B. Authenticated multiplication outputs exercise high
128-bit value limbs and deferred arithmetic checks. HAL secret-secret,
secret-public and public-secret comparisons cover equality, signed boundaries,
random non-overflowing pairs and explicit modulo-2^64 subtraction overflow.
Wrong domain/session/input type/lifecycle are rejected. Corrupt carry records
and padded SIMD lanes are rejected at finalization for each party position.

Regression logs and XML results are retained in the final comparison archive's
`validation/regression/` directory. These runs use the existing C++ prototype
registration entrypoint; production factory gating is unchanged.

## Checked comparison benchmark — 2026-10-22

The same HAL `i_less` call is used for both protocols. Inputs are two
arrays of 100,000 `DT_I64`/FM64 arithmetic secrets, with identical deterministic
plaintext values and shape. Both outputs are serializable one-bit Boolean
secret-share arrays with dtype `DT_I1` and the same shape. Runtime assertions
check these contracts; the complete outputs are reconstructed against the
same signed less-than and modular-MSB oracle outside timing.

Timing begins after native input preparation/authentication and ends after
output preparation. For SPDZ-wise, the normal output transaction calls
`prepareOutputs -> finalize_protocol -> verify -> checkMac + checkBoolean`,
then exports and commits checked shares, all before the stop timestamp.
No security check is left after the stop timestamp. The input authentication
and context/PRSS setup remain excluded under the already-authenticated-input
assumption.

`checkMac` executes its count/status exchange inside timing. Its nonempty
batch count is zero because authenticated inputs and linear subtraction do
not create new arithmetic MAC work. `checkBoolean` verifies the whole batch
of 24,107,712 padded AND bits per party, including random material generation
and all verification traffic. Each party performs one call to each check via
the normal finalization path and records one nonempty Boolean batch.

The rebuilt benchmark and a 65-pair smoke run pass. The formal FactoryMem run
uses 64 worker threads, one warmup and seven measured iterations. All 800,000
outputs per protocol pass the oracle, and every SPDZ-wise run completes its
security checks successfully. Total time uses the maximum complete interval
across parties, then the median across iterations.

| Protocol | Checked total median ms | Range ms | Three-party sent bytes |
|---|---:|---:|---:|
| SPDZ-wise ring | 273.963 | 261.656–305.249 | 361,627,200 |
| Native ABY3 | 193.877 | 189.551–199.582 | 10,800,000 |

SPDZ-wise/ABY3 total time is **1.413x**. SPDZ-wise finalization and output
preparation have a diagnostic median of 250.763 ms and send 352,586,808 bytes
across three parties. Phase medians must not be added to reconstruct total
median time. Internal circuit/layout optimizations differ, so this measures
the implementations at the same HAL interface rather than isolating the
cost of the security model.

[Checked results and raw artifacts](benchmarks/comparison-msb-checked-100k-2026-10-22/RESULTS.md)
include source/binary fingerprints, a source archive, per-party CSV,
environment, commands, build log, smoke-test evidence and protocol regression
results. Only the final checked comparison experiment is retained.
