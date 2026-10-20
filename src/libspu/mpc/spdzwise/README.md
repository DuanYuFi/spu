# SPDZ-wise development status

This directory is **not a completed malicious PPML backend**. The runtime
factory deliberately rejects `SPDZWISE` execution until the acceptance gates
below pass. There is no fallback to ABY3, a trusted dealer, or plaintext
execution. `registerArithmeticPrototype()` is an explicit C++ development
entrypoint, not a production configuration.

## Arithmetic domain configuration and source layout

The protocol name remains `SPDZWISE` in both domains. Configure the arithmetic
domain independently from the SPU word size:

```json
{
  "protocol": "SPDZWISE",
  "field": "FM64",
  "spdzwise_config": {"domain": "FIELD"}
}
```

`FIELD` (the default, including legacy configurations) selects the Mersenne
prime `p = 2^61 - 1` with FM64 storage. `RING` reserves arithmetic modulo `2^64`;
its registration and IO entrypoints currently throw `not implemented` before
installing kernels or constructing field IO. Both domains reject FM32 and
FM128. Neither setting enables the production runtime factory.

In Python, use `config.spdzwise_config.domain = spu.SpdzWiseConfig.FIELD`
or `spu.SpdzWiseConfig.RING`. JSON/protobuf serialization and config copies
preserve the domain, which is also included in execution agreement.

- `protocol.h/.cc`: configuration-based domain dispatch, lifecycle and common
  registration; explicit field and reserved ring prototype entrypoints.
- `arithmetic.h/.cc`: F61 arithmetic operations and `spdzwise::field` kernels,
  including `field::MulAA`, `field::MulAP` and field matrix multiplication.
- `boolean.h/.cc`: shared ABFL implementation and Boolean kernel adapters.
- `conversion.h/.cc`: reserved checked field A2B/B2A registration; currently
  registers no conversions. Ring conversions and domain-specific PPML kernels
  (truncation/comparison) remain future work.
- `state.h/.cc`: checked session lifecycle, field authentication and checks.
  The current state rejects ring configurations.

The source files share one implementation library because the kernels and
checked session depend on each other. The `:boolean` build label remains
available as an alias. Existing F61/ABFL development calls continue to use
`registerArithmeticPrototype()`, which now selects the configured domain.

## Baseline and scope

- SPU baseline: `826537d90d60c1a94286852c5ba13c8eb470a785`.
- MP-SPDZ reference: `73eb8556cdfc670c5d3dc1067a3ae5b4ad0d4441`.
- Three replicated-share servers, one static malicious corruption at most,
  honest external clients, authenticated confidential channels, security with
  abort. Network identities and client authorization belong to the deployment.
- Arithmetic: `p = 2^61 - 1`, FM64 storage, default 16 fractional bits. Boolean
  prototype: ABFL post-sacrifice. Forty-bit statistical checking parameters are
  per reference procedure; a complete task needs a separately derived composed
  error bound. No security bound is certified for this prototype.
- Target acceptance: logistic regression training/inference, depth-three tree
  inference, small Conv/ReLU/MaxPool/Dense inference with private weights and
  inputs. These model acceptances have **not** been run.

The implementation follows algorithms in MP-SPDZ; the upstream notices are
retained in `MP-SPDZ-LICENSE.txt`. No MP-SPDZ binary or Python compiler is used
as a runtime dependency.

## Implemented development components

- Dependency-free scalar F61 arithmetic, signed/ring conversion, bounded input
  checks, and arbitrary-precision Python differential tests using the actual
  C++ header and UBSan.
- External input/output types, rank/binding checks, replicated duplicate checks,
  noncanonical-element rejection, noncontiguous-array handling. Internal MAC
  types cannot be deserialized or serialized as `ValueProto`.
- An arithmetic prototype with session-local replicated key, fresh random
  sharing, input authentication, linear operations, multiplication, batched
  matrix products and the SpdzWise secret-coefficient check structure.
- Optional checked execution lifecycle: import, finalization, staged exports,
  commit and terminal failure. Existing protocols install no session and keep
  their old execution paths.
- C++ tests for IO, arithmetic transactions, matrix products, cross-run reuse,
  corrupted MACs and lifecycle failure. Native arithmetic/IO/session tests pass; full device/Python integration
  remains outstanding. See the validation record below.
- Plaintext specifications for full field decomposition, two's-complement
  conversion, floor truncation and fixed-point range accounting. These are
  numerical oracles, **not** cryptographic conversion implementations.

## Source mapping and protocol dependencies

| Component | MP-SPDZ reference | Local implementation / outstanding work |
|---|---|---|
| Arithmetic types and key | `Protocols/SpdzWiseShare.h`, `SpdzWiseShare.hpp` | Two value and two tag components; fresh key per runtime; no persisted keys |
| Input authentication | `Protocols/SpdzWiseInput.hpp::exchange` | Replicated-copy validation followed by shared-key multiplication and arithmetic check; external-client integrity still needs source/proof review |
| Multiplication and dot product | `Protocols/SpdzWise.hpp::prepare_mul`, `prepare_dotprod` | Multiply value/value and tag/value, enqueue result; matrix terms accumulated locally before one batched exchange |
| Arithmetic check | `Protocols/SpdzWise.hpp::check`, `zero_check` | Secret random coefficients, two shared dot products, shared-key product, independently blinded zero opening |
| Boolean online and checking | ABY3 online AND; `Protocols/ShuffleSacrifice.h`, `MalRepRingPrep.hpp` | ABFL prototype with 64-bit word buckets, final post-sacrifice and checked Boolean IO; uses one fresh random pool rather than `PostSacriBin`'s two-level pools |
| Correlated bits | `Protocols/ReplicatedPrep.hpp`, `DabitSacrifice`, `ShuffleSacrifice` | **Not implemented**; do not substitute unchecked daBits |
| A2B | `Compiler/non_linear.py::KnownPrime`, `floatingpoint.py::BitDecFull` | Plaintext oracle only; cryptographic masking and binary circuits outstanding |
| B2A | `Compiler/types.py` secret conversion and `bit_compose` | Plaintext oracle only; requires validated daBits and checked Boolean opens |
| Truncation | KnownPrime decomposition route, adapted to SPU rounding | Plaintext floor oracle only; A2B/shift/B2A kernels outstanding |

Arithmetic multiplication now uses a shared F61 replicated semi-honest
primitive (`replicated.h/.cc`) for both SPDZwise and the ABY3-F61 benchmark.
Per-party, independently seeded AES-CTR PRSS streams generate the MAC key,
multiplication masks, and secret checking coefficients. Field sampling rejects
p after masking to 61 bits, consuming each correlated stream independently.
Masks and checking coefficients are expanded locally and are not transmitted.
The checked values still enter the authentication queue before release.
Equivalence under malicious scheduling and input handling remains a review gate;
these performance changes do not enable the production factory.

External input types currently carry the destination rank, but no client-issued
request nonce. Output types carry a common digest; reconstruction checks that
the three digests agree, but its generic IO API does not yet accept an expected
request digest. Cross-batch disagreement is rejected; replay of an entire old
three-share bundle is **not** distinguished by this IO prototype. Client/server
request binding and the authenticated-channel deployment example remain M1
work and are required before enabling external use.

### Message inventory

| Message | Contents and visibility | Required protection |
|---|---|---|
| Execution agreement | Digest of executable/config/input metadata/public inputs | Every party compares to its own descriptor; reject disagreement |
| Session setup | Public random nonces and their agreed digest | Bind subsequent exports to this runtime and execution counter |
| PRSS setup | Independent seed exchanges for key, multiplication masks and checking coefficients | Confidential authenticated point-to-point channels; setup precedes computation |
| Input-copy check | A component to the party already holding that component | Exact equality; never send it to the missing-share party |
| Optimistic multiplication | Masked local product/dot-product contribution to duplicate holder | Result must enter arithmetic checks before release |
| Arithmetic check opening | Both components of the blinded check value | All parties reconstruct from both copies, reject disagreement or nonzero result |
| Client output | Checked value components only, with execution/name/shape/dtype binding | Three authenticated senders; all duplicate components and metadata must agree |

`openRaw()` is private to the arithmetic checker. It must never become a
general reveal kernel. Application-level reveal kernels explicitly reject.
No destructor performs network communication.

## ABFL Boolean development prototype

The existing `registerArithmeticPrototype()` entrypoint also registers `p2b`,
`and_bb`, `and_bp`, `xor_bb`, and `xor_bp`. This remains a C++ development
entrypoint; the production factory is still closed. Boolean values have two
64-bit XOR-share components and 1–64 effective bits. AND uses the smaller operand
width; XOR uses the larger width; public operands and `p2b` have width 64.
Unused share bits are zero. Generic shape operations preserve the complete pair.

Use `SpdzWiseIo::toBooleanShares(raw, nbits)` for packed FM64 input, or the
standard `makeBitSecret(PtBufferView)` interface for individual bits (including
packed bitsets). `toShares()` retains its arithmetic semantics. Import validates
replicated copies. Internal `BShrTy` values bind to the runtime session and
cannot be serialized; committed `BOutputTy` values carry rank, width and the
execution/output binding. `fromShares()` checks these and reconstructs by XOR.

Online secret AND performs the usual ABY3 local products, PRSS masking and one
ring exchange (8 payload bytes per array element per party), then snapshots
both operands and the result. It performs no sacrifice. The pending bit count
is the sum of `numel * min(lhs.nbits, rhs.nbits)` across calls; snapshots are
packed across call boundaries without counting padding bits. Full-word snapshots
and one overflow triple replace per-bit insertion. Full-width batches own
separate contiguous snapshot blocks without disturbing the overflow or
relocating old records; narrower batches use word shifts and ORs in a buffer
whose capacity grows geometrically. Completed words are not zeroed before
being overwritten. All six share components follow the same mapping.

The full-width online fast path borrows compact operands directly, normalizes
noncontiguous views once, and fuses output construction with snapshot insertion.
It avoids input/result staging vectors and, on little-endian hosts, sends the
product buffer directly and loads received words with alignment-safe memcpy.
The canonical little-endian wire format is unchanged; big-endian hosts retain
explicit encoding/decoding. Empty inputs perform no exchange. Type/session
validation is retained, but width-64 values skip the vacuous high-bit scan.
Narrow-width values retain canonical checks and the general masked packing path.
Verification visits the narrow-word buffer, full-width call blocks, then the
tail in a deterministic public order, without concatenating the blocks. The
batch agreement uses the `word64-v8-parallel-views` format tag and binds the
65,536-word tile size. Mixed versions fail agreement before random generation.
The sacrifice formula, bucket parameters and random-pool-only permutation are
unchanged. The implementation uses existing parallel and transport APIs; it
does not modify framework PRGs, communication implementations or thread pools.

Finalization counts M=ceil(N/64) packed word triples, where N is the active
bit count. It selects B=3 for M>=1048576, B=4 for M>=10368, and B=5 for
M>=1024. A nonempty smaller word batch fails and poisons the session; N=0 is a
no-op. Thus 1,024 **bits** no longer suffice: the smallest accepted bit count
is 65,473 (1,023 full words plus a one-bit tail). No extra full words are added
to reach a bucket threshold. One partial word, if present, is zero-extended
with valid (0,0,0) lanes and checked in its entirety, including padding.
The 10368 threshold follows the local MP-SPDZ `ShuffleSacrifice.h` reference.
The checker generates `(B-1)*M+3` full-width random word triples using independent Boolean
PRSS streams and the semi-honest AND primitive. It checks the online and random
replicated copies before a fresh, session/execution/batch-bound commit–reveal
coin. An unbiased Fisher–Yates permutation applies only to the random pool, with
whole words as permutation entries. Three full random words (192 bit lanes)
are opened and checked; each online word is checked against B-1 random word
companions. The batch agreement binds the word64 format version, N, M, B,
overflow width and call layout. Online inputs are never sampled for
public opening.

Verification buffers use the following optimizations:

- Random generation and sacrifice reuse scratch space for 65,536-word tiles.
  Random triples are all generated before the pool is fixed and the public
  permutation coin is revealed. Generation tiles are not separate ABFL batches.
- Fixed PRSS tasks own the First/Second streams and mutate distinct counters;
  each stream generates a then b within a tile. Full tiles contain an even
  number of words, preserving AES-block consumption across tile boundaries.
  Serial-reference tests compare all generated shares and subsequent streams.
- Independent word loops use the existing `yacl::parallel_for` and uint64_t
  reductions (avoiding the shared-bit storage of `vector<bool>`). Outputs have
  disjoint ranges; PRSS counters and communications are never updated by these
  element workers. Scratch storage is fully written before reading.
- Pre-coin replica checks hash both local components in bounded tiles using
  BLAKE3-256. Each digest binds the session/execution/batch tag, component owner
  and tile bounds. First-component digests are sent only to the party already
  holding those exact components as its second shares. All comparisons and
  the three-party status agreement complete before commit–reveal. This replaces
  raw-copy transport with a collision-resistant equality check; it relies on
  BLAKE3 collision resistance, as do the existing transcript checks.
- Openings still exchange both shares, compare every replica and check public
  view agreement before use. Each rank's payload is hashed independently in
  parallel; an outer hash binds the opening tag, ordered lengths and payload
  digests. Local inputs are borrowed, peer messages remain owned, and lengths
  are checked before alignment-safe word loads. Wire words remain little-endian.
- The existing synchronous `Send` API avoids an asynchronous staging copy for
  borrowed buffers. Its completion means peer transport-buffer delivery, not
  application `Recv`, so send-before-receive does not create a cyclic wait.
- Companion collection and rho/sigma construction share a pass, with prefetch
  based only on public permutation indices. The permutation PRG caches 4096
  words. Swap indices are prepared in small batches, but rejection sampling
  and swap order match the serial Fisher–Yates reference exactly.

C2 measurements use FactoryMem with three parties in one process. The latest
ABY3 / ABFL comparison uses the same binary and `YACL_NUM_THREADS=12`; no global
thread default was changed. Both protocols run one warmup and seven measured
iterations and check every output. Tiling adds communication rounds, so these
local timings do not predict WAN performance. Only the latest Boolean
performance experiment is retained; see the
[results and commands](benchmarks/boolean-c2-latest/RESULTS.md).

For each pair, open rho=x XOR a and sigma=y XOR b, then check that
`z XOR c XOR (sigma AND a) XOR (rho AND b) XOR (rho AND sigma)` opens to zero.
There is no per-bit permutation or companion repacking. Every
opening validates replicated copies and compares ordered transcript digests
before its values are used. Check failures abort the session and discard all
pending outputs and checking state. Finalization runs even with no outputs;
arithmetic input authentication still checks only the arithmetic queue.

These are the requested prototype parameters, not a certified 40-bit bound for
this modified, fixed-online-bucket construction. The parameter table and C=3
come from reference procedures with different pooling/selection schedules;
security analysis and whole-execution composition remain acceptance gates.
A2B/B2A, daBits, comparisons, truncation and full Python/device integration are
not supplied by these Boolean kernels.

## Numerical contract

Public and Boolean words retain SPU's 64-bit two's-complement representation.
Secret arithmetic represents the centered interval `[-(p-1)/2, (p-1)/2]`.
Ring encoding `2^64-1` means `-1` and maps to `p-1`, not to `7`.

The external IO rejects out-of-range signed inputs. Intermediate arithmetic
does not secretly check ranges; models must supply a range argument and a
plaintext trace. This is not full 64-bit wraparound compatibility. Complex
values, private/colocated input, snapshots and context parallelism are disabled
in the prototype.

For the future mixed backend:

1. Full decomposition uses a uniform mask in `[0,p)`. Assemble 61 validated
   daBits and reject the all-one candidate before using the mask. Revealing
   this acceptance bit must not depend on the input.
2. Recover residue bits with the modular-wrap correction; then convert to the
   centered integer's 64-bit two's-complement representation.
3. B2A uses negative weight for bit 63 of a full word; shorter bit vectors are
   unsigned. Secret validity/range assumptions must be recorded at call sites.
4. Truncation means `floor(x / 2^m)`, including negative numbers; it is not
   modular division by `2^m` and is not nearest/probabilistic rounding.
5. Compare operands' signed binary representations, not just `msb(x-y)` when
   a difference may cross the field's centered boundary.

## Framework changes

- **F1**: optional `ProtocolSession` at the device execution entrypoint calls
  `import_s`. Internal authenticated parameters may be reused in the same
  session; serialized client input cannot assert this status.
- **F2**: `finalize_protocol` is mandatory for a checked session, even for empty
  output lists. Missing finalization fails closed. A run guard poisons the
  session on any execution exception.
- **F3**: outputs are prepared before internal environment publication and only
  exposed after commit. Python value/meta/chunk accessors read the same export
  view. `TypeObject::isSerializable()` defaults to true; the internal MAC type
  overrides it. `Value::toProto()` checks it before copying bytes. Client
  reconstruction checks dtype/shape as well as protocol wire metadata.
- **F4**: HAL optionally dispatches `less_ss`, `less_sp`, and `less_ps`, preserving
  operand order and the old fallback for protocols without these kernels.
  Dispatch regression tests are supplied; SPDZWISE comparison kernels still
  require the unimplemented checked Boolean backend.

`SPDZWISE = 7` is registered in C++/protobuf/Python configuration with 16
fractional bits by default. IO construction is available for development, but
`Factory::RegisterProtocol` explicitly refuses execution. Do not remove that
gate merely because normal-execution arithmetic tests pass.

## Validation and remaining gates

Run the dependency-free specifications:

```sh
python3 src/libspu/mpc/spdzwise/field_reference_test.py
python3 src/libspu/mpc/spdzwise/reference_test.py
```

Run the native development tests from the `src` Bazel module:

```sh
USE_BAZEL_VERSION=7.7.1 bazel test \
  //libspu/mpc/spdzwise:boolean_test \
  //libspu/mpc/spdzwise:io_test \
  //libspu/mpc/spdzwise:protocol_test \
  //libspu/mpc:protocol_session_test
```

Validation record for this change is maintained in `VALIDATION.md`.

Remaining acceptance gates, in order:

1. The three native prototype test targets pass (23 cases, including MAC
   corruption at all three server positions); see `VALIDATION.md`. Next test
   real device execution and additional fault paths. Review input/key/check composition and
   derive the bound rather than inferring it from finite fault tests.
2. Review the ABFL prototype and implement malicious daBits; prove/check the combined opening
   schedule before enabling A2B/B2A, precise truncation and F4 comparisons.
3. Add the specified LR/tree/CNN fixtures, pinned data splits and weights,
   actual secure executions, interval/range records and float/quantized
   numerical acceptance. Plaintext oracle tests do not satisfy this gate.
4. Run old-protocol IO/kernel/device regression tests; measure preprocessing,
   computation, checks, export, communication and memory; publish composed
   failure accounting. Only then enable the production factory.

## Benchmark

Benchmark sources, build targets, parameters and measurement archives are in
[`benchmarks`](benchmarks/).
The retained workloads cover the two-input F61 circuit, native ABY3 ring64
arithmetic and a recorded native ABY3 Boolean AND measurement.
For the project progress and complete test/build commands, see
[`readme-malicious-ppml.md`](../../../../readme-malicious-ppml.md).
