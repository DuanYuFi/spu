# SPDZ-wise F61 development status

This directory is **not a completed malicious PPML backend**. The runtime
factory deliberately rejects `SPDZWISE` execution until the acceptance gates
below pass. There is no fallback to ABY3, a trusted dealer, or plaintext
execution. `registerArithmeticPrototype()` is an explicit C++ development
entrypoint, not a production configuration.

## Baseline and scope

- SPU baseline: `826537d90d60c1a94286852c5ba13c8eb470a785`.
- MP-SPDZ reference: `73eb8556cdfc670c5d3dc1067a3ae5b4ad0d4441`.
- Three replicated-share servers, one static malicious corruption at most,
  honest external clients, authenticated confidential channels, security with
  abort. Network identities and client authorization belong to the deployment.
- Arithmetic: `p = 2^61 - 1`, FM64 storage, default 16 fractional bits. Boolean
  target: ABFL post-sacrifice. Forty-bit statistical checking parameters are
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
| Boolean online and checking | `GC/PostSacriSecret.h`, `PostSacriBin.cpp` | **Not implemented**; preserve reference pool/sacrifice parameters before optimization |
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
2. Implement ABFL and malicious daBits; prove/check the combined opening
   schedule before enabling A2B/B2A, precise truncation and F4 comparisons.
3. Add the specified LR/tree/CNN fixtures, pinned data splits and weights,
   actual secure executions, interval/range records and float/quantized
   numerical acceptance. Plaintext oracle tests do not satisfy this gate.
4. Run old-protocol IO/kernel/device regression tests; measure preprocessing,
   computation, checks, export, communication and memory; publish composed
   failure accounting. Only then enable the production factory.

## Benchmark

The sole retained benchmark is `circuit_bench`: two secret F61 inputs and
1,048,576 dependent arithmetic multiplication gates. The circuit definition,
current measurements, timing boundaries and raw data are in
[`benchmarks/2026-10-19-circuit/RESULTS.md`](benchmarks/2026-10-19-circuit/RESULTS.md).
For the project progress and complete test/build commands, see
[`readme-malicious-ppml.md`](../../../../readme-malicious-ppml.md).
