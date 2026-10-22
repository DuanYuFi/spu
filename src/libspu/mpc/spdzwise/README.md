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
prime `p = 2^61 - 1` with FM64 storage. `RING` implements arithmetic modulo
`2^64` using internal value/MAC components modulo `2^128` (64 extension bits).
Both domains reject FM32 and FM128. Neither setting enables the production runtime factory.

In Python, use `config.spdzwise_config.domain = spu.SpdzWiseConfig.FIELD`
or `spu.SpdzWiseConfig.RING`. JSON/protobuf serialization and config copies
preserve the domain, which is also included in execution agreement.

- `protocol.h/.cc`: configuration-based domain dispatch, lifecycle and common
  registration; explicit field and ring prototype entrypoints.
- `arithmetic.h/.cc`: F61 arithmetic operations and `spdzwise::field` kernels,
  including `field::MulAA`, `field::MulAP` and field matrix multiplication.
- `ring.h/.cc`, `ring_math.h`, `ring_type.h`, `ring_arithmetic.h`: independent
  128-bit arithmetic state, Protocol 9 checking, domain types and ring kernels.
- `boolean.h/.cc`: shared ABFL implementation and Boolean kernel adapters.
- `correlated_randomness.h/.cc`: RING authenticated random-bit generation,
  private bit inputs, public challenges, bitness checks and scratch storage.
- `conversion.h/.cc`: reserved checked field A2B/B2A registration; currently
  registers no conversions. FIELD truncation and comparisons remain future work.
  `ring_conversion.h/.cc` implements value-only split-3 RING A2B.
  `ring_truncation.h/.cc` implements signed RING `TruncPrRandomBit`, routed
  through `TruncASelector`.
- `state.h/.cc`: checked session lifecycle, field authentication and checks;
  owns the separate ring arithmetic state and shared Boolean state.

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
- FIELD arithmetic: `p = 2^61 - 1`, FM64 storage, default 16 fractional bits. Boolean
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
| FIELD arithmetic check | `Protocols/SpdzWise.hpp::check`, `zero_check` | Secret random coefficients, two shared dot products, shared-key product, independently blinded zero opening |
| Boolean online and checking | ABY3 online AND; `Protocols/ShuffleSacrifice.h`, `MalRepRingPrep.hpp` | ABFL prototype with 64-bit word buckets, final post-sacrifice and checked Boolean IO; uses one fresh random pool rather than `PostSacriBin`'s two-level pools |
| Correlated bits | `Protocols/ReplicatedPrep.hpp`, `DabitSacrifice`, `ShuffleSacrifice` | **Not implemented**; do not substitute unchecked daBits |
| FIELD A2B | `Compiler/non_linear.py::KnownPrime`, `floatingpoint.py::BitDecFull` | Plaintext oracle only; cryptographic masking and binary circuits outstanding |
| RING A2B | `SpdzWiseRingShare::split`, `Rep3Share2::split(case 3)` | Value-only local split-3, bit-sliced CSA + RCA, existing ABFL final verification |
| B2A | `Compiler/types.py` secret conversion and `bit_compose` | Plaintext oracle only; requires validated daBits and checked Boolean opens |
| FIELD truncation | KnownPrime decomposition route, adapted to SPU rounding | Plaintext floor oracle only; A2B/shift/B2A kernels outstanding |
| RING probabilistic truncation | `floatingpoint.py::TruncPrRing`, signed, k=63 | Authenticated random masks, low-64-bit opening and signed offset correction |

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

### FIELD message inventory

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

## RING arithmetic prototype

Use `SPDZWISE + FM64 + spdzwise_config.domain=RING` with the explicit C++
prototype registration. The implementation provides input authentication,
`p2a`, negation, secret/public addition and multiplication, and secret/public
matrix multiplication. Matrices accumulate each dot product locally before
exchanging two 128-bit words per output cell. A fixed status exchange also
rejects malformed wire lengths; payload does not grow with the inner dimension.
RING `a2b` converts the application value to 64-bit Boolean RSS. B2A, exact
truncation, comparison and PPML integration remain unimplemented. Signed
probabilistic truncation is described below.

Client input words are zero-extended from 64 to 128 bits, then shared in
Z/(2^128). Internal authenticated elements contain two value and two MAC words
(64 bytes), carry the session identity, and cannot be serialized. Wire words
use exactly 16 little-endian bytes. Shapes and strided views operate on complete
authenticated elements. Public words retain FM64 storage. FIELD and RING types
are distinct and cannot be interchanged.

Imports check replicated copies, generate MACs and immediately check the
authentication batch. Every secret multiplication stores an immutable copy of
its result in the pending queue. After finalization succeeds, exports contain
only the low 64 bits of each value share plus rank/execution binding. Clients
reconstruct modulo 2^64. External outputs are lifted component by component and
reauthenticated on reimport; authenticated internal values can be reused only
within the same healthy session. Empty batches skip zero checking; executions
with no outputs still finalize. Any checking failure clears pending exports
and permanently poisons the session.

The checker agrees batch size, verifies copies, draws fresh secret coefficients
from a dedicated PRSS stream, and computes one aggregation
`w=sum(r_i*x_i), u=sum(r_i*mac_i), t=u-alpha*w`. Neither `t` nor a multiple
of it is opened. Keys, masks and coefficients use separate PRSS instances.

Final RING measurements are included in the project
[Arithmetic Benchmark](../../../../readme-malicious-ppml.md#arithmetic-benchmark);
raw samples, parameters and validation evidence are in
[`benchmarks/spdzwise-ring128-latest/`](benchmarks/spdzwise-ring128-latest/).

### Signed RING probabilistic truncation

The prototype registers `TruncASelector` as `trunc_a`. Its `proc()` currently
routes to the sole implementation, `TruncPrRandomBit`; future public-parameter
selection belongs in this selector. The implementation's state entry points
are named `truncPrRandomBit`. Both kernels preserve the input dtype, and the
selector retains the probabilistic-LSB/no-MSB-error contract. The random-bit
protocol, security checks and timing boundaries are unchanged.

`trunc_a(x, m, SignType::Unknown)` accepts authenticated signed 63-bit inputs
`-2^62 <= x < 2^62`. This is a caller contract, not a secret range check. Both
sign hints use the same signed path. `m=0` is an identity; `1 <= m <= 62` is
supported, and other shifts fail. FIELD does not register this kernel. The
input/output internal RING type, dtype, shape and session binding are retained.

Reference: MP-SPDZ commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52`,
`Compiler/floatingpoint.py::TruncPrRing`, `signed=True`, `k<n_ring` (the compiler
labels this CdH10-fixed Protocol 3.1). `sy-rep-ring-party` uses this compiler
expansion, not the semi-honest Replicated `trunc_pr` instruction. The full-width
`k=64` / `TruncLeakyInRing` route is not implemented.

For `ell=64,k=63`, compute `x'=x+2^62`. Generate authenticated independent
`r_L` (m bits), `r_H` (63-m bits), and `r_B` (one bit), with
`r=r_L+2^m*r_H+2^63*r_B`. Open only `c=(x'+r) mod 2^64`; never open the
128-bit extension. Set `q=(c mod 2^63)>>m`, `c_B=c>>63`, and return

```
[b] = [r_B] + c_B - 2*c_B*[r_B]
[y] = q - [r_H] + 2^(63-m)*[b] - 2^(62-m)
```

Thus `y=floor((x+r_L)/2^m)` for either sign, without computing or revealing the
input MSB. With `t=x-2^m*floor(x/2^m)`, round-up probability is `t/2^m`.
For example `x=-7,m=2` gives -2 with probability 3/4, or -1 with probability
1/4. The final operations are linear on all four authenticated components;
there is no output reauthentication or separate output MAC check.

Material generation follows the default arithmetic-bit route of
`ReplicatedPrep.hpp::buffer_ring_bits_without_check` and
`Spdz2kPrep.hpp::MaliciousRingPrep::buffer_bits`: two rotating input owners
provide private bits, authenticate them, and combine with `u+v-2uv`.
Private inputs use `ReplicatedInput.hpp::add_mine_prepared`: in SPU's component
orientation, owner o shares a bit as `(s_o,s_{o+1},s_{o+2})=(r,b-r,0)`.
The fresh pairwise r is already known to o-1; only b-r is sent to o+1. Neither
non-owner knows the bit. The two owners' inputs are queued together, checked
for replica consistency, and authenticated in one batched exchange. Every
rank advances both truncation PRSS streams through both input batches,
including unused components, to preserve alignment across owner rotation.
Randomness is independent of keys, multiplication masks and secret check
coefficients. Batches of 4096 application elements bound transient memory;
all immutable input/product records remain in the session queue.

Temporary input/PRSS/coefficient buffers are reused within one truncation
call. Input authentication multiplies directly by the scalar shared key,
without allocating a repeated-key array. Copy checks stream directly from
authenticated storage in the same canonical value/MAC order. Combining the
64 bits uses a 256-input parallel grain; raw local products are parallelized.
These changes preserve the bitness checks, freeze-before-challenge ordering,
and `3*64*N + ceil(N/4096)` deferred records per nonzero truncation call.

After freezing and checking bit/MAC replicas, a fresh commit/reveal coin
produces public 64-bit coefficients for `sum_i r_i*(1-b_i)*b_i`. The value/MAC
dot product is queued for the normal deferred authentication check. An explicit
adaptation to this prototype is to apply the existing Protocol 9 `CheckZero`
to the full 128-bit bitness sum, instead of opening only its low word. Honest
generated bits are 0/1 in the full internal ring. No bitness residual, MAC
residual or extension is revealed. This implementation does not establish a
new composed malicious-security bound or enable the production factory.

Mask openings check low-word replicas, send only the missing 64-bit component,
and compare reconstructed transcript hashes with all-party failure propagation.
Any rejection poisons the session. Finalization still checks all queued records
before external export, including when the truncation caller does not export
the result. Generic reveal kernels remain disabled.

The signed 100K benchmark compares against the normally registered ABY3
`TruncAPr2` on the common strict range `|x|<2^62`, with `m=16` and unknown sign.
It includes fresh material generation and required internal checks, stops when
output shares exist, and reports final authentication checking separately.
ABY3 retains its native `+1` correction; its exact rounding distribution is not
assumed equal to MP-SPDZ's. See [benchmark instructions](benchmarks/README.md).

### Protocol 9 and security model

Reference: Mark Abspoel, Anders Dalskov, Daniel Escudero, Ariel Nof,
*An Efficient Passive-to-Active Compiler for Honest-Majority MPC over Rings*,
[Section 6.3, Protocol 9](https://ir.cwi.nl/pub/31033/31033.pdf#page=23).
The reference is also adjacent to `ring::CheckZero`.

Party i holds (t_i,t_{i+1}), sends H(-(t_i+t_{i+1})) to i+1, and checks the
digest from i-1 against H(t_{i+1}). BLAKE3-256 instantiates the random oracle.
The hash frames version, domain, session, execution, batch and component owner.
The sender targets owner i+2 and the receiver expects owner i+1. Replicated-copy
checks use a separate domain. Two status rounds propagate a detected failure
to every honest party, including a corrupt sender's equivocation about status.

| Proof premise / interface | Prototype mapping |
|---|---|
| Three-party replicated shares, at most one static malicious party | Pair (component i, component i+1); channels must be authenticated and confidential |
| k=64, s=64, arithmetic in Z/(2^(k+s)) | Unsigned native 128-bit wraparound; only low 64-bit results leave the session |
| Random key and secret fresh coefficients | Separate AES-CTR PRSS streams; uniform full-ring words in the ideal-PRG model |
| Fixed records before fresh challenges | Owned pending snapshots; count agreement and copy hashes before coefficient generation |
| Passive multiplication / dot product with additive errors | Replicated local cross terms, fresh zero masks, batched exchange |
| Zero test in the random-oracle model | Protocol 9, with additional domain/copy/status checks; extra rejection is allowed |
| Checked release with abort | Mandatory finalization, staged exports, terminal session failure |

The paper's main protocol gives the statistical term (s+1)*2^-s, which at
s=64 is 65/2^64, approximately 2^-57.98. This is a **hybrid-model reference
bound for incorrect low-64-bit computations**, not a claim that every isolated
high-bit MAC perturbation has that detection probability. Protocol 9 additionally
requires the random-oracle assumption; real BLAKE3 and AES-CTR introduce
computational assumptions. The hash output length alone is not a full protocol
security proof.

There is a material distinction from Protocol 4 step 5(d-e): that presentation
opens the key for its terminal check. This prototype follows the approved
SPDZ-wise session design, retains the key, and computes alpha*w with a secret
multiplication. Its extra additive error enters the residual alongside the dot
product errors. A composable proof must account for that error, repeated
successful checks with a reused key, and authenticated-value reuse. The mapping
above identifies these obligations; unit tests do not discharge them. The
production factory remains closed pending that source/proof review.

For q checks, the conditional union-bound budget is
`min(1, q*65/2^64 + epsilon_hash(q) + epsilon_PRSS(q))`, **provided** the
per-check bound holds conditional on prior accepted transcripts. That
conditioning is part of the reuse proof obligation, not something established
by a test run. Include immediate input checks in q; there is no blanket
40-bit lifetime guarantee for an unbounded session. Client IO still does not
detect replay of a whole previously valid three-share output bundle.

No 192-bit auxiliary sacrifice or random-bit zero-test circuit is used.
Performance measurements are deferred: measure aggregation and Protocol 9
separately, including copy/status exchanges. The earlier 23% estimate for a
different zero checker does not apply.

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

## RING split-3 A2B

The RING prototype registers `a2b` for an active session's authenticated
`ring::AShrTy`. It returns `BShrTy(session, 64)` with the input shape and dtype.
The input is a valid authenticated arithmetic sharing. Conversion reads only
its two value components, reduced to their low 64 bits; it neither converts
MACs nor repeats input authentication. FIELD A2B remains unimplemented.

Following MP-SPDZ `SpdzWiseRingShare::split` and `Rep3Share2::split(case 3)`,
party i embeds component x_i in slot 0 of Boolean summand i and component
x_(i+1) in slot 1 of summand (i+1) mod 3; all other slots are zero. This local
mapping respects this repository's replica ordering and sends no shares.
The circuit computes:

```
s = x0 XOR x1 XOR x2
c = ((x0 XOR x2) AND (x1 XOR x2)) XOR x2
result = s + (c << 1) mod 2^64
```

The retained circuit is a bit-sliced CSA followed by a ripple-carry adder.
Each 64-bit word packs the same bit of 64 separate inputs; a 64x64 transpose
packs and restores the ordinary word representation. The CSA uses 63 ANDs
and the adder 62, for 125 active ANDs per input and 63 online AND rounds.
The last SIMD block is zero-padded on input; its lanes are still checked.
The checker records `125 * ceil(numel / 64)` full AND words.

The carry identity `carry_out = carry_in XOR ((a XOR carry_in) AND
(b XOR carry_in))` needs just one AND per full adder. This is an existing
circuit optimization: [Kolesnikov–Sadeghi–Schneider, CANS 2009, Section 3.1](https://encrypto.de/papers/KSS09.pdf)
describes the one-AND full adder; [ABY3, CCS 2018](https://eprint.iacr.org/2018/403.pdf)
uses the same three-to-two CSA reduction followed by a parallel-prefix adder
for malicious A2B; [SECRECY, NSDI 2023, Appendix A.1](https://www.usenix.org/system/files/nsdi23-liagouris.pdf)
also describes RCA-based arithmetic/Boolean sharing conversion.

Our choice combines these existing techniques with bit-slicing and deferred
ABFL verification. RCA reduces AND work but has a linear carry dependency;
prefix adders reduce communication rounds. In the 100K-input FactoryMem
experiment, CSA + RCA was fastest including final verification; this does not
establish the fastest circuit on high-latency networks. Word-packed AND counts
include unused bit positions and are not minimum circuit gate counts.
Native ABY3 uses a different default adder and is a semi-honest baseline, so
this comparison does not isolate per-gate protocol overhead. See the
[selection and final benchmark report](benchmarks/a2b-split3-100k-2026-10-22/RESULTS.md).

Every secret AND uses the existing ABFL session and its saved verification
records. A2B itself does not finalize or reveal. Callers can compose its output
with Boolean operations and accumulate multiple conversions before one final
ABFL check. Existing pending arithmetic checks still run at finalization;
unchecked output cannot be exported. Wrong domains, sessions or lifecycle
states abort through the normal session mechanism.

There is no minimum A2B input size. Empty inputs need no ANDs. The existing
ABFL finalization requirement of at least 1024 packed AND words still applies
to nonempty queues; small conversions must accumulate enough real circuit work
in the session. No artificial ANDs or new verifier parameters are introduced.
The security assumptions and outstanding ABFL composition review described
above are unchanged by this conversion implementation.

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
