# Checked SPDZ-wise ring comparison versus ABY3

100,000 pairs; FM64; three-party FactoryMem; 64 background workers; one warmup and 7 measured iterations. Fresh native shares and contexts each iteration.

Both protocols call HAL i_less on two prepared DT_I64/FM64 arithmetic secret-share arrays and return a serializable one-bit Boolean secret-share array with DT_I1. Time includes subtraction, native MSB, and output preparation. SPDZ-wise prepareOutputs invokes finalize_protocol -> checkMac + checkBoolean inside timing, including all random material generation and verification communication. Excludes input sharing/authentication, context/PRSS setup, reconstruction and plaintext comparison.

Checked SPDZ-wise comparison versus native semi-honest ABY3 at the same HAL input/output abstraction. All comparison security checks are timed. Protocol-specific internal circuits/layouts remain different; this does not isolate security-model overhead.

| Protocol | Checked total median ms | Range ms | Comparisons/s | Three-party sent bytes |
|---|---:|---:|---:|---:|
| spdzwise | 273.963 | 261.656–305.249 | 365,013 | 361627200 |
| aby3 | 193.877 | 189.551–199.582 | 515,791 | 10800000 |

SPDZ-wise / ABY3 checked total median time: **1.413x**.

Maximum whole checked interval across parties per iteration, then median/min/max across measured iterations. Phase maxima/medians are diagnostic and must not be summed. Traffic sums sent bytes across parties including all finalization traffic.

| Protocol | Circuit ms | Finalization + output ms | Circuit sent bytes | Finalization sent bytes |
|---|---:|---:|---:|---:|
| spdzwise | 23.430 | 250.763 | [9040392] | [352586808] |
| aby3 | 193.875 | 0.002 | [10800000] | [0] |

Two arrays of signed 63-bit inputs in [-2^62, 2^62-1], SplitMix64 seed 0x4d5342434d503634 and 25 edge pairs. Identical plaintext across protocols, fresh native shares and contexts per iteration.

All outputs including warmup checked against modular MSB and signed less-than outside timing. Inside timing, one checkMac call and one checkBoolean call through normal output finalization; all pending Boolean gates are verified. Already authenticated inputs and linear subtraction leave zero new arithmetic MAC batches; checkMac still performs its count/status exchange, included in timing.

Reproduce from this benchmark directory with `bash build.sh //:comparison_bench` and `python3 run_comparison.py NEW_OUTPUT_DIRECTORY --count 100000 --repetitions 7`. Commands, source archive and fingerprints, binary fingerprint, environment, raw samples and resource records are stored alongside this report. These measurements use local-memory transport.
