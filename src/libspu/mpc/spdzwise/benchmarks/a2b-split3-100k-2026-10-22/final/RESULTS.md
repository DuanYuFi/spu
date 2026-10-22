# RING split-3 A2B versus native ABY3

100,000 full-width 64-bit inputs; three-party FactoryMem; 64 background workers; one warmup and 7 measured iterations. Fresh context and native input shares every iteration. All outputs, including warmup, were reconstructed and compared exactly, outside timing.

SPDZwise timing includes conversion and one final ABFL verification, including fresh verification material. Native ABY3 timing includes its complete registered A2B operation. Input authentication, protocol initialization and output reconstruction are excluded. ABY3 is a semi-honest baseline.

| Protocol | Total ms | Range ms | Conversion ms | Verification ms | Total sent bytes (3 parties) | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| spdzwise | 134.182 | 129.609–144.828 | 12.950 | 121.232 | 187,567,008 | 431.4 |
| aby3 | 146.386 | 125.306–149.766 | 146.385 | 0.000 | 33,600,000 | 105.7 |

Each sample is the maximum whole interval across the three parties; the table reports medians and ranges. Separately aggregated phase medians cannot be summed. CSV includes per-party phase traffic and message counts. Peak RSS covers the whole three-party process, including input preparation.

spdzwise / ABY3: total time **0.917x**, conversion time **0.088x**, total traffic **5.582x**.

Reproduce with `bash build.sh //:conversion_bench` and `python3 run_conversion.py NEW_OUTPUT_DIRECTORY`. Commands, source/binary fingerprints, environment, raw samples and resources are archived alongside this report. These are local-memory transport results.
