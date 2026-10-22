# RING split-3 A2B versus native ABY3

1,025 full-width 64-bit inputs; three-party FactoryMem; 64 background workers; one warmup and 1 measured iterations. Fresh context and native input shares every iteration. All outputs, including warmup, were reconstructed and compared exactly, outside timing.

A2B plus one final ABFL verification; native ABY3 A2B only. Exclude input sharing/authentication, context/PRSS setup, export/reconstruction/comparison. ABY3 is a semi-honest baseline. 

| Protocol | Total ms | Range ms | Conversion ms | Verification ms | Total sent bytes (3 parties) | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| spdzwise | 5.778 | 5.778–5.778 | 2.290 | 3.570 | 2,707,032 | 30.2 |
| aby3 | 1.631 | 1.631–1.631 | 1.631 | 0.000 | 344,400 | 30.2 |

Each sample is the maximum whole interval across the three parties; the table reports medians and ranges. Separately aggregated phase medians cannot be summed. CSV includes per-party phase traffic and message counts. Peak RSS covers the whole three-party process, including input preparation.

spdzwise / ABY3: total time **3.542x**, conversion time **1.404x**, total traffic **7.860x**.

Reproduce with `bash build.sh //:conversion_bench` and `python3 run_conversion.py NEW_OUTPUT_DIRECTORY --operation a2b`. Commands, source/binary fingerprints, environment, raw samples and resources are archived alongside this report. These are local-memory transport results.
