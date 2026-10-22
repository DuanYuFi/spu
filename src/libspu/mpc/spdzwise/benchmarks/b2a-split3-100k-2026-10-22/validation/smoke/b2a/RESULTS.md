# RING split-3 B2A versus native ABY3

1,025 full-width 64-bit inputs; three-party FactoryMem; 64 background workers; one warmup and 1 measured iterations. Fresh context and native input shares every iteration. All outputs, including warmup, were reconstructed and compared exactly, outside timing.

B2A includes fresh random mask/MAC generation, split-3 A2B, Boolean addition, arithmetic and one ABFL check before opening, opening and linear output; native ABY3 B2A only. Exclude external input preparation/import, context/PRSS setup, export/reconstruction/comparison. ABY3 is a semi-honest baseline. For B2A, verification runs inside the conversion before masked opening; `online_ms` means the remaining measured time, including material generation. Both protocols start from fresh XOR RSS in their native types.

| Protocol | Total ms | Range ms | Conversion ms | Verification ms | Total sent bytes (3 parties) | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| spdzwise | 6.264 | 6.264–6.264 | 2.712 | 3.553 | 4,217,658 | 30.1 |
| aby3 | 1.376 | 1.376–1.376 | 1.376 | 0.000 | 377,200 | 30.1 |

Each sample is the maximum whole interval across the three parties; the table reports medians and ranges. Separately aggregated phase medians cannot be summed. CSV includes per-party phase traffic and message counts. Peak RSS covers the whole three-party process, including input preparation.

| B2A phase | Median ms | Three-party sent bytes |
|---|---:|---:|
| material | 0.108 | 49,212 |
| circuit | 2.457 | 76,704 |
| check | 3.553 | 3,993,150 |
| open | 0.122 | 98,592 |

Material includes mask authentication; check includes arithmetic verification and one ABFL batch; open includes checked opening and arithmetic subtraction. Phase medians must not be summed.

spdzwise / ABY3: total time **4.552x**, conversion time **1.970x**, total traffic **11.181x**.

Reproduce with `bash build.sh //:conversion_bench` and `python3 run_conversion.py NEW_OUTPUT_DIRECTORY --operation b2a`. Commands, source/binary fingerprints, environment, raw samples and resources are archived alongside this report. These are local-memory transport results.
