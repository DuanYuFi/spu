# RING split-3 B2A versus native ABY3

100,000 full-width 64-bit inputs; three-party FactoryMem; 64 background workers; one warmup and 7 measured iterations. Fresh context and native input shares every iteration. All outputs, including warmup, were reconstructed and compared exactly, outside timing.

B2A includes fresh random mask/MAC generation, split-3 A2B, Boolean addition, arithmetic and one ABFL check before opening, opening and linear output; native ABY3 B2A only. Exclude external input preparation/import, context/PRSS setup, export/reconstruction/comparison. ABY3 is a semi-honest baseline. For B2A, verification runs inside the conversion before masked opening; `online_ms` means the remaining measured time, including material generation. Both protocols start from fresh XOR RSS in their native types.

| Protocol | Total ms | Range ms | Conversion ms | Verification ms | Total sent bytes (3 parties) | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| spdzwise | 250.632 | 232.225–260.404 | 61.325 | 192.303 | 296,500,854 | 447.7 |
| aby3 | 136.924 | 123.232–166.031 | 136.924 | 0.000 | 36,800,000 | 119.1 |

Each sample is the maximum whole interval across the three parties; the table reports medians and ranges. Separately aggregated phase medians cannot be summed. CSV includes per-party phase traffic and message counts. Peak RSS covers the whole three-party process, including input preparation.

| B2A phase | Median ms | Three-party sent bytes |
|---|---:|---:|
| material | 13.067 | 4,800,012 |
| circuit | 20.936 | 7,052,256 |
| check | 192.303 | 275,048,394 |
| open | 31.739 | 9,600,192 |

Material includes mask authentication; check includes arithmetic verification and one ABFL batch; open includes checked opening and arithmetic subtraction. Phase medians must not be summed.

spdzwise / ABY3: total time **1.830x**, conversion time **0.448x**, total traffic **8.057x**.

Reproduce with `bash build.sh //:conversion_bench` and `python3 run_conversion.py NEW_OUTPUT_DIRECTORY --operation b2a`. Commands, source/binary fingerprints, environment, raw samples and resources are archived alongside this report. These are local-memory transport results.

## Validation and provenance

All six regression targets passed: conversion 11, protocol 11, Boolean 16,
RING 15, truncation 7, session 7 (67 cases). The conversion tests include
zero-extension, strided views, MAC reconstruction, consumers and round trips,
the 320/321-element batch boundary, three party positions, corrupt padding,
and actual masked-opening wire equivocation. Preconditions failing arithmetic
or ABFL verification never reach the `b2a-open` message. No production factory
gate was changed.

Both default A2B and `--operation b2a` benchmark smoke runs passed with 1,025
inputs and one measured repetition. Their artifacts and the full regression
XML/logs are in `validation/`; reproduction commands and the successful build
log are included. `source.patch` captures the tracked implementation and
related documentation changes relative to the revision in `parameters.json`.
The reference MP-SPDZ checkout is commit
`113c4e47dd94bca9ecab029d2f0abe649bea5c52`.

The full benchmark verified 800,000 outputs per protocol, including warmup.
Each SPDZ-wise run checked one arithmetic mask batch and one ABFL batch before
opening, recorded 18,806,016 AND bits per party (including tail padding), and
performed no second nonempty verification during export. `resources.json`
files record whole-process CPU time and peak memory, including setup and
plaintext validation; these are distinct from the timed conversion intervals.
