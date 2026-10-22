# RING split-3 A2B versus native ABY3

100,000 full-width 64-bit inputs; three-party FactoryMem; 64 background workers; one warmup and 7 measured iterations. Fresh context and native input shares every iteration. All outputs, including warmup, were reconstructed and compared exactly, outside timing.

SPDZwise timing includes conversion and one final ABFL verification, including fresh verification material. Native ABY3 timing includes its complete registered A2B operation. Input authentication, protocol initialization and output reconstruction are excluded. ABY3 is a semi-honest baseline.

| Protocol | Total ms | Range ms | Conversion ms | Verification ms | Total sent bytes (3 parties) | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| rca | 144.831 | 143.177–149.935 | 15.967 | 130.408 | 187,567,008 | 430.5 |
| bk | 305.753 | 299.929–374.700 | 23.991 | 282.674 | 436,652,832 | 574.4 |
| ks | 625.534 | 579.474–681.485 | 76.555 | 555.411 | 842,423,520 | 925.0 |
| aby3 | 142.011 | 133.413–159.610 | 142.011 | 0.000 | 33,600,000 | 104.9 |

Each sample is the maximum whole interval across the three parties; the table reports medians and ranges. Separately aggregated phase medians cannot be summed. CSV includes per-party phase traffic and message counts. Peak RSS covers the whole three-party process, including input preparation.

rca / ABY3: total time **1.020x**, conversion time **0.112x**, total traffic **5.582x**.

bk / ABY3: total time **2.153x**, conversion time **0.169x**, total traffic **12.996x**.

ks / ABY3: total time **4.405x**, conversion time **0.539x**, total traffic **25.072x**.

Historical candidate comparison. The losing implementations and candidate selection flags were subsequently removed. See [the selection report](../RESULTS.md) for their structures and [the final benchmark](../final/RESULTS.md) for current reproduction commands. Source/binary fingerprints and raw samples are preserved here.
