"""Compare saved before/after ring_circuit_bench binaries in adjacent runs.

Usage: python3 run_ring_comparison.py BEFORE_BINARY AFTER_BINARY NEW_DIRECTORY
Each invocation includes one warmup and seven measured, fully checked runs.
Order alternates between circuits. All invocations use 64 background workers.
"""

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_server import CIRCUITS, digest, summarize


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    binaries = dict(zip(("before", "after"), map(lambda s: Path(s).resolve(), sys.argv[1:3])))
    out = Path(sys.argv[3]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    params = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "binaries": {v: {"path": str(p), "sha256": digest(p)} for v, p in binaries.items()},
        "YACL_NUM_THREADS": "65", "background_workers": 64,
        "warmups": 1, "repetitions": 7, "loadavg_at_start": os.getloadavg(),
        "method": "adjacent before/after invocations, alternating order by circuit; phase maxima across parties then medians over iterations 1..7",
        "script_sha256": {n: digest(Path(__file__).with_name(n)) for n in ("run_ring_comparison.py", "run_server.py")},
    }
    (out / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    summaries = {}
    for index, (circuit, (width, depth)) in enumerate(CIRCUITS.items()):
        pair = {}
        for version in (("after", "before") if index % 2 else ("before", "after")):
            name = f"{circuit}-{version}"
            command = [str(binaries[version]), str(width), str(depth), "7"]
            print(name, flush=True)
            with (out / f"{name}.stdout").open("w") as stdout, (out / f"{name}.log").open("w") as stderr:
                subprocess.run(command, env={**os.environ, "YACL_NUM_THREADS": "65"}, stdout=stdout, stderr=stderr, check=True)
            raw = (out / f"{name}.stdout").read_text()
            assert "Create a fixed thread pool with size 64" in raw
            lines = [s for s in raw.splitlines(True) if s.startswith(("protocol,", "spdzwise-ring128,"))]
            (out / f"{name}.csv").write_text("".join(lines))
            pair[version] = summarize(list(csv.DictReader(lines)), "spdzwise-ring128", circuit, True, False)
        pair["reduction_percent"] = {
            key: 100 * (1 - pair["after"]["phases"][key]["median_ms"] / pair["before"]["phases"][key]["median_ms"])
            for key in ("circuit_ms", "mul_ms", "verify_ms", "core_ms", "total_ms")
        }
        assert pair["before"]["bytes_per_party"] == pair["after"]["bytes_per_party"]
        summaries[circuit] = pair
        print(circuit, pair["reduction_percent"], flush=True)
    assert all(digest(p) == params["binaries"][v]["sha256"] for v, p in binaries.items())
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()))


if __name__ == "__main__":
    main()
