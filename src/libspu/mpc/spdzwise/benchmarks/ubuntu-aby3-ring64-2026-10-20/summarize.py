"""Validate recorded circuit-benchmark CSVs and summarize phase maxima."""

import csv
import json
import statistics
from pathlib import Path


def stats(values):
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main():
    directory = Path(__file__).resolve().parent
    summaries = []
    for path in sorted(directory.glob("w*-d*.stdout")):
        # YACL's thread-pool startup log also goes to stdout. Preserve that
        # raw stream, but put only the CSV header and data in the .csv file.
        lines = [line for line in path.read_text().splitlines(keepends=True)
                 if line.startswith(("protocol,", "aby3-ring64,"))]
        csv_path = path.with_suffix(".csv")
        csv_path.write_text("".join(lines))
        rows = list(csv.DictReader(lines))
        assert len(rows) == 21, (path, len(rows))
        for protocol in ("aby3-ring64",):
            samples = [r for r in rows if r["protocol"] == protocol]
            wise = protocol == "spdzwise-f61"
            assert len(samples) == 21
            width, depth = int(samples[0]["width"]), int(samples[0]["depth"])
            gates = width * depth
            assert gates == 1048576
            for r in samples:
                assert int(r["width"]) == width and int(r["depth"]) == depth
                assert int(r["gates"]) == gates and int(r["secret_inputs"]) == 2
                assert int(r["checks"]) == int(wise)
                assert int(r["pending"]) == (gates + 2 if wise else 0)
                assert int(r["input_mac_bytes"]) == (16 if wise else 0)
                assert int(r["expand_bytes"]) == 0
                assert int(r["circuit_bytes"]) == gates * (16 if wise else 8)
                assert int(r["verify_bytes"]) == (64 if wise else 0)
                assert abs(float(r["circuit_ms"]) - float(r["mul_ms"])
                           - float(r["local_ms"])) < 0.00001
            phases = {k: [] for k in ("input_mac_ms", "expand_ms", "circuit_ms",
                                      "mul_ms", "local_ms", "verify_ms")}
            core, total = [], []
            assert {int(r["iteration"]) for r in samples} == set(range(7))
            for iteration in range(7):
                batch = [r for r in samples if int(r["iteration"]) == iteration]
                assert len(batch) == 3 and {int(r["rank"]) for r in batch} == {0, 1, 2}
                maxima = {k: max(float(r[k]) for r in batch) for k in phases}
                for key, value in maxima.items():
                    phases[key].append(value)
                core.append(maxima["circuit_ms"] + maxima["verify_ms"])
                total.append(core[-1] + maxima["input_mac_ms"] + maxima["expand_ms"])
            summaries.append({
                "source": csv_path.name,
                "protocol": protocol,
                "width": width,
                "depth": depth,
                "multiplication_gates": gates,
                "secret_inputs": 2,
                "checked_elements": gates + 2 if wise else 0,
                "phases": {k: stats(v) for k, v in phases.items()},
                "core": stats(core),
                "all_timed_phases": stats(total),
            })
    assert len(summaries) == 1
    output = {
        "aggregation": "For each repetition, take the maximum over three parties "
                       "for each phase; summarize the seven maxima. Core and total "
                       "sum phase maxima within each repetition before summarizing.",
        "runs": summaries,
    }
    (directory / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    for r in summaries:
        print(r["width"], r["depth"], r["protocol"],
              {k: round(v["median_ms"], 3) for k, v in r["phases"].items()},
              "core", round(r["core"]["median_ms"], 3),
              "total", round(r["all_timed_phases"]["median_ms"], 3))


if __name__ == "__main__":
    main()
