#!/usr/bin/env python3
"""Time the program on a case the size the final will be, not the size the practice is.

The final's f00 is cut at the deadline, 2.5 days in, with a read every 30 minutes on launch
night and every 10 minutes after: a few hundred reads and a few dozen distinct pages. This
builds one of that shape (a code exercise, from the same generator as
test_multi_refresh.py) and times `main.py` on it, because the rules' cap is 8 seconds and
the practice's 7 reads say nothing about that.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.dirname(HERE)
sys.path.insert(0, ENTRY)

import main as v2  # noqa: E402


def build(reads=360, states=30, hours=60.0):
    with open("/data/workspace/makefun/kits/rf-kit/input/p00.in") as f:
        base = json.load(f)
    rates0 = base["history"][-1]["rates"]
    trophies = base["trophies"]
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 11, 19, tzinfo=timezone.utc)

    def iso(h):
        return (t0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # a page that changes `states` times across `reads` reads: every read inside one page
    # is byte-identical, as Exophase's are between refreshes
    history = []
    for k in range(reads):
        h = 0.2 + (hours - 0.2) * k / (reads - 1)
        s = min(states - 1, int(states * k / reads))
        state_h = 0.2 + (hours - 0.2) * (s * reads / states) / (reads - 1)
        tracked = 638.0 * (max(state_h, 0.5) / 0.5) ** 0.35
        rates = []
        for i, r0 in enumerate(rates0):
            if r0 >= 100.0:
                rates.append(100.0)
            else:
                e0 = round(r0 * 638.0 / 100.0)
                rates.append(round(100.0 * e0 * (tracked / 638.0) ** 0.16 / tracked, 2))
        history.append({"slot": iso(h), "at": iso(h), "players_tracked": round(tracked),
                        "trophies_earned": int(sum(rates) * tracked / 100.0), "rates": rates})
    case = dict(base)
    case["id"] = "x00"
    case["at"] = iso(hours)
    case["horizon_minutes"] = 6510
    case["target"] = "2026-11-26T00:00:00Z"
    case["history"] = history
    return case


def main():
    case = build()
    n_states = len(v2.page_states(case))
    print("stress case: %d reads, %d page states, %d trophies, horizon 6510 min"
          % (len(case["history"]), n_states, len(case["trophies"])))
    t0 = time.perf_counter()
    rows = v2.forecast_case(case)
    ms = (time.perf_counter() - t0) * 1000.0
    print("forecast_case: %.0f ms, %d rows" % (ms, len(rows)))

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in")
        out = os.path.join(d, "out")
        os.makedirs(src)
        with open(os.path.join(src, "x00.in"), "w") as f:
            json.dump(case, f)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, os.path.join(ENTRY, "main.py")], check=True,
                       env=dict(os.environ, MAKEFUN_INPUT_DIR=src, MAKEFUN_OUTPUT_DIR=out))
        wall = (time.perf_counter() - t0) * 1000.0
        size = os.path.getsize(os.path.join(out, "x00.out"))
    print("whole program (interpreter start included): %.0f ms, output %d bytes"
          % (wall, size))
    print("the rules' cap is 8000 ms per run")


if __name__ == "__main__":
    main()
