#!/usr/bin/env python3
"""Exercise the multi-refresh path on a case built here, because no real one exists locally.

The kit ships two practice cases and both are a single flat page, so nothing in the bundle
has two page states to fit a refresh on. This builds a case that does -- tracked players
growing as a power law, each trophy's earners growing with its own elasticity to that
count -- and checks that the program recovers the structure, holds the contract, and is
deterministic. It is a test of the code, not evidence about the real data: the rates it
generates come from the same model the program fits, so its numbers cannot show that the
model is right about GTA VI, only that the machinery works when a case has refreshes.
"""

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ENTRY)

import main as v2  # noqa: E402

STATES_H = (0.5, 4.5, 8.0, 14.0, 20.0, 26.0, 34.0, 42.0)
TARGET_H = 60.0
N0 = 638.0
GROWTH_EXPONENT = 0.35


def build_case():
    with open("/data/workspace/makefun/kits/rf-kit/input/p00.in") as f:
        base = json.load(f)
    rates0 = base["history"][-1]["rates"]
    trophies = base["trophies"]

    def tracked(h):
        return N0 * (max(h, 0.5) / 0.5) ** GROWTH_EXPONENT

    # a per-trophy elasticity to the tracked count, spread around the published 0.1614
    eps = []
    for i in range(len(trophies)):
        w = math.sin(1.7 * (i + 1)) * 0.045      # deterministic, no randomness
        eps.append(max(0.0, 0.1614 + w))

    def earners_at(i, h):
        e0 = round(rates0[i] * N0 / 100.0)
        if rates0[i] >= 100.0:
            return tracked(h)                     # pinned: every tracked player holds it
        return max(0.0, e0 * (tracked(h) / N0) ** eps[i])

    history = []
    for h in STATES_H:
        rates = []
        for i in range(len(trophies)):
            if rates0[i] >= 100.0:
                rates.append(100.0)
            else:
                rates.append(round(100.0 * earners_at(i, h) / tracked(h), 2))
        history.append({"slot": _iso(h), "at": _iso(h), "players_tracked": round(tracked(h)),
                        "trophies_earned": int(sum(r * tracked(h) / 100.0 for r in rates)),
                        "rates": rates})

    case = dict(base)
    case["id"] = "s00"
    case["at"] = _iso(STATES_H[-1])
    case["horizon_minutes"] = int((TARGET_H - STATES_H[-1]) * 60)
    case["target"] = _iso(TARGET_H)
    case["history"] = history
    return case, rates0, earners_at


def _iso(hours):
    """An ISO instant `hours` after the game's opening (2026-11-19T00:00:00Z)."""
    from datetime import datetime, timedelta, timezone
    t = datetime(2026, 11, 19, tzinfo=timezone.utc) + timedelta(hours=hours)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    case, rates0, earners_at = build_case()
    states = v2.page_states(case)
    print("page states built: %d, tracked %s" % (
        len(states), [int(s["tracked"]) for s in states]))

    obs = v2.transition_observations(states)
    pooled, betas = v2.fit_betas(obs, len(case["trophies"]))
    dlog_n = v2.tracked_growth(states, TARGET_H)
    print("fitted pooled beta %.3f (built with %.3f), dlogN to target %.3f"
          % (pooled, -(1 - 0.1614), dlog_n))

    rows = v2.forecast_case(case)
    n = len(case["trophies"])
    truth = []
    for i in range(n):
        if rates0[i] >= 100.0:
            truth.append(100.0)
        else:
            truth.append(round(100.0 * earners_at(i, TARGET_H) / (N0 * (TARGET_H / 0.5) ** GROWTH_EXPONENT), 2))

    def logit(x):
        p = min(99.99, max(0.01, x)) / 100.0
        return math.log(p / (1 - p))

    def pinball(y, q, tau):
        d = logit(y) - logit(q)
        return 2 * max(tau * d, (tau - 1) * d)

    items, cover = [], 0
    for r, y in zip(rows, truth):
        q = r["q"]
        cover += 1 if q[0] <= y <= q[4] else 0
        for tau, qi in zip((0.05, 0.25, 0.5, 0.75, 0.95), q):
            items.append(pinball(y, qi, tau))
    loss = sum(items) / len(items)
    flat_items = []
    for i, y in enumerate(truth):
        last = case["history"][-1]["rates"][i]
        for tau in (0.05, 0.25, 0.5, 0.75, 0.95):
            flat_items.append(pinball(y, last, tau))
    print("held-out loss at the target: v2 %.4f, flat %.4f, 90%% coverage %.3f"
          % (loss, sum(flat_items) / len(flat_items), cover / n))
    ids = [t["id"] for t in case["trophies"]]
    ok = (len(rows) == n and [r["id"] for r in rows] == ids
          and all(len(r["q"]) == 5 and all(r["q"][k] <= r["q"][k + 1] for k in range(4))
                  and all(0 <= v <= 100 for v in r["q"]) for r in rows))
    again = v2.forecast_case(json.loads(json.dumps(case)))
    print("contract ok:", ok, "| deterministic:", again == rows)


if __name__ == "__main__":
    main()
