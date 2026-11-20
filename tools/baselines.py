#!/usr/bin/env python3
"""The two baselines the rewrite is measured against, as forecasters.

  flat          every quantile is the trophy's rate in the case's last read -- the kit's
                persistence baseline with no spread at all, so its loss is exactly the
                mean |logit error| (the kit publishes 1.51 for it on the practice).
  flat_spread   the same centre, with v2's calibrated spread: what to ship if flat wins.
  linear        a least-squares line of logit(rate) on ln(hours since the opening) over
                every read, read at the target, with the same spread. This is the kit's
                "house curve baseline"; on a flat history its line is flat and it collapses
                onto flat_spread, which is what the kit says it does.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import main as v2  # noqa: E402


def _spread(case, states, target_h):
    """v2's residual sample for this case, so the baselines differ only in their centre."""
    n = len(case.get("trophies") or [])
    obs = v2.transition_observations(states)
    if len(states) >= 2:
        dlog_n = v2.tracked_growth(states, target_h)
        per, pooled = v2.refresh_residuals(states, obs, n, dlog_n)
    else:
        per, pooled = v2.allocation_residuals(states, n)
    return per, pooled


def _target_h(case, states):
    t0 = v2.opening_time(case)
    target = v2.parse_time(case.get("target"))
    return (target - t0).total_seconds() / 3600.0 if target else states[-1]["last_h"]


def flat(case):
    states = v2.page_states(case)
    if not states:
        return [{"id": t.get("id"), "q": [1.0] * 5} for t in case.get("trophies") or []]
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        r = v2.rate_at(states[-1], i)
        if r is None:
            r = 0.01
        q = round(min(99.99, max(0.01, r)), 4)
        rows.append({"id": trophy.get("id"), "q": [q] * 5})
    return rows


def flat_spread(case):
    states = v2.page_states(case)
    if not states:
        return flat(case)
    target_h = _target_h(case, states)
    per, pooled = _spread(case, states, target_h)
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        r = v2.rate_at(states[-1], i)
        if r is None or r <= 0.0:
            rows.append({"id": trophy.get("id"), "q": [0.01] * 5})
            continue
        sample = per[i] if len(per[i]) >= v2.MIN_OWN_RESIDUALS else pooled
        rows.append({"id": trophy.get("id"),
                     "q": v2.quantiles(v2.logit(r), sample, r, r)})
    return rows


def linear(case):
    """logit(rate) on ln(hours), least squares over every read, read at the target."""
    states = v2.page_states(case)
    if not states:
        return flat(case)
    target_h = _target_h(case, states)
    per, pooled = _spread(case, states, target_h)
    # every read, not every page: the kit's curve baseline fits the reads it was given
    t0 = v2.opening_time(case)
    reads = []
    for read in case.get("history") or []:
        t = v2.parse_time(read.get("slot") or read.get("at"))
        if t is None:
            continue
        reads.append(((t - t0).total_seconds() / 3600.0, read.get("rates") or []))
    reads.sort(key=lambda r: r[0])
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        pts = []
        for hours, rates in reads:
            if i < len(rates) and rates[i] is not None and float(rates[i]) > 0.0:
                pts.append((math.log(max(hours, v2.TIME_FLOOR_H)), v2.logit(float(rates[i]))))
        r_last = v2.rate_at(states[-1], i)
        if len(pts) < 3 or r_last is None or r_last <= 0.0:
            rows.append({"id": trophy.get("id"), "q": [round(min(99.99, max(0.01, r_last or 0.01)), 4)] * 5})
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        den = sum((p[0] - mx) ** 2 for p in pts)
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den if den > 1e-12 else 0.0
        # anchor the line at the last observed level, as the kit's baseline does
        z = v2.logit(r_last) + slope * (math.log(max(target_h, v2.TIME_FLOOR_H))
                                        - math.log(max(states[-1]["last_h"], v2.TIME_FLOOR_H)))
        z = min(v2.logit(99.99), z)
        sample = per[i] if len(per[i]) >= v2.MIN_OWN_RESIDUALS else pooled
        rows.append({"id": trophy.get("id"), "q": v2.quantiles(z, sample, r_last, 100.0 * v2.inv_logit(z))})
    return rows
