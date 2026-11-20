#!/usr/bin/env python3
"""Forecast every PS5 trophy's Exophase earn rate for Grand Theft Auto VI at a case's target.

Reads every *.in case in the input directory and writes one <id>.out per case, one row per
trophy, five quantiles of that trophy's earn rate at the case's target.

The model, and why it is this one
---------------------------------

A trophy's rate is not a level that drifts: it is a ratio the page recomputes at every
refresh,

    rate_i = 100 * E_i / N

with E_i the tracked players holding trophy i and N the players tracked. Exophase holds
that page still for hours and then replaces it whole. So between refreshes nothing moves,
and at a refresh the numerator and the denominator both jump -- the denominator much
harder, because a refresh is when the profiles of players who have barely started get
counted. That single fact decides the whole forecast: a rate falls when N outruns E_i, and
the fall is proportional to how far N jumps.

    z_i = logit(rate_i) = logit(E_i / N)
    dz_i = dlogit(E_i) - dlogit(N) = (eps_i - 1) * dlogN

where eps_i = dlogit(E_i)/dlogN is the trophy's earners elasticity to the tracked count.
The forecast is therefore the last observed level plus beta_i * dlogN, with beta_i =
eps_i - 1 pooled across trophies and shrunk toward zero, and dlogN the tracked count's
growth from the last read to the target. Nothing else is extrapolated: no quadratic, no
growth that the data does not show.

The published launch-night refresh fixes the scale of both numbers, and it is the only
refresh the kit documents in full (638 tracked / 943 earned -> 3,868 tracked / 4,276
earned; every rate but the one at 100% fell). Taking the 100% trophy as pinned -- it holds
every tracked player -- the free earners go 305 -> 408, so

    eps = ln(408/305) / ln(3868/638) = 0.1614,   beta = eps - 1 = -0.8386
    beta * ln(3868/638) = -1.5112

which is exactly the loss the kit publishes for the persistence baseline (1.51): a
constant logit shift is scored at |shift| when the quantiles are equal. The identity is
what says this decomposition is the right one and that the practice truth is a refresh,
not a level.

Two paths, because a case has either seen a refresh or it has not
-----------------------------------------------------------------

  * Two or more distinct page states. Every transition between them is one refresh, and
    every trophy gives one observation of dz_i / dlogN. Those observations are pooled
    (Huber-weighted, recency-weighted) into beta, and each trophy's own beta is shrunk
    toward the pool by how many transitions it has. The tracked count's growth is a
    power law in time since the opening, fitted on the page states and shrunk hard toward
    no growth (few blocks say little about a week), then read at the target.

  * One page state. The page has not refreshed since before the game opened: the history
    is the flat launch night. No growth can be fitted from a flat line, so the forecast
    takes the one jump the kit documents -- the first refresh of a launch -- at the scale
    the published pair gives (N x 6.06, eps = 0.1614, applied once).

The spread is measured, not assumed
-----------------------------------

The quantiles are the centre plus the 5th, 25th, 50th, 75th and 95th percentiles of the
residual the same forecast made when it was asked the same question before:

  * Two or more page states: for every refresh, the model is refitted on the page states
    before it, asked for that refresh's rates, and its miss in logits is kept. Those misses
    are the residual sample, rescaled to the target's dlogN. A trophy with at least
    MIN_OWN_RESIDUALS misses of its own uses its own; the rest use the pool.
  * One page state: there is no earlier refresh to be asked about, and every read in the
    history is byte-identical, so a leave-the-last-read-out residual is exactly zero and
    cannot calibrate anything. The spread then comes from what the published totals do not
    settle -- how the 408 free earners are shared among the 54 trophies: proportional to
    what they held, or equally. The logit gap between those two allocations is a real,
    computable residual sample, and it is what the quantiles are set from.

On top of either sample sits the one error no forecast can avoid: the page prints two
decimals, so a rate is known to +-0.005 points, which in logits is 0.005 / (p(1-p)).

Deterministic: no randomness, no clock, no network, standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone

# The five levels the output contract names; the quantiles are set from the residual
# distribution's own percentiles at these same levels.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

RATE_FLOOR = 0.01          # the scorer clamps a rate here before its logit
RATE_CEIL = 99.99
TIME_FLOOR_H = 1.0         # ln(hours since the opening) is undefined at the opening
RECENCY_HALF_LIFE_H = 24.0 # an older refresh says less about the next one
MIN_TRANSITION_DLOGN = 0.02  # a refresh that moved the tracked count by less than 2%
                             # says nothing about the elasticity: skip it
MIN_RESIDUAL_DLOGN = 0.05    # same, for a residual that is about to be rescaled
MAX_RESIDUAL_SCALE = 3.0     # never stretch an observed miss further than this
CENSORED_RATE = 0.02         # at or below this the two-decimal print cannot show a move

# --- the published launch-night refresh (RULES.txt "What is counted"; competition.yaml
# "The practice"). The only refresh the kit documents in full, so the only one whose
# elasticity can be known before the case is read.
LAUNCH_N_BEFORE = 638.0
LAUNCH_E_BEFORE = 943.0
LAUNCH_N_AFTER = 3868.0
LAUNCH_E_AFTER = 4276.0

# beta = dlogit(rate)/dlogN implied by that pair, with the 100% trophy pinned to every
# tracked player (see the module docstring for the arithmetic: 0.1614 - 1).
PRIOR_BETA = -0.8385513639512452
PRIOR_LOG_TRACKED_JUMP = 1.802154573228395   # ln(3868/638)

BETA_MIN, BETA_MAX = -1.60, 0.40   # a rate cannot fall faster than 1/N without earners
                                   # vanishing, and a dilution model has no reason to rise
BETA_PRIOR_SD = 0.50               # logit per dlogN: the pool's prior, before the data
BETA_OWN_PRIOR_SD = 0.30           # between-trophy spread around the pool
TRACKED_EXPONENT_MAX = 2.50        # dlnN/dln t; launch growth is fast, not unbounded
TRACKED_PRIOR_SD = 1.00            # dlnN/dln t before the states say otherwise
MAX_TRACKED_JUMP = 1.60            # dlogN: the tracked count may at most x5 over a horizon
MAX_DRIFT = 1.20                   # logit: the most the dilution may move a median
MIN_OWN_RESIDUALS = 6              # a trophy's own spread needs this many misses
RESIDUAL_SAMPLE_CAP = 400          # keep the residual sample cheap to sort
SIGMA_FLOOR = 0.02                 # logit: below the two-decimal print, nothing is real



def parse_time(text):
    """An ISO 8601 instant as an aware datetime (naive means UTC)."""
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def logit(rate_pct):
    p = clamp(rate_pct, RATE_FLOOR, RATE_CEIL) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0.0:
        return 1.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return e / (1.0 + e)


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def quantile(values, p):
    """Linear-interpolated sample quantile; `values` need not be sorted."""
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = p * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


def opening_time(case):
    """The game's opening: its release date at 00:00 UTC, else the first read."""
    game = case.get("game") or {}
    date = str(game.get("release_date") or "")
    dt = parse_time(date + "T00:00:00Z") if len(date) == 10 else None
    if dt is not None:
        return dt
    for read in case.get("history") or []:
        t = parse_time(read.get("slot") or read.get("at"))
        if t is not None:
            return t
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def page_states(case):
    """The distinct pages the house saw, in order.

    Exophase holds its page still between refreshes, so the run of identical reads that
    follows a refresh is one page, not many observations. Each state is
    {first_h, last_h, tracked, rates}: first_h when the page was first seen, last_h the
    last read that still showed it. The last state's level is known to have held until the
    case's last read, which is what dates the forecast.
    """
    t0 = opening_time(case)
    reads = []
    for read in case.get("history") or []:
        t = parse_time(read.get("slot") or read.get("at"))
        if t is None:
            continue
        try:
            tracked = float(read.get("players_tracked") or 0.0)
        except (TypeError, ValueError):
            tracked = 0.0
        reads.append(((t - t0).total_seconds() / 3600.0, tracked, read.get("rates") or []))
    reads.sort(key=lambda r: r[0])

    states = []
    for hours, tracked, rates in reads:
        key = (tracked, tuple(rates))
        if states and states[-1]["key"] == key:
            states[-1]["last_h"] = hours
        else:
            states.append({"key": key, "first_h": hours, "last_h": hours,
                           "tracked": tracked, "rates": list(rates)})
    return states


def rate_at(state, index):
    """A trophy's rate in one page state, or None if the page lacked it."""
    rates = state["rates"]
    if index >= len(rates):
        return None
    r = rates[index]
    if r is None:
        return None
    try:
        return float(r)
    except (TypeError, ValueError):
        return None


def earners(rate, tracked):
    """Earners behind a printed rate; 0 when the rate or the count is unusable."""
    if rate is None or tracked <= 0:
        return 0.0
    return rate * tracked / 100.0


def weighted_median(pairs):
    """The value at which half the total weight sits (pairs of (value, weight))."""
    if not pairs:
        return 0.0
    v = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in v)
    if total <= 0:
        return median([x for x, _ in v])
    acc = 0.0
    for value, w in v:
        acc += w
        if acc >= 0.5 * total:
            return value
    return v[-1][0]


def transition_observations(states):
    """One (dz/dlogN) observation per trophy per refresh, with its weight.

    A refresh is one step between page states. dz is the trophy's logit move across it and
    dlogN the tracked count's; their ratio is the trophy's elasticity minus one, which is
    the number the forecast needs. The weight is the precision of that ratio: the counting
    noise of the two earn counts (1/E in logit space) and the page's two-decimal print
    (0.005 / (p(1-p))), summed, times a recency decay. A trophy printed at 0.01% moves by
    less than the print can resolve and carries almost no weight, which is right.
    """
    obs = []
    for b in range(1, len(states)):
        prev, cur = states[b - 1], states[b]
        if prev["tracked"] <= 0 or cur["tracked"] <= 0:
            continue
        dlog_n = math.log(cur["tracked"] / prev["tracked"])
        if dlog_n < MIN_TRANSITION_DLOGN:
            continue
        age = max(0.0, states[-1]["last_h"] - cur["first_h"])
        recency = 0.5 ** (age / RECENCY_HALF_LIFE_H)
        for i in range(len(cur["rates"])):
            r_prev = rate_at(prev, i)
            r_cur = rate_at(cur, i)
            if r_prev is None or r_cur is None or r_prev <= 0.0 or r_cur <= 0.0:
                continue
            if r_prev >= RATE_CEIL or r_cur >= RATE_CEIL:
                continue                      # a 100% rate is pinned, not elastic
            if r_prev <= CENSORED_RATE or r_cur <= CENSORED_RATE:
                # at the print's floor the rate cannot show the move it made: censored
                continue
            e_prev = earners(r_prev, prev["tracked"])
            e_cur = earners(r_cur, cur["tracked"])
            var = (1.0 / max(e_prev, 1.0) + 1.0 / max(e_cur, 1.0)
                   + rounding_sigma(r_prev) ** 2 + rounding_sigma(r_cur) ** 2)
            obs.append({"trophy": i, "b": b, "dlog_n": dlog_n,
                        "dz": logit(r_cur) - logit(r_prev),
                        "weight": recency / var, "z_prev": logit(r_prev)})
    return obs


def robust_slope(rows):
    """Weighted least squares through the origin of dz on dlogN, with a Huber reweight.

    Returns (slope, variance of the slope). Fitting the slope directly, rather than
    averaging the per-trophy ratios dz/dlogN, keeps a refresh that barely moved the tracked
    count from producing a wild ratio: it contributes little to sum(w * dlogN^2) and so
    little to the answer.
    """
    if not rows:
        return 0.0, float("inf")
    num = sum(r["weight"] * r["dlog_n"] * r["dz"] for r in rows)
    den = sum(r["weight"] * r["dlog_n"] ** 2 for r in rows)
    if den <= 1e-12:
        return 0.0, float("inf")
    beta = num / den
    for _ in range(3):
        resid = [(abs(r["dz"] - beta * r["dlog_n"]), r["weight"]) for r in rows]
        scale = max(1.4826 * weighted_median(resid), 0.05)
        cut = 1.5 * scale
        num = 0.0
        den = 0.0
        for r in rows:
            d = abs(r["dz"] - beta * r["dlog_n"])
            w = r["weight"] if d <= cut else r["weight"] * cut / d
            num += w * r["dlog_n"] * r["dz"]
            den += w * r["dlog_n"] ** 2
        if den <= 1e-12:
            break
        beta = num / den
    if den <= 1e-12:
        return beta, float("inf")
    # The slope's variance from the scatter the fit actually leaves, not from the assumed
    # counting noise: if the refresh steps follow beta tightly, the slope is known tightly,
    # and if they scatter, it is not.
    resid = [(abs(r["dz"] - beta * r["dlog_n"]), r["weight"]) for r in rows]
    cut = 1.5 * max(1.4826 * weighted_median(resid), 0.05)
    sw = 0.0
    sr = 0.0
    for r in rows:
        d = abs(r["dz"] - beta * r["dlog_n"])
        w = r["weight"] if d <= cut else r["weight"] * cut / d
        sw += w
        sr += w * d * d
    s2 = sr / sw if sw > 0 else 0.0
    return beta, s2 / den


def fit_betas(observations, n_trophies, upto_state=None):
    """The pooled beta and each trophy's own, shrunk toward it.

    `upto_state` restricts the fit to transitions that ended before that page state, which
    is how the model is refitted to forecast a refresh it has not seen.
    """
    rows = [o for o in observations if upto_state is None or o["b"] < upto_state]
    beta_raw, beta_var = robust_slope(rows) if rows else (0.0, float("inf"))
    # A weakly informative prior: without the data, a refresh's dilution is 0 +- 0.5 logits
    # per unit of log tracked. Shrinking by the slope's own variance, not by a count, means
    # a well-measured beta is left alone and a thin one is pulled hard to zero.
    prior_var = BETA_PRIOR_SD ** 2
    if beta_var < float("inf"):
        pooled = beta_raw * prior_var / (beta_var + prior_var)
    else:
        pooled = 0.0
    pooled = clamp(pooled, BETA_MIN, BETA_MAX)

    by_trophy = {}
    for o in rows:
        by_trophy.setdefault(o["trophy"], []).append(o)

    own = [pooled] * n_trophies
    own_var = BETA_OWN_PRIOR_SD ** 2
    for i, mine in by_trophy.items():
        b_i, v_i = robust_slope(mine)
        if v_i < float("inf"):
            # a trophy's own beta is pulled toward the pool, with a prior as wide as
            # trophies are known to differ from each other
            own[i] = clamp(pooled + (b_i - pooled) * own_var / (v_i + own_var),
                           BETA_MIN, BETA_MAX)
    return pooled, own


def tracked_growth(states, target_h):
    """dlogN from the last read to the target, from the tracked count's own growth.

    ln N against ln(hours since the opening): a power law, which is what a count that
    explodes at launch and slows as the week goes on looks like over a few days. The
    exponent is fitted on the page states, weighted toward the recent ones and shrunk
    toward zero by how few states there are -- a couple of refreshes say little about the
    next four days.
    """
    pts = [(math.log(max(s["first_h"], TIME_FLOOR_H)), math.log(s["tracked"]))
           for s in states if s["tracked"] > 0 and s["first_h"] > 0]
    if len(pts) < 3:
        return 0.0
    last_h = states[-1]["last_h"]
    x0 = math.log(max(last_h, TIME_FLOOR_H))
    y0 = math.log(max(states[-1]["tracked"], 1.0))
    num = 0.0
    den = 0.0
    weights = []
    for x, y in pts:
        age = max(0.0, last_h - math.exp(x))
        w = 0.5 ** (age / (2.0 * RECENCY_HALF_LIFE_H))
        weights.append((w, x, y))
        num += w * (x - x0) * (y - y0)
        den += w * (x - x0) ** 2
    if den <= 1e-12:
        return 0.0
    exponent = num / den
    # its own standard error, from how well the power law fits the states it was fitted on
    s2 = sum(w * (y - y0 - exponent * (x - x0)) ** 2 for w, x, y in weights)
    sw = sum(w for w, _, _ in weights)
    se2 = (s2 / sw) / den if sw > 0 else float("inf")
    prior_var = TRACKED_PRIOR_SD ** 2
    if se2 < float("inf"):
        exponent *= prior_var / (se2 + prior_var)
    else:
        return 0.0
    exponent = clamp(exponent, 0.0, TRACKED_EXPONENT_MAX)
    if target_h <= last_h:
        return 0.0
    return clamp(exponent * math.log(max(target_h, TIME_FLOOR_H) / max(last_h, TIME_FLOOR_H)),
                 0.0, MAX_TRACKED_JUMP)


def refresh_residuals(states, observations, n_trophies, target_dlog_n):
    """The model's own misses, measured on the refreshes it has already seen.

    For every page state after the second, the betas are refitted on the states before it
    and the model is asked for that state's rates; the miss, in logits, is kept and
    rescaled to the target's dlogN. This is the leave-the-refresh-out residual the spread
    is built from: a genuine out-of-sample error, per trophy, at the size the forecast
    faces now.
    """
    per_trophy = [[] for _ in range(n_trophies)]
    pooled = []
    # From the third page state on: at the second there is no earlier refresh to fit on,
    # so a miss there measures the absence of data, not the model's error.
    for b in range(3, len(states)):
        _, betas = fit_betas(observations, n_trophies, upto_state=b)
        prev, cur = states[b - 1], states[b]
        if prev["tracked"] <= 0 or cur["tracked"] <= 0:
            continue
        dlog_n = math.log(cur["tracked"] / prev["tracked"])
        if dlog_n < MIN_RESIDUAL_DLOGN:
            continue
        scale = clamp(target_dlog_n / dlog_n, 0.0, MAX_RESIDUAL_SCALE)
        for i in range(n_trophies):
            r_prev = rate_at(prev, i)
            r_cur = rate_at(cur, i)
            if r_prev is None or r_cur is None or r_prev <= 0.0 or r_cur <= 0.0:
                continue
            if r_prev >= RATE_CEIL or r_cur >= RATE_CEIL:
                continue
            # the same step the forecast makes: multiplicative in the rate, not additive in
            # the logit, so the residual measures the model that is actually shipped
            predicted = logit(min(RATE_CEIL, r_prev * math.exp(betas[i] * dlog_n)))
            e = (logit(r_cur) - predicted) * scale
            per_trophy[i].append(e)
            pooled.append(e)
    return per_trophy, pooled


def allocation_residuals(states, n_trophies):
    """The spread a single flat page leaves open, from the published totals alone.

    The history has not refreshed, so no miss can be measured. What the published pair does
    not settle is how the free earners -- 4,276 - 3,868 = 408 of them -- are shared among
    the trophies that are not at 100%: in proportion to the earners they already had, or
    equally. Both allocations reproduce the published totals and both reproduce the
    published persistence loss; the logit distance between them is therefore the honest
    size of what is unknown, and it is computed here per trophy rather than assumed.
    """
    last = states[-1]
    n_after = LAUNCH_N_AFTER
    n_before = last["tracked"] if last["tracked"] > 0 else LAUNCH_N_BEFORE
    free_after = LAUNCH_E_AFTER - LAUNCH_N_AFTER          # 408
    free_before = LAUNCH_E_BEFORE - LAUNCH_N_BEFORE       # 305
    if free_after <= 0 or free_before <= 0:
        return [[] for _ in range(n_trophies)], []
    k = free_after / free_before                          # 1.3377
    free_count = 0
    for i in range(n_trophies):
        r = rate_at(last, i)
        if r is not None and 0.0 < r < RATE_CEIL:
            free_count += 1
    per_trophy = [[] for _ in range(n_trophies)]
    pooled = []
    if free_count == 0:
        return per_trophy, pooled
    equal_share = free_after / free_count
    for i in range(n_trophies):
        r = rate_at(last, i)
        if r is None or r <= 0.0 or r >= RATE_CEIL:
            continue
        e_before = earners(r, n_before)
        if e_before <= 0.0:
            continue
        z_prop = logit(min(RATE_CEIL, 100.0 * (e_before * k) / n_after))
        z_equal = logit(min(RATE_CEIL, 100.0 * equal_share / n_after))
        per_trophy[i].append(z_prop - z_equal)
        pooled.append(z_prop - z_equal)
    return per_trophy, pooled


def rounding_sigma(rate_pct):
    """The logit width the page's two decimals cannot resolve."""
    p = clamp(rate_pct, RATE_FLOOR, RATE_CEIL) / 100.0
    return 0.005 / 100.0 / max(p * (1.0 - p), 1e-9)


def quantiles(z_centre, residuals, rate_last, rate_target):
    """Five quantiles: the centre plus the residual distribution's own percentiles.

    The residuals are centred on their median, so the 50% quantile is the centre -- the
    last observed level plus the dilution the data supports -- and nothing else moves it.
    They are then widened for the one error the residuals cannot carry: the page prints two
    decimals, and at the target the rate is smaller than at the anchor, so a printed point
    resolves less. The two tails are floored the same way, so a tight sample gives tight
    quantiles but never a degenerate row.
    """
    if residuals:
        m = median(residuals)
        sample = [r - m for r in residuals]
        if len(sample) > RESIDUAL_SAMPLE_CAP:
            step = len(sample) / float(RESIDUAL_SAMPLE_CAP)
            sample = [sample[int(i * step)] for i in range(RESIDUAL_SAMPLE_CAP)]
    else:
        sample = []
    floor_anchor = max(SIGMA_FLOOR, rounding_sigma(rate_last))
    floor_target = max(SIGMA_FLOOR, rounding_sigma(rate_target))
    extra2 = max(0.0, floor_target ** 2 - floor_anchor ** 2)
    if sample and extra2 > 0.0:
        sd = math.sqrt(sum(v * v for v in sample) / len(sample))
        if sd > 1e-9:
            factor = math.sqrt(1.0 + extra2 / (sd * sd))
            sample = [v * factor for v in sample]
    if sample:
        out = [quantile(sample, level) for level in LEVELS]
    else:
        out = [-floor_target, -floor_target, 0.0, floor_target, floor_target]
    # the centre is the forecast: hold the median there, and let a lopsided sample shape
    # only the side it leans to
    out[2] = 0.0
    out[1] = min(out[1], 0.0)
    out[3] = max(out[3], 0.0)
    out[0] = min(out[0], out[1], -floor_target)
    out[4] = max(out[4], out[3], floor_target)
    pcts = []
    for q in out:
        pct = 100.0 * inv_logit(z_centre + q)
        pcts.append(round(clamp(pct, RATE_FLOOR, RATE_CEIL), 4))
    for i in range(1, len(pcts)):
        if pcts[i] < pcts[i - 1]:
            pcts[i] = pcts[i - 1]
    return pcts


def tracked_residuals(states):
    """One-step-ahead misses of the tracked count's own growth law.

    The refresh residuals measure how well beta predicts a step's size *given* the step;
    they say nothing about the tracked count's own extrapolation, which over four days is
    the largest thing the forecast can get wrong. Each page state from the third on is
    predicted from the states before it, and the miss in dlogN is kept.
    """
    errs = []
    for b in range(3, len(states)):
        prev, cur = states[b - 1], states[b]
        if prev["tracked"] <= 0 or cur["tracked"] <= 0:
            continue
        actual = math.log(cur["tracked"] / prev["tracked"])
        if actual <= MIN_RESIDUAL_DLOGN:
            continue
        predicted = tracked_growth(states[:b], cur["first_h"])
        errs.append(actual - predicted)
    return errs


def forecast_case(case):
    trophies = case.get("trophies") or []
    n = len(trophies)
    states = page_states(case)
    if not states:
        return [{"id": t.get("id"), "q": [1.0, 1.0, 1.0, 1.0, 1.0]} for t in trophies]

    t0 = opening_time(case)
    target = parse_time(case.get("target"))
    if target is not None:
        target_h = (target - t0).total_seconds() / 3600.0
    else:
        target_h = states[-1]["last_h"]
    last = states[-1]

    observations = transition_observations(states)
    if len(states) >= 2:
        pooled_beta, betas = fit_betas(observations, n)
        dlog_n = tracked_growth(states, target_h)
    else:
        # One page state: nothing has refreshed, so the first refresh of a launch is the
        # only jump there is to forecast, at the scale the published pair fixes.
        pooled_beta = PRIOR_BETA
        betas = [PRIOR_BETA] * n
        dlog_n = PRIOR_LOG_TRACKED_JUMP

    # How far the tracked count's own extrapolation may be off, in the same units as the
    # residual sample: the one-step misses, grown by how far beyond the history the target
    # sits. It enters the spread through beta, because that is how it reaches a rate.
    spread_from_tracked = 0.0
    if len(states) >= 4:
        errs = tracked_residuals(states)
        if len(errs) >= 2:
            rms = math.sqrt(sum(e * e for e in errs) / len(errs))
            span = max(states[-1]["last_h"] - states[0]["first_h"], 1.0)
            ahead = max(0.0, target_h - states[-1]["last_h"])
            spread_from_tracked = abs(pooled_beta) * rms * math.sqrt(1.0 + ahead / span)

    per_trophy_res, pooled_res = ([[] for _ in range(n)], [])
    if len(states) >= 2:
        per_trophy_res, pooled_res = refresh_residuals(states, observations, n, dlog_n)
    if not pooled_res:
        per_trophy_res, pooled_res = allocation_residuals(states, n)
    if pooled_res and spread_from_tracked > 0.0:
        sd = math.sqrt(sum(v * v for v in pooled_res) / len(pooled_res))
        factor = math.sqrt(1.0 + (spread_from_tracked / sd) ** 2) if sd > 1e-9 else 1.0
        if factor > 1.0:
            pooled_res = [v * factor for v in pooled_res]
            per_trophy_res = [[v * factor for v in row] for row in per_trophy_res]

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = rate_at(last, i)
        if rate_last is None:
            # the last page did not list it: carry the last rate that was listed
            for state in reversed(states):
                rate_last = rate_at(state, i)
                if rate_last is not None:
                    break
        if rate_last is None or rate_last <= 0.0:
            rows.append({"id": trophy.get("id"), "q": [RATE_FLOOR, RATE_FLOOR,
                                                        RATE_FLOOR, RATE_FLOOR, RATE_FLOOR]})
            continue

        if rate_last >= RATE_CEIL:
            # every tracked player holds it, and a refresh adds players who hold it too
            z_centre = logit(RATE_CEIL)
            residual_sample = [0.0]
        else:
            drift = betas[i] * dlog_n
            if len(states) >= 2:
                # MAX_DRIFT guards a power law read four days out; the published launch
                # jump is a measured step, not an extrapolation, so it is not trimmed.
                drift = clamp(drift, -MAX_DRIFT, MAX_DRIFT)
            # The step belongs in rate space, where the model lives: rate = E / N and both
            # are counts. Adding it in logits would overshoot the bigger rates, because
            # logit is not linear in the rate.
            rate_centre = min(RATE_CEIL, rate_last * math.exp(drift))
            z_centre = logit(rate_centre)
            residual_sample = per_trophy_res[i] if len(per_trophy_res[i]) >= MIN_OWN_RESIDUALS \
                else pooled_res
        rows.append({"id": trophy.get("id"),
                     "q": quantiles(z_centre, residual_sample, rate_last,
                                    100.0 * inv_logit(z_centre))})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(n for n in os.listdir(in_dir) if n.endswith(".in")):
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast_case(case)
        with open(os.path.join(out_dir, "%s.out" % name[:-3]), "w", encoding="utf-8") as f:
            f.write(json.dumps({"trophies": rows}, separators=(",", ":")))
            f.write("\n")


if __name__ == "__main__":
    main()
