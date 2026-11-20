#!/usr/bin/env python3
"""Forecast every PS5 trophy's Exophase earn rate for Grand Theft Auto VI at a case's target.

Reads every *.in case in the input directory and writes one <id>.out per case, one row per
trophy, five quantiles of that trophy's earn rate at the case's target.

How the forecast is made:

  1. Series. For each trophy, the steps at which its earn rate actually changed -- one
     point per Exophase refresh, not the run of identical reads that follows it. The last
     step is dated by the case's last read, because the page is known to have held that
     level until then. Each point carries its earn count (rate x players tracked / 100),
     which is how precisely the rate is known: a rate printed from seven earners is far
     noisier than one printed from seven hundred, and the weight of a point follows it.

  2. Transform. y = logit(rate) against u = ln(hours since the game's opening). A curve
     that climbs fast and then flattens is close to a straight line in that plane over a
     few days, and its bend is the part that matters four days later, so each trophy gets
     a quadratic in u -- weighted, robust (Huber), so one stale or re-based page cannot
     drag it.

  3. Pooling. Slopes and curvatures are shared. The pooled values are precision-weighted
     means over the trophies, and each trophy's own are shrunk toward them by how much its
     data can support (empirical Bayes, with the between-trophy spread estimated from the
     trophies themselves). A trophy with fewer than four steps takes the pooled line.

  4. Prediction. The line is anchored at the last observed level and read at the target's
     log time. The drift is trimmed: a local log-log fit extrapolated four days out
     overshoots a curve that has to flatten, so the forecast takes 85% of it, and never
     more than two logits.

  5. Spread. The five quantiles are normal in logit space around that median. The spread
     is the model's own uncertainty about the extrapolation (a fixed number of logits per
     e-fold of time extrapolated, so it grows with the distance to the target), plus the
     page's own refresh-to-refresh jitter, plus the rate's counting noise
     (1/sqrt(earn count), which matters for a trophy a few dozen players have earned),
     plus a floor for model miss.

Deterministic: no randomness, no clock, no network, standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# The standard normal's quantile at each level.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)

RATE_FLOOR = 0.01           # the scorer clamps a rate here before its logit
RATE_CEIL = 99.99
TIME_FLOOR_H = 1.0          # ln(hours since the opening) is undefined at the opening
RECENCY_HALF_LIFE_H = 48.0  # a step two days old weighs half what the last one does
COUNT_WEIGHT_CAP = 4000.0   # a very common trophy is not more precise than this
SLOPE_MIN, SLOPE_MAX = -0.30, 2.50
CURVE_MIN, CURVE_MAX = -0.80, 0.10
PRIOR_SLOPE = 0.60          # logit per ln-hour, when no trophy has a slope of its own
PRIOR_SLOPE_SD = 0.40       # between-trophy slope spread, before the trophies say otherwise
PRIOR_CURVE = -0.20         # between-trophy curvature, before the trophies say otherwise
PRIOR_CURVE_SD = 0.25
TAU2_MIN, TAU2_MAX = 0.03, 0.50
DRIFT_TRIM = 0.85           # a local log-log fit over-extrapolates a curve that must flatten
MAX_DRIFT = 2.00            # logit: the most the trend may move a trophy's median
TREND_PER_EFOLD = 0.28      # logit of uncertainty per unit of ln-time extrapolated
STEP_JITTER_MAX = 0.25      # logit: a bigger refresh-to-refresh miss is model misfit
STEP_JITTER_WEIGHT = 0.25   # how much of that jitter carries into the target's spread
SIGMA_FLOOR = 0.15          # logit: model miss the spread above cannot see
SIGMA_MIN, SIGMA_MAX = 0.10, 2.50
MIN_QUAD_POINTS = 4         # steps needed before a trophy has a curvature of its own
MIN_QUAD_SPAN = 0.45        # ln-hours its steps must span
SE_TWO_POINTS = 0.70        # a two-step slope is worth little: shrink it hard


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


def weighted_normal(xs, ys, ws, power):
    """Weighted least squares for a polynomial of degree `power`.

    Returns (coefficients, inverse of the normal matrix) or (None, None) if singular.
    """
    size = power + 1
    ata = [[0.0] * size for _ in range(size)]
    atb = [0.0] * size
    for x, y, w in zip(xs, ys, ws):
        xp = 1.0
        row = []
        for i in range(size):
            row.append(xp)
            xp *= x
        for i in range(size):
            atb[i] += w * row[i] * y
            for j in range(size):
                ata[i][j] += w * row[i] * row[j]
    aug = [ata[i][:] + [atb[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None, None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(size):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    coef = [aug[i][size] for i in range(size)]
    # inverse of A^T A: Gauss-Jordan on the normal matrix
    inv = [ata[i][:] + [1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(inv[r][col]))
        if abs(inv[pivot][col]) < 1e-12:
            return coef, None
        inv[col], inv[pivot] = inv[pivot], inv[col]
        pv = inv[col][col]
        inv[col] = [v / pv for v in inv[col]]
        for r in range(size):
            if r != col and inv[r][col] != 0.0:
                f = inv[r][col]
                inv[r] = [a - f * b for a, b in zip(inv[r], inv[col])]
    inv = [row[size:] for row in inv]
    return coef, inv


def robust_poly(xs, ys, ws, power):
    """Huber-weighted polynomial. Returns (coefficients, covariance matrix or None).

    A step whose level is off the rest -- launch night's stale page, a refresh that
    re-based everything at once -- is down-weighted instead of dragging the fit.
    """
    w = list(ws)
    coef, inv = weighted_normal(xs, ys, w, power)
    if coef is None:
        return None, None
    n = len(xs)
    for _ in range(16):
        res = [y - sum(c * x ** i for i, c in enumerate(coef)) for x, y in zip(xs, ys)]
        scale = 1.4826 * median([abs(r) for r in res])
        if scale < 0.12:
            scale = 0.12
        c = 1.5 * scale
        w = [(wi if abs(r) <= c else wi * c / abs(r)) for wi, r in zip(ws, res)]
        coef2, inv2 = weighted_normal(xs, ys, w, power)
        if coef2 is None:
            break
        done = max(abs(a - b) for a, b in zip(coef2, coef)) < 1e-10
        coef, inv = coef2, inv2
        if done:
            break
    sw = sum(w)
    if inv is not None and sw > 0.0 and n > power + 1:
        s2 = sum(wi * (y - sum(c * x ** i for i, c in enumerate(coef))) ** 2
                 for wi, x, y in zip(w, xs, ys)) / sw
        s2 = max(s2, 0.01)
        inv = [[v * s2 for v in row] for row in inv]
    else:
        inv = None
    return coef, inv


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


def build_series(case):
    """Per trophy: the steps at which its rate changed, as [hours, rate, tracked]."""
    t0 = opening_time(case)
    reads = []
    for read in case.get("history") or []:
        t = parse_time(read.get("slot") or read.get("at"))
        if t is None:
            continue
        try:
            tracked = float(read.get("players_tracked"))
        except (TypeError, ValueError):
            tracked = 0.0
        reads.append(((t - t0).total_seconds() / 3600.0, tracked, read.get("rates") or []))
    reads.sort(key=lambda r: r[0])

    per_trophy = []
    for i in range(len(case.get("trophies") or [])):
        steps = []
        for hours, tracked, rates in reads:
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            rate = float(rate)
            if steps and steps[-1][1] == rate:
                steps[-1][0] = hours        # the level held to this read
            else:
                steps.append([hours, rate, tracked])
        if steps:
            steps[-1][0] = reads[-1][0]
        per_trophy.append(steps)
    return per_trophy, reads


def point_weight(hours, rate, tracked, last_hours):
    """A step's weight: its earn count (Poisson precision in logit space), aged."""
    count = rate * tracked / 100.0 if tracked > 0 else 1.0
    count = clamp(count, 1.0, COUNT_WEIGHT_CAP)
    return count * 0.5 ** (max(0.0, last_hours - hours) / RECENCY_HALF_LIFE_H)


def fit_trophy(steps, last_hours):
    """A trophy's own (slope, curvature, covariance), or None if it has too few steps.

    Time is centred on the last step, so the slope is the rate's local slope at the last
    read and the curvature its local bend -- the two numbers the extrapolation needs, and
    the two that can sensibly be shared with other trophies.
    """
    if len(steps) < 2:
        return None
    u_last = math.log(max(steps[-1][0], TIME_FLOOR_H))
    xs = [math.log(max(h, TIME_FLOOR_H)) - u_last for h, _, _ in steps]
    ys = [logit(r) for _, r, _ in steps]
    ws = [point_weight(h, r, n, last_hours) for h, r, n in steps]
    span = max(xs) - min(xs)
    if len(steps) >= MIN_QUAD_POINTS and span >= MIN_QUAD_SPAN:
        coef, cov = robust_poly(xs, ys, ws, 2)
        if coef is None:
            return None
        return {"b": coef[1], "c": coef[2], "cov": cov, "quad": True}
    if span <= 1e-6:
        return None
    coef, cov = robust_poly(xs, ys, ws, 1)
    if coef is None:
        return None
    return {"b": coef[1], "c": 0.0, "cov": cov, "quad": False}


def pooled_params(fits):
    """The shared slope and curvature, and how much trophies differ around them."""
    lin = [(f["b"], f["cov"][1][1] if f["cov"] else SE_TWO_POINTS ** 2) for f in fits
           if f is not None]
    quad = [(f["c"], f["cov"][2][2] if f["cov"] else PRIOR_CURVE_SD ** 2) for f in fits
            if f is not None and f["quad"]]

    b_pool, tau2 = PRIOR_SLOPE, PRIOR_SLOPE_SD ** 2
    if lin:
        for _ in range(8):
            ws = [1.0 / (se2 + tau2) for _, se2 in lin]
            sw = sum(ws)
            b_pool = clamp(sum(w * b for w, (b, _) in zip(ws, lin)) / sw, SLOPE_MIN, SLOPE_MAX)
            var_obs = sum(w * (b - b_pool) ** 2 for w, (b, _) in zip(ws, lin)) / sw
            mean_se2 = sum(se2 for _, se2 in lin) / len(lin)
            tau2 = clamp(max(var_obs - mean_se2, TAU2_MIN), TAU2_MIN, TAU2_MAX)

    c_pool, tau_c2 = PRIOR_CURVE, PRIOR_CURVE_SD ** 2
    if len(quad) >= 5:
        for _ in range(8):
            ws = [1.0 / (se2 + tau_c2) for _, se2 in quad]
            sw = sum(ws)
            c_pool = clamp(sum(w * c for w, (c, _) in zip(ws, quad)) / sw, CURVE_MIN, CURVE_MAX)
            var_obs = sum(w * (c - c_pool) ** 2 for w, (c, _) in zip(ws, quad)) / sw
            mean_se2 = sum(se2 for _, se2 in quad) / len(quad)
            tau_c2 = clamp(max(var_obs - mean_se2, 0.01), 0.01, 0.36)
    return b_pool, tau2, c_pool, tau_c2


def shrunk(fit, b_pool, tau2, c_pool, tau_c2):
    """The trophy's own slope and curvature, pulled toward the pooled ones."""
    if fit is None:
        return b_pool, c_pool
    cov = fit["cov"]
    if cov is None:
        return b_pool, c_pool
    vb = max(cov[1][1], 1e-6)
    b = (fit["b"] / vb + b_pool / tau2) / (1.0 / vb + 1.0 / tau2)
    if fit["quad"] and cov[2][2] > 1e-9:
        vc = cov[2][2]
        c = (fit["c"] / vc + c_pool / tau_c2) / (1.0 / vc + 1.0 / tau_c2)
    else:
        c = 0.0
    return clamp(b, SLOPE_MIN, SLOPE_MAX), clamp(c, CURVE_MIN, CURVE_MAX)


def step_noise(series, params, reads, b_pool, tau2, c_pool, tau_c2):
    """How far a refresh-to-refresh move lands from the fitted line, in logits.

    Measured honestly: each trophy is refitted on every step but its last, and that line
    is asked for the last step. This is the page's own jitter from refresh to refresh -- a
    level error, not a drift error, so it is added to the spread as it stands rather than
    grown with the horizon.
    """
    misses = []
    for steps, (b, c) in zip(series, params):
        if len(steps) < 2:
            continue
        (h_prev, r_prev, n_prev), (h_last, r_last, _) = steps[-2], steps[-1]
        if n_prev <= 0 or r_prev * n_prev / 100.0 < 30.0:
            continue
        du = math.log(max(h_last, TIME_FLOOR_H)) - math.log(max(h_prev, TIME_FLOOR_H))
        if du <= 1e-6:
            continue
        if len(steps) >= 3:
            fit = fit_trophy(steps[:-1], h_prev)
            b_use, c_use = shrunk(fit, b_pool, tau2, c_pool, tau_c2)
        else:
            b_use, c_use = b, c
        predicted = logit(r_prev) + b_use * du + c_use * du * du
        misses.append(abs(logit(r_last) - predicted))
    if len(misses) < 5:
        return 0.0
    return clamp(math.sqrt(sum(m * m for m in misses) / len(misses)), 0.0, STEP_JITTER_MAX)


def forecast_trophy(steps, b, c, target_hours, step_jitter):
    """One trophy's median logit at the target and the spread around it."""
    if not steps:
        return logit(1.0), 1.30
    h_last, r_last, n_last = steps[-1]
    y_last = logit(r_last)

    if r_last <= 0.02:                      # nothing to extrapolate from
        return y_last, 1.20

    du = math.log(max(target_hours, TIME_FLOOR_H)) - math.log(max(h_last, TIME_FLOOR_H))
    drift = DRIFT_TRIM * (b * du + c * du * du)
    # A rate at the top of the page cannot go above 99.99, and its logit has no room to
    # rise, so the median is held there while the spread still allows it to slip.
    y_hat = min(logit(RATE_CEIL), y_last + clamp(drift, -MAX_DRIFT, MAX_DRIFT))

    count = r_last * n_last / 100.0 if n_last > 0 else 1.0
    sigma_count = min(1.20, 1.0 / math.sqrt(max(count, 1.0)))
    # The extrapolation's own uncertainty, plus the page's refresh-to-refresh jitter.
    sigma = math.sqrt((TREND_PER_EFOLD * du) ** 2 + (STEP_JITTER_WEIGHT * step_jitter) ** 2
                      + sigma_count ** 2 + SIGMA_FLOOR ** 2)
    return y_hat, clamp(sigma, SIGMA_MIN, SIGMA_MAX)


def quantiles(y_hat, sigma):
    out = []
    for z in Z:
        pct = 100.0 * inv_logit(y_hat + z * sigma)
        out.append(round(clamp(pct, RATE_FLOOR, RATE_CEIL), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    series, reads = build_series(case)
    t0 = opening_time(case)
    target = parse_time(case.get("target"))
    if target is not None:
        target_hours = (target - t0).total_seconds() / 3600.0
    else:
        target_hours = reads[-1][0] if reads else 1.0

    last_hours = reads[-1][0] if reads else 0.0
    fits = [fit_trophy(steps, last_hours) for steps in series]
    b_pool, tau2, c_pool, tau_c2 = pooled_params(fits)
    params = [shrunk(f, b_pool, tau2, c_pool, tau_c2) for f in fits]
    step_jitter = step_noise(series, params, reads, b_pool, tau2, c_pool, tau_c2)

    rows = []
    for trophy, steps, (b, c) in zip(case.get("trophies") or [], series, params):
        y_hat, sigma = forecast_trophy(steps, b, c, target_hours, step_jitter)
        rows.append({"id": trophy.get("id"), "q": quantiles(y_hat, sigma)})
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
