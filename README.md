[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Amburadullzz/makefun-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Amburadullzz/makefun-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy earn-rate forecast

An entry for `gta6-trophy-rarity-forecast-01`: for every PlayStation 5 trophy of Grand
Theft Auto VI, five quantiles of its Exophase earn rate at the case's target.

`main.py` is the whole entry. It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else
`/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
Python 3, standard library only, no network, no randomness.

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

## The model

A trophy's rate is not a level that drifts. It is a ratio the page recomputes at every
refresh, `rate = 100 * E / N`, with `E` the tracked players holding the trophy and `N` the
players tracked. Exophase holds the page still for hours and then replaces it whole, so
between refreshes nothing moves, and at a refresh both counts jump -- the denominator much
harder, because a refresh is when players who have barely started get counted. That is the
whole forecast:

    z = logit(rate)   ->   dz = (eps - 1) * dlogN,   eps = dlogit(E) / dlogN

1. **Page states.** Reads are collapsed to the distinct pages behind them. A run of
   identical reads is one observation, not many; only a refresh is news.
2. **Beta.** Each refresh gives one observation per trophy of `dz / dlogN`; they are pooled
   by weighted least squares through the origin with a Huber reweight, weighted by the
   counting precision of the two earn counts and the page's two-decimal print, and shrunk
   toward zero by the slope's own variance. Each trophy's own beta is pulled toward the
   pool. A rate printed at 0.01% cannot show a move and is dropped as censored.
3. **Tracked growth.** `ln N` against `ln(hours since the opening)` -- a power law, which is
   what a count that explodes at launch and slows over the week looks like -- fitted on the
   page states, weighted to the recent ones, shrunk toward no growth by its own standard
   error and capped. Read at the target, that is `dlogN`; the drift follows from it and is
   applied in rate space, where the model lives.
4. **One page state.** If nothing has refreshed, no growth can be fitted from a flat line,
   so the forecast takes the one jump the kit documents in full: the first refresh of a
   launch, at the scale the published pair fixes (638 tracked / 943 earned -> 3,868 /
   4,276). It gives `eps = 0.1614`, `beta = -0.8386`, and `beta * ln(3868/638) = -1.5112`,
   which is exactly the loss the kit publishes for the persistence baseline (1.51). A
   constant logit shift is scored at `|shift|` when the quantiles are equal, so that
   identity is the check that the decomposition is the right one.
5. **Spread, measured.** The quantiles are the centre plus the residual distribution's own
   5th, 25th, 50th, 75th and 95th percentiles. With refreshes in the history, the residual
   is the model's own miss when it was refitted on the states before each refresh and asked
   for that refresh, rescaled to the target's `dlogN`, plus the tracked count's one-step
   extrapolation error propagated through beta. With a single flat page there is no earlier
   refresh to be asked about -- every read is byte-identical, so a leave-the-last-read-out
   residual is exactly zero -- and the spread comes from what the published totals leave
   open: how the refresh's 408 free earners are shared among the 54 trophies not at 100%.
   Proportional and equal allocation bracket it; the gap between them is the band.

Quantiles never cross, stay in (0, 100), and the 50% quantile is always the centre.

## The tools

`tools/` holds the evidence, not the entry:

    python3 tools/evaluate.py            # loss table, calibration, contract, determinism, timing
    python3 tools/test_multi_refresh.py  # the multi-refresh path, on a case built here
    python3 tools/stress_timing.py       # a case the size the final will be, timed

`tools/practice_truth.py` rebuilds the practice answers from the three numbers the kit
publishes, and checks the rebuild against the two baseline results it also publishes
(persistence 1.51, typical miss 0.68): the rebuild lands on 1.494 and 0.678. The kit's
sealed answer file is not in the bundle, so every loss in the tables is against that
rebuild and is labelled so.
