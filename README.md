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

1. Each trophy's history is reduced to the steps where its rate actually changed -- one
   point per Exophase refresh -- each weighted by its earn count (rate x players tracked),
   which is how precisely that rate is known.
2. In logit space against log time since the game's opening, each trophy gets a weighted,
   robust (Huber) quadratic: the local slope and the local bend, which is what decides
   where the curve goes next.
3. Slopes and curvatures are shared: precision-weighted pooled values, with each trophy's
   own shrunk toward them by how much its data supports (empirical Bayes).
4. The line is anchored at the last observed level, read at the target's log time, its
   drift trimmed to 85% (a local fit over-extrapolates a curve that must flatten) and
   capped at two logits.
5. The spread is normal in logit space: a fixed number of logits per e-fold of time
   extrapolated, plus a quarter of the measured refresh-to-refresh jitter, plus the
   rate's counting noise (1/sqrt(earn count)), plus a floor for model miss.
