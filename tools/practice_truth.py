#!/usr/bin/env python3
"""A reconstruction of the two practice cases' answers, for local scoring.

The real answer file (reference/answer.json) is sealed on make.fun's side and is not in
the kit, so a local loss table needs an answer to score against. This module rebuilds it
from the three numbers the kit publishes, and then checks the rebuild against the two
baseline results the kit also publishes. It is a reconstruction, not the answer file.

What the kit says (competition.yaml, "The practice"):
  - before the refresh: 638 players tracked, 943 trophies earned (the last read of p00 and
    p01, and the sum of their rates: sum(rate) = 147.78% = 943/638);
  - at the refresh (the 04:30Z read, both cases' answer): 3,868 players tracked, 4,276
    trophies earned, "every rate but the one at 100% fell";
  - so the 100% trophy holds 3,868 of the 4,276 earners, and the other 54 trophies share
    4,276 - 3,868 = 408, where they shared 943 - 638 = 305 before.

If those 408 are spread over the 54 trophies in proportion to the earners they had, every
one of them scales by k = 408/305, and every rate by k * 638/3868 = 0.220644, printed to
two decimals. Two published aggregates then test the rebuild without the answer file:

  persistence loss  1.51   <- |ln 0.220644| = 1.511   (a constant logit shift is scored at
                              exactly |ln f| when the five quantiles are equal)
  typical miss      0.68   <- mean over the 55 trophies of rate * (1 - f)
                              = (147.78 - 100) * 0.779356 / 55 = 0.677

Both land on the published figures, which is what makes the rebuild usable as a proxy for
the practice answer. It cannot, however, tell two forecasts apart inside a trophy: it fixes
the fall at one global factor, so it scores the LEVEL of a forecast, not its per-trophy
shape. Losses measured against it are therefore a check on the level, and are labelled so.
"""

TRACKED_BEFORE = 638.0
EARNED_BEFORE = 943.0
TRACKED_AFTER = 3868.0
EARNED_AFTER = 4276.0


def scale_factor(rates):
    """The factor every non-100% rate is multiplied by at the practice's refresh."""
    n100 = sum(1 for r in rates if r >= 100.0)
    earned_100_before = TRACKED_BEFORE * n100          # a 100% trophy is every tracked player
    earned_100_after = TRACKED_AFTER * n100
    rest_before = EARNED_BEFORE - earned_100_before
    rest_after = EARNED_AFTER - earned_100_after
    if rest_before <= 0 or rest_after <= 0:
        return TRACKED_BEFORE / TRACKED_AFTER
    return (rest_after / rest_before) * (TRACKED_BEFORE / TRACKED_AFTER)


def truth_for(case, allocation="proportional"):
    """The practice answer, per trophy, as the page would print it (two decimals).

    `allocation` is the one thing the published totals leave open: how the 408 free
    earners are shared among the 54 trophies not at 100%. "proportional" scales the
    earners each already held; "equal" splits them evenly. Both reproduce 4,276 earned and
    both land near the published persistence loss; the two together bracket the answer.
    """
    rates = case["history"][-1]["rates"]
    n_after = TRACKED_AFTER
    n100 = sum(1 for r in rates if r is not None and r >= 100.0)
    free_after = EARNED_AFTER - n_after * n100
    free_count = sum(1 for r in rates if r is not None and 0.0 < r < 100.0)
    out = []
    if allocation == "equal":
        share = free_after / free_count if free_count else 0.0
        for r in rates:
            if r is None:
                out.append(None)
            elif r >= 100.0:
                out.append(100.0)
            else:
                out.append(round(min(99.99, max(0.01, 100.0 * share / n_after)), 2))
        return out

    f = scale_factor(rates)
    for r in rates:
        if r is None:
            out.append(None)
        elif r >= 100.0:
            out.append(100.0)
        else:
            out.append(round(min(99.99, max(0.01, r * f)), 2))
    return out


if __name__ == "__main__":
    import json
    import math
    import sys

    for path in sys.argv[1:]:
        case = json.load(open(path))
        rates = case["history"][-1]["rates"]
        f = scale_factor(rates)
        truth = truth_for(case)

        def logit(x):
            p = min(99.99, max(0.01, x)) / 100.0
            return math.log(p / (1.0 - p))

        # persistence: all five quantiles at the last read
        items = []
        miss = []
        for r, y in zip(rates, truth):
            d = logit(y) - logit(r)
            items.append(abs(d))
            miss.append(abs(y - r))
        # typical miss excludes nothing; the 100% trophy simply contributes 0
        print(case["id"], "factor %.6f" % f,
              "persistence loss %.4f" % (sum(items) / len(items)),
              "typical miss %.4f pp" % (sum(miss) / len(miss)),
              "| kit publishes 1.51 and 0.68")
