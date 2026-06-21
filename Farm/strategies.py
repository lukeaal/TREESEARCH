"""Reusable baseline strategies and a small policy runner.

A *policy* here is just a callable ``policy(farm) -> None`` that inspects the
farm and issues actions; the runner calls it once per period and then advances
the calendar. These baselines double as (a) sanity checks, (b) the demo's
"sensible operator", and (c) the reference profit the RL environment normalises
rewards against.
"""

from __future__ import annotations

from collections.abc import Callable

from .objects import Farm
from .spec import FarmSpec

Policy = Callable[[Farm], None]


def do_nothing(farm: Farm) -> None:
    """Plant it and walk away — never tap, never spend. Profit stays at 0."""
    return None


def tap_only(farm: Farm) -> None:
    """Tap every mature tree; never irrigate or fertilize. Free revenue."""
    farm.tap(on=True, targets="mature")


def tend_and_tap(farm: Farm) -> None:
    """A sensible operator: tap mature trees, top up water/nutrients when low."""
    a = farm.observe()["averages"]
    farm.tap(on=True, targets="mature")
    if a["moisture"] < 0.45:
        farm.water(gallons_per_tree=5, targets="all")
    if a["nutrients"] < 0.35:
        farm.fertilize(nitrogen=0.2, phosphorus=0.15, potassium=0.15, targets="all")


def worst_case(farm: Farm) -> None:
    """The worst possible operator: spend maximally on inputs and never harvest.

    Maximum cost, zero revenue -> the most negative profit achievable. This is
    the reward's 0.0 anchor, so any agent that actually taps (earns revenue) or
    simply spends less ends up strictly above it.
    """
    farm.tap(on=False, targets="all")
    farm.water(gallons_per_tree=25, targets="all")
    farm.fertilize(nitrogen=0.5, phosphorus=0.5, potassium=0.5, targets="all")


def run_policy(spec: FarmSpec, policy: Policy, period_days: int = 365) -> Farm:
    """Run ``policy`` to the end of the scenario and return the finished farm.

    The policy is applied once every ``period_days`` (default: once a year),
    then the calendar advances by that many days.
    """
    farm = Farm.from_spec(spec)
    while not farm.finished:
        policy(farm)
        farm.step(days=period_days)
    return farm


def reference_profit(spec: FarmSpec, policy: Policy = tend_and_tap) -> float:
    """Profit a baseline policy earns on ``spec`` — used to scale RL rewards."""
    return run_policy(spec, policy).profit


def best_baseline_profit(spec: FarmSpec) -> float:
    """Profit of the strongest simple baseline (the reward's 1.0 anchor).

    On these scenarios "just tap everything" (``tap_only``) usually beats the
    fertilizing operator, so we take the max to be safe.
    """
    return max(reference_profit(spec, tap_only), reference_profit(spec, tend_and_tap))


# Reward earned at the worst-case anchor. Kept above 0 on purpose: a forgiving
# floor means even poor play gets a little signal (and a gradient just below the
# floor), instead of a flat-zero desert that gives RL nothing to climb.
SOFT_FLOOR_REWARD = 0.1


def reward_anchors(spec: FarmSpec) -> tuple[float, float, float]:
    """``(floor, human, ceil)`` profit anchors for the env reward.

    - ``floor`` = the worst-case operator (max spend, no harvest) -> reward 0.1.
    - ``human`` = a sensible human-like operator (``tend_and_tap``) -> reward 0.5.
    - ``ceil``  = the best simple baseline (tap everything) -> reward 1.0.

    See :func:`scale_reward` for how a profit maps through these. The middle
    anchor lets us pin "what a human would do" at 0.5, with room to score lower
    (poor play, but still positive) or higher (beating the human baseline).
    """
    floor = reference_profit(spec, worst_case)
    human = reference_profit(spec, tend_and_tap)
    ceil = max(best_baseline_profit(spec), human)
    return floor, human, ceil


def scale_reward(profit: float, anchors: tuple[float, float, float]) -> float:
    """Map ``profit`` to ``0..1`` through ``(floor, human, ceil)`` piecewise.

    - ``human..ceil`` -> ``0.5..1.0`` (beating the human baseline climbs to 1.0).
    - ``floor..human`` -> ``SOFT_FLOOR_REWARD..0.5`` (poor-but-trying play still
      earns a meaningful positive reward; do-nothing lands comfortably above 0).
    - below ``floor`` -> a gentle tail from ``SOFT_FLOOR_REWARD`` down to 0 over
      another floor-span, so catastrophic spending is penalised *smoothly*
      instead of slamming to a hard, gradient-free zero.

    Clamped to ``[0, 1]``.
    """
    floor, human, ceil = anchors
    if profit >= human:
        denom = ceil - human
        reward = 0.5 + 0.5 * (profit - human) / denom if denom > 0 else 1.0
    elif profit >= floor:
        denom = human - floor
        frac = (profit - floor) / denom if denom > 0 else 0.0
        reward = SOFT_FLOOR_REWARD + (0.5 - SOFT_FLOOR_REWARD) * frac
    else:
        # Below the worst-case anchor: decay to 0 over a span the size of |floor|.
        span = abs(floor) if floor else 1.0
        frac = max(0.0, 1.0 + (profit - floor) / span)
        reward = SOFT_FLOOR_REWARD * frac
    return max(0.0, min(1.0, reward))
