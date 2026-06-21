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


# Smallest positive reward for a plan that harvested *some* rubber but lost money
# heavily. Any harvest at all clears 0 (you did something productive); the score
# then climbs as losses shrink and, past break-even, as profit grows.
MIN_HARVEST_REWARD = 0.1


def reward_anchors(spec: FarmSpec) -> tuple[float, float]:
    """``(loss_floor, ceil)`` profit anchors for the env reward.

    - ``loss_floor`` = the worst-case operator's (negative) profit: the bottom of
      the "harvested but losing money" band.
    - ``ceil`` = the best simple baseline (tap everything): the profit that earns
      a full 1.0.

    Break-even (profit 0) always maps to 0.5. See :func:`scale_reward`.
    """
    loss_floor = reference_profit(spec, worst_case)
    ceil = max(best_baseline_profit(spec), reference_profit(spec, tend_and_tap))
    return loss_floor, ceil


def scale_reward(
    profit: float, latex_lb: float, anchors: tuple[float, float]
) -> float:
    """Map ``(profit, rubber harvested)`` to ``0..1``.

    - harvested NO rubber (``latex_lb <= 0``) -> ``0.0``. Doing nothing earns
      nothing; you must actually tap and sell rubber to score.
    - harvested rubber but UNPROFITABLE (``profit <= 0``) -> a positive band,
      ``MIN_HARVEST_REWARD..0.5``, rising as losses shrink toward break-even.
    - harvested rubber and PROFITABLE (``profit > 0``) -> ``0.5..1.0``, rising
      with profit and reaching ``1.0`` at the best-baseline profit (``ceil``).

    So a full ``1.0`` requires being profitable; harvesting-while-losing still
    gets meaningful positive credit, and being profitable scores strictly higher.
    """
    loss_floor, ceil = anchors
    if latex_lb <= 0:
        return 0.0
    if profit <= 0:
        span = -loss_floor if loss_floor < 0 else 1.0
        frac = max(0.0, min(1.0, (profit - loss_floor) / span)) if span > 0 else 1.0
        return MIN_HARVEST_REWARD + (0.5 - MIN_HARVEST_REWARD) * frac
    frac = min(1.0, profit / ceil) if ceil > 0 else 1.0
    return 0.5 + 0.5 * frac
