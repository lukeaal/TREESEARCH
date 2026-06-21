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
