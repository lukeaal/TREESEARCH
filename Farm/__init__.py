"""Rubber tree farm simulation.

A small, clean simulation of a rubber (Hevea) plantation. The goal of the game
is to maximise the rubber yield from the farm over ~30 years while keeping costs
(water + fertilizer) as low as possible.

Quick start::

    from Farm import Farm, default_spec, Water, Fertilize, Tap

    farm = Farm.from_spec(default_spec(seed=0))
    print(farm.render())

    # take some actions on day 0...
    farm.water(gallons_per_tree=8)
    farm.fertilize(nitrogen=0.2, phosphorus=0.1, potassium=0.1)
    farm.tap(on=True, targets="mature")

    # ...then advance time and observe the new state
    summary = farm.step(days=7)
    print(farm.observe())
"""

from .actions import Action, Fertilize, Tap, Targets, Wait, Water
from .objects import Farm, Ledger, RubberTree, Weather
from .spec import FarmSpec, Schedule, TreeSpec, default_spec
from .strategies import (
    best_baseline_profit,
    do_nothing,
    reference_profit,
    reward_anchors,
    run_policy,
    scale_reward,
    tap_only,
    tend_and_tap,
    worst_case,
)

__all__ = [
    "Farm",
    "RubberTree",
    "Ledger",
    "Weather",
    "FarmSpec",
    "TreeSpec",
    "Schedule",
    "default_spec",
    "Action",
    "Water",
    "Fertilize",
    "Tap",
    "Wait",
    "Targets",
    "run_policy",
    "reference_profit",
    "best_baseline_profit",
    "reward_anchors",
    "scale_reward",
    "do_nothing",
    "tap_only",
    "tend_and_tap",
    "worst_case",
]
