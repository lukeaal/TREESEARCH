"""Actions an operator (or an agent) can take on a given day.

Actions are small, declarative data objects. You hand them to
``Farm.apply(action)`` and the farm figures out which trees they touch, what
they cost, and how they change the soil/trees. Keeping actions as plain data
(rather than imperative calls) makes them trivial to log, serialise, replay, or
parse out of text for a text-based sim.

Every action carries a ``targets`` selector:

* ``"all"``       -> every living tree,
* ``"mature"``    -> trees old/large enough to tap (tappable trees),
* ``"immature"``  -> living trees that are not yet tappable,
* ``[(r, c), ...]`` -> an explicit list of grid coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

#: A selector describing which trees an action applies to.
Targets = Union[Literal["all", "mature", "immature"], list[tuple[int, int]]]


@dataclass
class Water:
    """Irrigate the targeted trees.

    Cost is ``gallons_per_tree * (#targets) * water_cost($/gal that year)``.
    """

    gallons_per_tree: float
    targets: Targets = "all"


@dataclass
class Fertilize:
    """Add nutrients to the soil around the targeted trees.

    Amounts are in abstract "nutrient-units" (each roughly 0..1 of a tree's soil
    capacity per unit). Cost is the total units applied times the per-unit
    fertilizer price for the year.
    """

    nitrogen: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    targets: Targets = "all"

    @property
    def total_units(self) -> float:
        return self.nitrogen + self.phosphorus + self.potassium


@dataclass
class Tap:
    """Start (``on=True``) or stop (``on=False``) tapping the targeted trees.

    Tapping only yields latex from trees that are actually tappable; flagging an
    immature tree is harmless and simply takes effect once it matures.
    """

    on: bool = True
    targets: Targets = "mature"


@dataclass
class Wait:
    """Do nothing this step (an explicit no-op, handy for text sims)."""


#: Union of all concrete action types.
Action = Union[Water, Fertilize, Tap, Wait]
