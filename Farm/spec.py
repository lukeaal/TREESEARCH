"""Specs that describe a rubber farm simulation before it starts.

A :class:`FarmSpec` is a *pure data* description of a scenario:

* the grid of trees you start with (see :class:`TreeSpec`),
* the market rate for raw (dry) rubber per pound, per year,
* the cost of water per gallon, per year,
* the cost of fertilizer per nutrient-unit, per year.

Everything here is plain, serialisable data. The actual living simulation
(:class:`Farm.objects.Farm`) is built from a spec via ``Farm.from_spec(spec)``.
Per-year economic values are held in :class:`Schedule` objects so a 30-year run
can have prices that drift, trend, and wiggle from one year to the next.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    """A value that changes year by year (e.g. ``$/lb`` of rubber).

    ``values[i]`` is the value for ``start_year + i``. Querying a year outside
    the covered range clamps to the first/last known value, so the schedule is
    always defined.
    """

    start_year: int
    values: tuple[float, ...]

    def get(self, year: int) -> float:
        if not self.values:
            return 0.0
        idx = year - self.start_year
        idx = max(0, min(idx, len(self.values) - 1))
        return self.values[idx]

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def constant(cls, value: float, start_year: int, years: int) -> "Schedule":
        return cls(start_year, tuple(float(value) for _ in range(years)))

    @classmethod
    def trend(
        cls,
        start_value: float,
        start_year: int,
        years: int,
        annual_growth: float = 0.0,
        volatility: float = 0.0,
        rng: random.Random | None = None,
        floor: float = 0.0,
    ) -> "Schedule":
        """Build a schedule that compounds ``annual_growth`` with random noise.

        ``annual_growth`` is a fractional drift per year (``0.03`` = +3%/yr),
        ``volatility`` is the fractional standard deviation of the yearly wiggle.
        """
        rng = rng or random.Random()
        out: list[float] = []
        value = float(start_value)
        for _ in range(years):
            noisy = value * (1.0 + rng.gauss(0.0, volatility)) if volatility else value
            out.append(max(floor, noisy))
            value *= 1.0 + annual_growth
        return cls(start_year, tuple(out))


@dataclass
class TreeSpec:
    """One tree's starting placement and (optional) innate traits.

    ``growth_rate`` and ``yield_multiplier`` default to ``None``, meaning "sample
    these from noise at init time" so every tree comes out a little different.
    Provide explicit values to pin a tree's genetics for a reproducible scenario.
    """

    row: int
    col: int
    age_years: float = 0.0
    growth_rate: float | None = None
    yield_multiplier: float | None = None


@dataclass
class FarmSpec:
    """A complete, serialisable description of a scenario to simulate."""

    rows: int
    cols: int
    trees: list[TreeSpec]
    market_rate: Schedule  # $/lb of dry rubber, by year
    water_cost: Schedule  # $/gallon of water, by year
    fertilizer_cost: Schedule  # $/nutrient-unit of fertilizer, by year
    start_year: int = 2025
    duration_years: int = 30
    seed: int = 0
    noise: float = 0.12  # fractional spread used for init + day-to-day jitter
    # Hard ceiling on what an operator may spend (water + fertilizer) within a
    # single calendar year. Caps the downside: the worst you can do is burn the
    # whole budget every year, and every unspent dollar stays as profit.
    annual_budget: float = 800.0

    # --- convenience constructors ------------------------------------------
    @classmethod
    def from_layout(
        cls,
        layout: list[str],
        *,
        market_rate: Schedule,
        water_cost: Schedule,
        fertilizer_cost: Schedule,
        age_years: float = 0.0,
        tree_char: str = "T",
        start_year: int = 2025,
        duration_years: int = 30,
        seed: int = 0,
        noise: float = 0.12,
    ) -> "FarmSpec":
        """Build a spec from an ASCII map.

        Example::

            layout = [
                "TT.TT",
                "T.T.T",
                "TTTTT",
            ]

        Any cell equal to ``tree_char`` becomes a tree; everything else is bare
        ground.
        """
        trees: list[TreeSpec] = []
        cols = max((len(row) for row in layout), default=0)
        for r, line in enumerate(layout):
            for c, ch in enumerate(line):
                if ch == tree_char:
                    trees.append(TreeSpec(row=r, col=c, age_years=age_years))
        return cls(
            rows=len(layout),
            cols=cols,
            trees=trees,
            market_rate=market_rate,
            water_cost=water_cost,
            fertilizer_cost=fertilizer_cost,
            start_year=start_year,
            duration_years=duration_years,
            seed=seed,
            noise=noise,
        )


def default_spec(
    rows: int = 6,
    cols: int = 6,
    seed: int = 0,
    duration_years: int = 30,
    start_year: int = 2025,
    annual_budget: float = 800.0,
) -> FarmSpec:
    """A reasonable starter scenario: a fully planted plot of mixed-age trees.

    Prices trend mildly upward over the 30 years with year-to-year noise, which
    gives an agent a non-trivial timing problem (when to invest vs. harvest).
    """
    rng = random.Random(seed)

    # Plant every cell, with a spread of starting ages so some trees are already
    # tappable while others are still maturing.
    trees = [
        TreeSpec(row=r, col=c, age_years=round(rng.uniform(0.0, 8.0), 2))
        for r in range(rows)
        for c in range(cols)
    ]

    market_rate = Schedule.trend(
        start_value=0.90,  # $/lb dry rubber
        start_year=start_year,
        years=duration_years,
        annual_growth=0.02,
        volatility=0.12,
        rng=rng,
        floor=0.20,
    )
    water_cost = Schedule.trend(
        start_value=0.004,  # $/gallon
        start_year=start_year,
        years=duration_years,
        annual_growth=0.03,
        volatility=0.08,
        rng=rng,
        floor=0.0005,
    )
    fertilizer_cost = Schedule.trend(
        start_value=2.50,  # $/nutrient-unit
        start_year=start_year,
        years=duration_years,
        annual_growth=0.025,
        volatility=0.10,
        rng=rng,
        floor=0.50,
    )

    return FarmSpec(
        rows=rows,
        cols=cols,
        trees=trees,
        market_rate=market_rate,
        water_cost=water_cost,
        fertilizer_cost=fertilizer_cost,
        start_year=start_year,
        duration_years=duration_years,
        seed=seed,
        noise=0.12,
        annual_budget=annual_budget,
    )
