"""The living simulation: :class:`RubberTree` and :class:`Farm`.

This is a deliberately *simple but faithful* abstraction of a Hevea (rubber)
plantation:

* Trees grow trunk girth over years. A tree becomes **tappable** once it is old
  enough and thick enough (~6 years / ~45 cm girth in this model).
* Tapping the bark yields latex (measured here in pounds of dry rubber). Tapping
  every day exhausts the tapping panel, so yield falls unless the tree is rested
  — which nudges an operator toward sustainable, alternate-day style tapping.
* Trees need soil **moisture** (rain + irrigation) and **N/P/K nutrients**
  (depleted by growth, replenished by fertilizer). Starving either one lowers
  health, which lowers both growth and yield.
* Trees decline and eventually die of old age past ~28-32 years.

The :class:`Farm` is the engine: build it from a :class:`Farm.spec.FarmSpec`,
apply actions on a given day, then ``step()`` to advance the calendar and watch
state evolve. All randomness flows through a single seeded RNG so runs are
reproducible, while small day-to-day noise keeps things lifelike.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .actions import Action, Fertilize, Tap, Targets, Wait, Water
from .spec import FarmSpec, TreeSpec

DAYS_PER_YEAR = 365

# --- Tree biology constants (tunable knobs for the abstraction) -------------
SEEDLING_GIRTH_CM = 3.0
MAX_GIRTH_CM = 70.0
MATURE_AGE_DAYS = 6 * DAYS_PER_YEAR
MATURE_GIRTH_CM = 45.0
BASE_GIRTH_GROWTH_CM = 0.045  # max girth gained per day for a perfect seedling

SENESCENCE_AGE_DAYS = 28 * DAYS_PER_YEAR  # decline begins
MAX_AGE_DAYS = 34 * DAYS_PER_YEAR  # hard cap on lifespan

# Latex / tapping
BASE_DAILY_LATEX_LB = 0.045  # lb dry rubber, reference tree, tapped, per day
REFERENCE_GIRTH_CM = 50.0
# The panel always heals a little toward full; tapping eats into it. Daily
# tapping therefore settles at a sustainable-but-reduced panel (~0.6), while a
# rested tree returns to ~1.0 — so over-tapping costs yield without killing the
# tree, nudging operators toward alternate-day style tapping.
TAP_PANEL_STRESS = 0.012  # panel health lost per day of tapping
TAP_PANEL_RECOVERY = 0.030  # fraction of the gap to full healed each day

# Soil dynamics. Evaporation is proportional to current moisture, so soil
# relaxes toward an equilibrium set by rainfall + irrigation (rainforest-ish):
# equilibrium_moisture ~= daily_water_in / EVAPORATION_RATE.
EVAPORATION_RATE = 0.09  # fraction of soil moisture lost per day
GALLONS_TO_MOISTURE = 0.020  # moisture gained per gallon per tree
BASE_NUTRIENT_DRAW = 0.0004  # each of N/P/K drawn per day (baseline metabolism)
GROWTH_NUTRIENT_DRAW = 0.004  # extra nutrient draw scaled by growth activity
TAP_NUTRIENT_DRAW = 0.0008  # extra nutrient draw while tapping

HEALTH_ADJUST_RATE = 0.06  # how fast health tracks its target each day


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Weather:
    """The day's environment, shared by every tree on the farm."""

    rainfall: float = 0.0  # moisture added to every tree's soil today
    temperature_stress: float = 0.0  # 0 = ideal, 1 = harsh (mild health drag)


@dataclass
class RubberTree:
    """A single rubber tree with its own genetics, soil, and tapping state."""

    row: int
    col: int
    age_days: int = 0

    # Innate genetics (sampled with noise at init; >1 is a better-than-average tree)
    growth_rate: float = 1.0
    yield_multiplier: float = 1.0

    # Physiological state
    girth_cm: float = SEEDLING_GIRTH_CM
    health: float = 1.0
    panel_health: float = 1.0  # condition of the tapping panel (over-tapping -> down)

    # Soil around this tree (0..1 each)
    moisture: float = 0.5
    nitrogen: float = 0.5
    phosphorus: float = 0.5
    potassium: float = 0.5

    tapping: bool = False
    alive: bool = True

    # Bookkeeping
    last_latex_lb: float = 0.0
    cumulative_latex_lb: float = 0.0

    # Per-tree RNG so each tree's day-to-day noise is independent of every other
    # tree and of the actions taken. The Farm seeds this from (seed, row, col).
    rng: random.Random = field(
        default_factory=random.Random, repr=False, compare=False
    )

    # ----------------------------------------------------------------- traits
    @property
    def age_years(self) -> float:
        return self.age_days / DAYS_PER_YEAR

    @property
    def is_tappable(self) -> bool:
        """Old enough and thick enough that tapping draws latex."""
        return (
            self.alive
            and self.age_days >= MATURE_AGE_DAYS
            and self.girth_cm >= MATURE_GIRTH_CM
        )

    @property
    def stage(self) -> str:
        if not self.alive:
            return "dead"
        if self.is_tappable:
            return "tapping" if self.tapping else "mature"
        if self.age_days < 2 * DAYS_PER_YEAR:
            return "seedling"
        return "young"

    @property
    def nutrient_level(self) -> float:
        """Single 0..1 summary of soil fertility (the limiting nutrient matters most)."""
        return (
            min(self.nitrogen, self.phosphorus, self.potassium) * 0.5
            + (self.nitrogen + self.phosphorus + self.potassium) / 3.0 * 0.5
        )

    # --------------------------------------------------------------- actions
    def water(self, gallons: float) -> None:
        self.moisture = _clamp(self.moisture + gallons * GALLONS_TO_MOISTURE)

    def fertilize(self, nitrogen: float, phosphorus: float, potassium: float) -> None:
        self.nitrogen = _clamp(self.nitrogen + nitrogen)
        self.phosphorus = _clamp(self.phosphorus + phosphorus)
        self.potassium = _clamp(self.potassium + potassium)

    # ------------------------------------------------------------ daily step
    def _senescence_factor(self) -> float:
        """1.0 in the prime of life, fading to 0 as the tree gets very old."""
        if self.age_days <= SENESCENCE_AGE_DAYS:
            return 1.0
        span = MAX_AGE_DAYS - SENESCENCE_AGE_DAYS
        return _clamp(1.0 - (self.age_days - SENESCENCE_AGE_DAYS) / span)

    def step(self, weather: Weather, noise: float) -> float:
        """Advance one day. Returns latex (lb) collected from this tree today."""
        self.last_latex_lb = 0.0
        if not self.alive:
            return 0.0

        self.age_days += 1

        # Draw every random for the day up front, so the count of RNG draws is
        # the same whether or not the tree is being tapped. That keeps each
        # tree's stochastic stream identical across different operator policies,
        # which is what makes rewards comparable for RL.
        r = self.rng
        evap_j = 1.0 + r.gauss(0.0, noise)
        health_noise = r.gauss(0.0, noise * 0.05)
        growth_j = 1.0 + r.gauss(0.0, noise)
        draw_j = 1.0 + r.gauss(0.0, noise)
        latex_j = max(0.0, 1.0 + r.gauss(0.0, noise))
        panel_j = 1.0 + r.gauss(0.0, noise)

        # --- soil moisture: rain in, proportional evaporation out ----------
        self.moisture = _clamp(self.moisture + weather.rainfall)
        self.moisture = _clamp(
            self.moisture - self.moisture * EVAPORATION_RATE * evap_j
        )

        # --- adequacy of the two big resources ----------------------------
        water_ok = _clamp(self.moisture / 0.5)  # happy at >=0.5 moisture
        nutrient_ok = _clamp(self.nutrient_level / 0.5)

        senescence = self._senescence_factor()

        # --- health relaxes toward what conditions can support ------------
        target_health = (
            0.5 * water_ok
            + 0.3 * nutrient_ok
            + 0.2 * self.panel_health
        ) * senescence
        target_health = _clamp(target_health - 0.15 * weather.temperature_stress)
        self.health = _clamp(
            self.health + (target_health - self.health) * HEALTH_ADJUST_RATE
            + health_noise
        )

        # --- growth: logistic approach to MAX_GIRTH, gated by health/resources
        growth_headroom = max(0.0, 1.0 - self.girth_cm / MAX_GIRTH_CM)
        growth = (
            BASE_GIRTH_GROWTH_CM
            * self.growth_rate
            * self.health
            * min(water_ok, nutrient_ok)
            * growth_headroom
            * senescence
            * growth_j
        )
        growth = max(0.0, growth)
        self.girth_cm = min(MAX_GIRTH_CM, self.girth_cm + growth)

        # growth and baseline metabolism draw down nutrients
        draw = (
            BASE_NUTRIENT_DRAW
            + GROWTH_NUTRIENT_DRAW * (growth / BASE_GIRTH_GROWTH_CM)
        ) * draw_j
        if self.tapping and self.is_tappable:
            draw += TAP_NUTRIENT_DRAW
        self.nitrogen = _clamp(self.nitrogen - draw)
        self.phosphorus = _clamp(self.phosphorus - draw)
        self.potassium = _clamp(self.potassium - draw)

        # --- latex production / panel dynamics ----------------------------
        # The panel always heals toward full; tapping then eats into it.
        self.panel_health = _clamp(
            self.panel_health + TAP_PANEL_RECOVERY * (1.0 - self.panel_health)
        )
        if self.tapping and self.is_tappable:
            girth_factor = self.girth_cm / REFERENCE_GIRTH_CM
            latex = (
                BASE_DAILY_LATEX_LB
                * self.yield_multiplier
                * self.health
                * girth_factor
                * self.panel_health
                * senescence
                * latex_j
            )
            self.last_latex_lb = max(0.0, latex)
            self.cumulative_latex_lb += self.last_latex_lb
            # tapping costs the tree extra water and stresses the panel
            self.moisture = _clamp(self.moisture - 0.01)
            self.panel_health = _clamp(self.panel_health - TAP_PANEL_STRESS * panel_j)

        # --- death: old age, or sustained collapse of health --------------
        if self.age_days >= MAX_AGE_DAYS or self.health <= 0.02:
            self.alive = False

        return self.last_latex_lb

    def symbol(self) -> str:
        """A single character for ASCII rendering of the farm grid."""
        if not self.alive:
            return "+"
        if self.health < 0.35:
            return "x"  # struggling
        return {
            "seedling": "s",
            "young": "y",
            "mature": "T",
            "tapping": "$",
        }.get(self.stage, "?")


@dataclass
class Ledger:
    """Running tally of money and resources over the whole run."""

    revenue: float = 0.0
    water_cost: float = 0.0
    fertilizer_cost: float = 0.0
    water_gallons: float = 0.0
    fertilizer_units: float = 0.0
    latex_lb: float = 0.0

    @property
    def cost(self) -> float:
        return self.water_cost + self.fertilizer_cost

    @property
    def profit(self) -> float:
        return self.revenue - self.cost


class Farm:
    """The simulation engine: a grid of trees plus economics and a calendar."""

    def __init__(self, spec: FarmSpec):
        self.spec = spec
        # Separate streams: one for tree genetics at init, one for daily weather.
        # Per-tree day-to-day noise lives on each tree (seeded by position), so a
        # scenario's stochastic realisation is fixed no matter what actions the
        # operator takes — only the deterministic dynamics differ.
        self._init_rng = random.Random(spec.seed)
        self.weather_rng = random.Random(spec.seed ^ 0x9E3779B9)
        self.day = 0  # days elapsed since the start of the run
        self.ledger = Ledger()
        # Spending booked against each calendar year, so the annual budget cap
        # resets automatically as the calendar rolls over.
        self._spend_by_year: dict[int, float] = {}
        self.grid: list[list[RubberTree | None]] = [
            [None for _ in range(spec.cols)] for _ in range(spec.rows)
        ]
        for ts in spec.trees:
            self.grid[ts.row][ts.col] = self._make_tree(ts)

    # ------------------------------------------------------------- factory
    @classmethod
    def from_spec(cls, spec: FarmSpec) -> "Farm":
        return cls(spec)

    def _make_tree(self, ts: TreeSpec) -> RubberTree:
        noise = self.spec.noise
        rng = self._init_rng
        growth_rate = (
            ts.growth_rate
            if ts.growth_rate is not None
            else max(0.3, rng.gauss(1.0, noise))
        )
        yield_multiplier = (
            ts.yield_multiplier
            if ts.yield_multiplier is not None
            else max(0.3, rng.gauss(1.0, noise))
        )
        age_days = int(ts.age_years * DAYS_PER_YEAR)

        # Approximate the girth a tree of this age would have reached, with noise.
        maturity_frac = _clamp(age_days / MATURE_AGE_DAYS)
        girth = SEEDLING_GIRTH_CM + (MATURE_GIRTH_CM + 8 - SEEDLING_GIRTH_CM) * (
            1 - math.exp(-2.2 * maturity_frac)
        )
        girth *= max(0.5, 1.0 + rng.gauss(0.0, noise * 0.5))
        girth = min(MAX_GIRTH_CM, max(SEEDLING_GIRTH_CM, girth))

        # Stable per-tree seed from the scenario seed and the tree's position.
        tree_seed = (
            (self.spec.seed * 73856093) ^ (ts.row * 19349663) ^ (ts.col * 83492791)
        ) & 0x7FFFFFFF

        return RubberTree(
            row=ts.row,
            col=ts.col,
            age_days=age_days,
            growth_rate=growth_rate,
            yield_multiplier=yield_multiplier,
            girth_cm=girth,
            health=_clamp(rng.gauss(0.9, noise * 0.3)),
            panel_health=1.0,
            moisture=_clamp(rng.gauss(0.55, noise * 0.4)),
            nitrogen=_clamp(rng.gauss(0.55, noise * 0.4)),
            phosphorus=_clamp(rng.gauss(0.55, noise * 0.4)),
            potassium=_clamp(rng.gauss(0.55, noise * 0.4)),
            rng=random.Random(tree_seed),
        )

    # -------------------------------------------------------------- calendar
    @property
    def year(self) -> int:
        return self.spec.start_year + self.day // DAYS_PER_YEAR

    @property
    def day_of_year(self) -> int:
        return self.day % DAYS_PER_YEAR

    @property
    def finished(self) -> bool:
        return self.day >= self.spec.duration_years * DAYS_PER_YEAR

    @property
    def market_rate(self) -> float:
        return self.spec.market_rate.get(self.year)

    @property
    def water_price(self) -> float:
        return self.spec.water_cost.get(self.year)

    @property
    def fertilizer_price(self) -> float:
        return self.spec.fertilizer_cost.get(self.year)

    # ------------------------------------------------------------- budget
    @property
    def annual_budget(self) -> float:
        return self.spec.annual_budget

    @property
    def spent_this_year(self) -> float:
        return self._spend_by_year.get(self.year, 0.0)

    @property
    def remaining_budget(self) -> float:
        """Money still available to spend in the current calendar year."""
        return max(0.0, self.annual_budget - self.spent_this_year)

    def _book_spend(self, amount: float) -> None:
        self._spend_by_year[self.year] = self.spent_this_year + amount

    # ------------------------------------------------------------ tree access
    def trees(self):
        """Iterate over every living-or-dead tree on the grid."""
        for row in self.grid:
            for tree in row:
                if tree is not None:
                    yield tree

    def living_trees(self):
        return (t for t in self.trees() if t.alive)

    def tree_at(self, row: int, col: int) -> RubberTree | None:
        if 0 <= row < self.spec.rows and 0 <= col < self.spec.cols:
            return self.grid[row][col]
        return None

    def _resolve_targets(self, targets: Targets) -> list[RubberTree]:
        if isinstance(targets, str):
            if targets == "all":
                return list(self.living_trees())
            if targets == "mature":
                return [t for t in self.living_trees() if t.is_tappable]
            if targets == "immature":
                return [t for t in self.living_trees() if not t.is_tappable]
            raise ValueError(f"unknown target selector: {targets!r}")
        out: list[RubberTree] = []
        for (r, c) in targets:
            tree = self.tree_at(r, c)
            if tree is not None and tree.alive:
                out.append(tree)
        return out

    def _affordable_factor(self, requested_cost: float) -> float:
        """Fraction of a requested spend that still fits in this year's budget."""
        remaining = self.remaining_budget
        if requested_cost <= 0:
            return 0.0
        return min(1.0, remaining / requested_cost)

    # --------------------------------------------------------------- actions
    def apply(self, action: Action) -> dict:
        """Apply one action immediately. Returns a small summary dict."""
        if isinstance(action, Water):
            return self._do_water(action)
        if isinstance(action, Fertilize):
            return self._do_fertilize(action)
        if isinstance(action, Tap):
            return self._do_tap(action)
        if isinstance(action, Wait):
            return {"action": "wait"}
        raise TypeError(f"unknown action: {action!r}")

    def _do_water(self, action: Water) -> dict:
        # Two safety rails make spending forgiving: (1) you only pay for water the
        # soil actually absorbs (moisture caps at 1.0), and (2) the spend is
        # clamped to this year's remaining budget, so an over-eager order is
        # scaled down to what you can afford instead of blowing past the cap.
        targets = self._resolve_targets(action.targets)
        requested = action.gallons_per_tree * len(targets)
        factor = self._affordable_factor(requested * self.water_price)
        gpt = action.gallons_per_tree * factor
        gallons = 0.0
        for tree in targets:
            before = tree.moisture
            tree.water(gpt)
            gallons += (tree.moisture - before) / GALLONS_TO_MOISTURE
        cost = gallons * self.water_price
        self.ledger.water_gallons += gallons
        self.ledger.water_cost += cost
        self._book_spend(cost)
        return {
            "action": "water",
            "trees": len(targets),
            "gallons": round(gallons, 3),
            "requested_gallons": round(requested, 3),
            "cost": round(cost, 4),
            "budget_remaining": round(self.remaining_budget, 2),
            "budget_capped": factor < 1.0,
        }

    def _do_fertilize(self, action: Fertilize) -> dict:
        # Same two rails as watering: charge only for nutrients the soil absorbs
        # (each of N/P/K caps at 1.0 per tree), and clamp the order to the annual
        # budget. Over-fertilizing is therefore forgiven instead of catastrophic.
        targets = self._resolve_targets(action.targets)
        requested = action.total_units * len(targets)
        factor = self._affordable_factor(requested * self.fertilizer_price)
        n, p, k = (
            action.nitrogen * factor,
            action.phosphorus * factor,
            action.potassium * factor,
        )
        units = 0.0
        for tree in targets:
            before = tree.nitrogen + tree.phosphorus + tree.potassium
            tree.fertilize(n, p, k)
            units += (tree.nitrogen + tree.phosphorus + tree.potassium) - before
        cost = units * self.fertilizer_price
        self.ledger.fertilizer_units += units
        self.ledger.fertilizer_cost += cost
        self._book_spend(cost)
        return {
            "action": "fertilize",
            "trees": len(targets),
            "units": round(units, 4),
            "requested_units": round(requested, 4),
            "cost": round(cost, 4),
            "budget_remaining": round(self.remaining_budget, 2),
            "budget_capped": factor < 1.0,
        }

    def _do_tap(self, action: Tap) -> dict:
        targets = self._resolve_targets(action.targets)
        for tree in targets:
            tree.tapping = action.on
        return {"action": "tap", "on": action.on, "trees": len(targets)}

    # convenience wrappers so callers don't have to import action classes
    def water(self, gallons_per_tree: float, targets: Targets = "all") -> dict:
        return self.apply(Water(gallons_per_tree, targets))

    def fertilize(
        self,
        nitrogen: float = 0.0,
        phosphorus: float = 0.0,
        potassium: float = 0.0,
        targets: Targets = "all",
    ) -> dict:
        return self.apply(Fertilize(nitrogen, phosphorus, potassium, targets))

    def tap(self, on: bool = True, targets: Targets = "mature") -> dict:
        return self.apply(Tap(on, targets))

    # ------------------------------------------------------------------ step
    def _weather(self) -> Weather:
        """Generate the day's weather with a gentle seasonal rhythm + noise."""
        rng = self.weather_rng
        # Seasonal wet/dry cycle over the year. In the wet season rain easily
        # keeps soil near its happy point; in the dry season irrigation pays off.
        season = 0.5 + 0.5 * math.sin(2 * math.pi * self.day_of_year / DAYS_PER_YEAR)
        rain_chance = 0.40 + 0.40 * season
        rainfall = 0.0
        if rng.random() < rain_chance:
            rainfall = max(0.0, rng.gauss(0.085, 0.04))
        temp_stress = _clamp(rng.gauss(0.12 * (1 - season), 0.06))
        return Weather(rainfall=rainfall, temperature_stress=temp_stress)

    def step(self, days: int = 1) -> dict:
        """Advance the simulation by ``days`` days, selling latex as it's tapped.

        Returns a summary of what happened over the stepped interval.
        """
        latex_lb = 0.0
        revenue = 0.0
        deaths = 0
        for _ in range(days):
            if self.finished:
                break
            weather = self._weather()
            rate = self.market_rate
            for tree in self.trees():
                was_alive = tree.alive
                produced = tree.step(weather, self.spec.noise)
                if produced:
                    latex_lb += produced
                    revenue += produced * rate
                if was_alive and not tree.alive:
                    deaths += 1
            self.day += 1

        self.ledger.latex_lb += latex_lb
        self.ledger.revenue += revenue
        return {
            "days": days,
            "latex_lb": latex_lb,
            "revenue": revenue,
            "deaths": deaths,
            "day": self.day,
            "year": self.year,
        }

    # --------------------------------------------------------------- observe
    def observe(self) -> dict:
        """A structured snapshot of the whole farm — ideal for an agent/text sim."""
        living = list(self.living_trees())
        tappable = [t for t in living if t.is_tappable]

        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return {
            "day": self.day,
            "year": self.year,
            "day_of_year": self.day_of_year,
            "finished": self.finished,
            "prices": {
                "rubber_per_lb": round(self.market_rate, 4),
                "water_per_gallon": round(self.water_price, 4),
                "fertilizer_per_unit": round(self.fertilizer_price, 4),
            },
            "trees": {
                "total": sum(1 for _ in self.trees()),
                "living": len(living),
                "tappable": len(tappable),
                "tapping": sum(1 for t in living if t.tapping),
            },
            "averages": {
                "age_years": round(avg([t.age_years for t in living]), 2),
                "girth_cm": round(avg([t.girth_cm for t in living]), 2),
                "health": round(avg([t.health for t in living]), 3),
                "panel_health": round(avg([t.panel_health for t in living]), 3),
                "moisture": round(avg([t.moisture for t in living]), 3),
                "nutrients": round(avg([t.nutrient_level for t in living]), 3),
            },
            "budget": {
                "annual": round(self.annual_budget, 2),
                "spent_this_year": round(self.spent_this_year, 2),
                "remaining_this_year": round(self.remaining_budget, 2),
            },
            "economics": {
                "revenue": round(self.ledger.revenue, 2),
                "cost": round(self.ledger.cost, 2),
                "profit": round(self.ledger.profit, 2),
                "latex_lb": round(self.ledger.latex_lb, 2),
                "water_gallons": round(self.ledger.water_gallons, 1),
                "fertilizer_units": round(self.ledger.fertilizer_units, 2),
            },
        }

    def render(self) -> str:
        """A human-friendly text view: ASCII grid + key stats. Great for text sims."""
        obs = self.observe()
        lines: list[str] = []
        lines.append(
            f"Year {obs['year']} (day {obs['day_of_year']}/{DAYS_PER_YEAR}) "
            f"| sim day {obs['day']}"
        )
        p = obs["prices"]
        lines.append(
            f"Prices: rubber ${p['rubber_per_lb']}/lb | "
            f"water ${p['water_per_gallon']}/gal | "
            f"fertilizer ${p['fertilizer_per_unit']}/unit"
        )
        lines.append("")
        for row in self.grid:
            lines.append(
                " ".join(tree.symbol() if tree is not None else "." for tree in row)
            )
        lines.append("")
        lines.append("Legend: . empty  s seedling  y young  T mature  $ tapping  x sick  + dead")
        t = obs["trees"]
        a = obs["averages"]
        lines.append(
            f"Trees: {t['living']}/{t['total']} alive, {t['tappable']} tappable, "
            f"{t['tapping']} tapping"
        )
        lines.append(
            f"Avg: age {a['age_years']}y  girth {a['girth_cm']}cm  health {a['health']}  "
            f"panel {a['panel_health']}  moisture {a['moisture']}  nutrients {a['nutrients']}"
        )
        b = obs["budget"]
        lines.append(
            f"Budget: ${b['remaining_this_year']}/{b['annual']} left this year "
            f"(spent ${b['spent_this_year']})"
        )
        e = obs["economics"]
        lines.append(
            f"Economics: revenue ${e['revenue']}  cost ${e['cost']}  "
            f"PROFIT ${e['profit']}  ({e['latex_lb']} lb rubber)"
        )
        return "\n".join(lines)

    # --------------------------------------------------------------- scoring
    @property
    def profit(self) -> float:
        return self.ledger.profit
