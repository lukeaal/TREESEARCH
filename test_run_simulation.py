"""Runnable walkthrough of the rubber farm simulation.

Run it from the repo root:

    .venv/bin/python run_simulation.py

It shows three things in order:

  1. The starting farm and how state evolves day-to-day after a few actions.
  2. A full 30-year run of a sensible "tend + tap" strategy, printed year by year.
  3. A head-to-head of three strategies so you can see the cost/yield trade-off.
"""

from __future__ import annotations

from Farm import Farm, default_spec


def hr(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Day-to-day: take a few actions and watch state change
# ---------------------------------------------------------------------------
def demo_daily() -> None:
    hr("1) DAY-TO-DAY STATE")
    farm = Farm.from_spec(default_spec(seed=0))

    print("\nStarting farm:")
    print(farm.render())

    print("\n-> Action: tap all mature trees, water everyone, add some fertilizer")
    print("  ", farm.tap(on=True, targets="mature"))
    print("  ", farm.water(gallons_per_tree=8, targets="all"))
    print("  ", farm.fertilize(nitrogen=0.2, phosphorus=0.1, potassium=0.1))

    print("\nStepping 1 day at a time for a week (watch latex + moisture):")
    print(f"  {'day':>4} {'latex_lb':>9} {'revenue':>9} {'avg_moist':>10} {'avg_health':>11}")
    for _ in range(7):
        result = farm.step(days=1)
        obs = farm.observe()
        print(
            f"  {result['day']:>4} {result['latex_lb']:>9.3f} "
            f"{result['revenue']:>9.3f} "
            f"{obs['averages']['moisture']:>10.3f} "
            f"{obs['averages']['health']:>11.3f}"
        )

    print("\nState after one week:")
    print(farm.render())


# ---------------------------------------------------------------------------
# 2. A full 30-year run, printed year by year
# ---------------------------------------------------------------------------
def demo_full_run() -> None:
    hr("2) FULL 30-YEAR RUN  (strategy: tap mature, irrigate/feed only when low)")
    farm = Farm.from_spec(default_spec(seed=0))

    header = (
        f"{'year':>5} {'living':>6} {'tappable':>8} {'rubber_lb':>10} "
        f"{'revenue':>10} {'cost':>10} {'profit':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    while not farm.finished:
        obs = farm.observe()
        a = obs["averages"]
        farm.tap(on=True, targets="mature")
        if a["moisture"] < 0.45:
            farm.water(gallons_per_tree=5, targets="all")
        if a["nutrients"] < 0.35:
            farm.fertilize(nitrogen=0.2, phosphorus=0.15, potassium=0.15)
        farm.step(days=365)

        o = farm.observe()
        e = o["economics"]
        print(
            f"{o['year'] - 1:>5} {o['trees']['living']:>6} "
            f"{o['trees']['tappable']:>8} {e['latex_lb']:>10.1f} "
            f"{e['revenue']:>10.0f} {e['cost']:>10.0f} {e['profit']:>10.0f}"
        )

    print("\nFinal farm:")
    print(farm.render())


# ---------------------------------------------------------------------------
# 3. Compare strategies on the same scenario (same seed = same weather/genetics)
# ---------------------------------------------------------------------------
def _run_strategy(name: str, water: bool, fertilize: bool, tap: bool) -> dict:
    farm = Farm.from_spec(default_spec(seed=0))
    while not farm.finished:
        obs = farm.observe()
        a = obs["averages"]
        if tap:
            farm.tap(on=True, targets="mature")
        if water and a["moisture"] < 0.45:
            farm.water(gallons_per_tree=5, targets="all")
        if fertilize and a["nutrients"] < 0.35:
            farm.fertilize(nitrogen=0.2, phosphorus=0.15, potassium=0.15)
        farm.step(days=365)
    e = farm.observe()["economics"]
    return {"name": name, **e}


def demo_compare() -> None:
    hr("3) STRATEGY COMPARISON  (same seed -> same weather + tree genetics)")
    strategies = [
        _run_strategy("do nothing (no tapping)", water=False, fertilize=False, tap=False),
        _run_strategy("tap only", water=False, fertilize=False, tap=True),
        _run_strategy("tap + water", water=True, fertilize=False, tap=True),
        _run_strategy("tap + water + feed", water=True, fertilize=True, tap=True),
    ]
    header = f"\n{'strategy':<26} {'rubber_lb':>10} {'revenue':>10} {'cost':>10} {'profit':>10}"
    print(header)
    print("-" * len(header))
    for s in strategies:
        print(
            f"{s['name']:<26} {s['latex_lb']:>10.1f} {s['revenue']:>10.0f} "
            f"{s['cost']:>10.0f} {s['profit']:>10.0f}"
        )


def main() -> None:
    demo_daily()
    demo_full_run()
    demo_compare()
    hr()
    print("Done. Tweak strategies / spec and re-run to explore the trade-offs.")


if __name__ == "__main__":
    main()
