"""A tiny text-based front end for the rubber farm simulation.

Two ways to run it from the repo root:

    python -m Farm.textsim            # interactive REPL
    python -m Farm.textsim --demo     # scripted, non-interactive demo

Interactive commands (a 'target' is: all | mature | immature | r,c)::

    render                      show the ASCII farm + stats
    obs                         print the structured observation dict
    water <gal> [target]        irrigate (default target: all)
    fert <n> <p> <k> [target]   fertilize with N/P/K nutrient-units
    tap on|off [target]         start/stop tapping (default target: mature)
    step [days]                 advance the calendar (default 1 day)
    year                        advance roughly one year (365 days)
    help                        show this help
    quit                        exit
"""

from __future__ import annotations

import sys

from .objects import Farm
from .spec import default_spec


def _parse_target(token: str | None):
    if token is None:
        return None
    if token in ("all", "mature", "immature"):
        return token
    if "," in token:
        r, c = token.split(",", 1)
        return [(int(r), int(c))]
    raise ValueError(f"bad target: {token!r}")


def _handle(farm: Farm, line: str) -> bool:
    """Execute one command line. Returns False to quit."""
    parts = line.split()
    if not parts:
        return True
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        print(__doc__)
    elif cmd in ("render", "r"):
        print(farm.render())
    elif cmd in ("obs", "o"):
        import json

        print(json.dumps(farm.observe(), indent=2))
    elif cmd == "water":
        gal = float(args[0])
        target = _parse_target(args[1]) if len(args) > 1 else "all"
        print(farm.water(gal, target))
    elif cmd in ("fert", "fertilize"):
        n, p, k = (float(args[0]), float(args[1]), float(args[2]))
        target = _parse_target(args[3]) if len(args) > 3 else "all"
        print(farm.fertilize(n, p, k, target))
    elif cmd == "tap":
        on = args[0].lower() in ("on", "true", "1", "yes")
        target = _parse_target(args[1]) if len(args) > 1 else "mature"
        print(farm.tap(on, target))
    elif cmd in ("step", "s"):
        days = int(args[0]) if args else 1
        print(farm.step(days))
    elif cmd in ("year", "y"):
        print(farm.step(365))
    else:
        print(f"unknown command: {cmd!r} (try 'help')")
    return True


def interactive() -> None:
    farm = Farm.from_spec(default_spec())
    print("Rubber Farm text sim. Type 'help' for commands, 'quit' to exit.\n")
    print(farm.render())
    while not farm.finished:
        try:
            line = input("\nfarm> ")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if not _handle(farm, line):
                break
        except (ValueError, IndexError) as exc:
            print(f"error: {exc}")
    print("\nFinal state:")
    print(farm.render())


def demo() -> None:
    """A scripted run: tap every mature tree, irrigate/fertilize, harvest 30 years."""
    farm = Farm.from_spec(default_spec(seed=0))
    print("=== DEMO: a simple 'tap everything, feed steadily' strategy ===\n")
    print(farm.render())
    print()

    while not farm.finished:
        obs = farm.observe()
        a = obs["averages"]
        # Begin tapping anything that has matured.
        farm.tap(on=True, targets="mature")
        # Only irrigate when the soil is drying out (rain does most of the work).
        if a["moisture"] < 0.45:
            farm.water(gallons_per_tree=5, targets="all")
        # Only fertilize when nutrients run low (fertilizer is the big cost).
        if a["nutrients"] < 0.35:
            farm.fertilize(nitrogen=0.2, phosphorus=0.15, potassium=0.15, targets="all")
        farm.step(days=30)
        if farm.day_of_year < 30:  # roughly once per simulated year
            obs = farm.observe()
            e = obs["economics"]
            print(
                f"Year {obs['year']}: profit ${e['profit']:>10.0f} | "
                f"rubber {e['latex_lb']:>8.1f} lb | "
                f"tappable {obs['trees']['tappable']:>2}/{obs['trees']['living']}"
            )

    print("\n=== FINAL ===")
    print(farm.render())


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if "--demo" in argv:
        demo()
    else:
        interactive()


if __name__ == "__main__":
    main()
