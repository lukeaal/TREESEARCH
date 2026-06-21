"""HUD v6 environment: run a rubber-tree farm for 30 years, maximise profit.

The agent acts on a living :class:`Farm.Farm` simulation through MCP tools
(observe / render / water / fertilize / tap / advance / status). The reward is
the farm's end-of-horizon **profit** (rubber revenue minus water + fertilizer
cost), rescaled into ``0..1`` against three baselines on the same scenario:
``0.0`` = the worst-case operator (spends maximally, never harvests), ``0.5`` =
a sensible human-like operator (``tend_and_tap``), and ``1.0`` = the best simple
baseline (tap everything). So a money-losing or do-nothing agent still earns a
positive reward, a human-level strategy scores ~0.5, and beating it pushes
toward 1.0.

Structure mirrors the blank template: tools are served from an in-process
FastMCP server started in ``@env.initialize`` and published as an ``mcp``
capability; tasks are ``@env.template`` async generators that prompt, let the
agent act, then yield the reward.

    hud eval env.py claude
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import socket

from hud import Environment
from hud.capabilities import Capability

from Farm import Farm, default_spec, reward_anchors, scale_reward
from Farm.actions import Targets

env = Environment(name="rubber-farm")

# ---------------------------------------------------------------------------
# Simulation state. HUD runs one container per evaluation, so a single
# module-global farm is safe (no in-process parallelism).
# ---------------------------------------------------------------------------
_sim: dict = {"farm": None, "reward_anchors": (0.0, 0.0, 0.0)}


def _farm() -> Farm:
    farm = _sim["farm"]
    if farm is None:
        raise RuntimeError("No farm is running. (A task initialises it.)")
    return farm


def _parse_target(target: str) -> Targets:
    """Turn an agent-supplied string into a Farm target selector."""
    target = (target or "all").strip().lower()
    if target in ("all", "mature", "immature"):
        return target  # type: ignore[return-value]
    if "," in target:
        r, c = target.split(",", 1)
        return [(int(r), int(c))]
    raise ValueError(f"bad target {target!r}: use all|mature|immature|'row,col'")


def _years_left(farm: Farm) -> float:
    total = farm.spec.duration_years * 365
    return round((total - farm.day) / 365, 2)


# ---------------------------------------------------------------------------
# Agent-facing tools
# ---------------------------------------------------------------------------
async def observe() -> str:
    """Return a JSON snapshot of the whole farm: date, prices, tree stats, economics."""
    return json.dumps(_farm().observe(), indent=2)


async def render() -> str:
    """Return a human-readable ASCII map of the farm plus key stats."""
    return _farm().render()


async def status() -> str:
    """Return running economics, this year's prices/budget, and years remaining."""
    farm = _farm()
    obs = farm.observe()
    e, p, b = obs["economics"], obs["prices"], obs["budget"]
    return (
        f"year={farm.year} years_left={_years_left(farm)} finished={farm.finished}\n"
        f"prices: rubber ${p['rubber_per_lb']}/lb | water ${p['water_per_gallon']}/gal "
        f"| fertilizer ${p['fertilizer_per_unit']}/unit\n"
        f"budget: ${b['remaining_this_year']} left of ${b['annual']} this year\n"
        f"totals: revenue=${e['revenue']} cost=${e['cost']} profit=${e['profit']} "
        f"rubber={e['latex_lb']}lb"
    )


async def water(gallons_per_tree: float, target: str = "all") -> str:
    """Irrigate trees now (target: all|mature|immature|'row,col').

    You're only charged for water the soil actually absorbs (moisture caps at
    1.0 per tree) and never more than this year's remaining budget — so an
    over-sized order is clamped, not punished. A few gallons/tree is plenty.
    """
    return json.dumps(_farm().water(gallons_per_tree, _parse_target(target)))


async def fertilize(
    nitrogen: float = 0.0,
    phosphorus: float = 0.0,
    potassium: float = 0.0,
    target: str = "all",
) -> str:
    """Add N/P/K nutrient-units to the soil now (target: all|mature|immature|'row,col').

    Each of N/P/K is a 0..1 soil level per tree, so sensible amounts are small
    (e.g. 0.2 each). You pay only for what the soil absorbs (levels cap at 1.0)
    and never more than this year's remaining budget, so over-fertilizing just
    saturates the soil cheaply rather than draining your money.
    """
    return json.dumps(
        _farm().fertilize(nitrogen, phosphorus, potassium, _parse_target(target))
    )


async def tap(on: bool = True, target: str = "mature") -> str:
    """Start/stop tapping trees for latex. Only tappable (mature) trees yield rubber."""
    return json.dumps(_farm().tap(on, _parse_target(target)))


async def advance(days: int = 365) -> str:
    """Advance the simulation by N days (default 365), selling latex as it's tapped.

    Standing tapping settings persist; watering/fertilizing are one-off and are
    not repeated automatically.
    """
    farm = _farm()
    result = farm.step(days)
    result["years_left"] = _years_left(farm)
    result["finished"] = farm.finished
    return json.dumps(result)


def _apply_plan_action(sim: Farm, step: dict) -> dict:
    """Apply one plan step to a (throwaway) farm copy. See ``simulate`` for the schema."""
    tool = str(step.get("tool", "")).strip().lower()
    if tool == "advance":
        return sim.step(int(step.get("days", 365)))
    if tool == "tap":
        return sim.tap(bool(step.get("on", True)), _parse_target(step.get("target", "mature")))
    if tool == "water":
        return sim.water(float(step.get("gallons", 0.0)), _parse_target(step.get("target", "all")))
    if tool == "fertilize":
        return sim.fertilize(
            float(step.get("n", 0.0)),
            float(step.get("p", 0.0)),
            float(step.get("k", 0.0)),
            _parse_target(step.get("target", "all")),
        )
    raise ValueError(f"unknown plan tool {tool!r}: use tap|water|fertilize|advance")


async def quote(action: str) -> str:
    """Preview what a single action would COST right now — without doing it.

    Pass one action as a JSON object using the same schema as a `simulate` step,
    e.g. ``{"tool": "fertilize", "n": 0.2, "p": 0.15, "k": 0.15, "target": "all"}``
    or ``{"tool": "water", "gallons": 5, "target": "all"}``.

    Returns how much it would actually cost (you only pay for what the soil
    absorbs), how much you *requested* vs. what would be applied, and how much of
    this year's budget would remain. Use it to size spending before committing.
    """
    try:
        step = json.loads(action)
        if not isinstance(step, dict):
            raise ValueError("action must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({"error": f"bad action: {exc}"})

    farm = _farm()
    sim = copy.deepcopy(farm)
    try:
        result = _apply_plan_action(sim, step)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": f"bad action {step!r}: {exc}"})

    result["would_cost"] = round(sim.ledger.cost - farm.ledger.cost, 4)
    result["preview_only"] = True
    result["note"] = "Estimate only — nothing was spent or changed on the real farm."
    return json.dumps(result, indent=2)


async def simulate(plan: str) -> str:
    """Dry-run a batch of actions on a COPY of the farm — no effect on the real run.

    This is your planning sandbox: submit a whole multi-year strategy at once and
    instantly see how it would play out (no waiting, no committing). Use it to
    search for a good policy, then execute the winning actions for real.

    ``plan`` is a JSON list of action steps, each a dict with a ``tool`` key —
    exactly the real tools, replayed in order::

        [
          {"tool": "tap", "on": true, "target": "mature"},
          {"tool": "fertilize", "n": 0.2, "p": 0.15, "k": 0.15, "target": "all"},
          {"tool": "water", "gallons": 5, "target": "all"},
          {"tool": "advance", "days": 365}
        ]

    Returns the projected end-of-plan economics, the normalised reward that
    trajectory would earn (same scale as the final score), and a year-by-year
    profit timeline. The real farm is untouched.
    """
    try:
        steps = json.loads(plan)
        if not isinstance(steps, list):
            raise ValueError("plan must be a JSON list of action steps")
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({"error": f"bad plan: {exc}"})

    sim = copy.deepcopy(_farm())
    timeline: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            return json.dumps({"error": f"each step must be an object, got {step!r}"})
        try:
            result = _apply_plan_action(sim, step)
        except (ValueError, TypeError) as exc:
            return json.dumps({"error": f"bad step {step!r}: {exc}"})
        if str(step.get("tool", "")).lower() == "advance":
            e = sim.observe()["economics"]
            timeline.append(
                {"year": sim.year, "profit": e["profit"], "revenue": e["revenue"], "cost": e["cost"]}
            )
        if sim.finished:
            break

    obs = sim.observe()
    projected_reward = scale_reward(sim.profit, _sim["reward_anchors"])
    return json.dumps(
        {
            "projected_profit": obs["economics"]["profit"],
            "projected_reward": round(projected_reward, 4),
            "final_year": sim.year,
            "finished": sim.finished,
            "economics": obs["economics"],
            "trees": obs["trees"],
            "timeline": timeline[-40:],
            "note": "Dry run on a farm copy — the real farm is unchanged.",
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# In-process MCP capability serving the farm tools
# ---------------------------------------------------------------------------
_MCP_PORT: int = 0
_MCP_SERVER_TASK: "asyncio.Task | None" = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _listening(host: str, port: int, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.2).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"farm MCP server never came up on {host}:{port}")


@env.initialize
async def _up() -> None:
    from fastmcp import FastMCP

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="farm")
        for tool in (
            observe, render, status, water, fertilize, tap,
            advance, quote, simulate,
        ):
            server.tool(tool)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(
                transport="http", host="127.0.0.1", port=_MCP_PORT, show_banner=False
            )
        )
        await _listening("127.0.0.1", _MCP_PORT)
    env.add_capability(
        Capability.mcp(name="farm", url=f"http://127.0.0.1:{_MCP_PORT}/mcp")
    )


@env.shutdown
async def _down() -> None:
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None
    _sim["farm"] = None


# ---------------------------------------------------------------------------
# Task: manage the farm for its full horizon, graded on profit
# ---------------------------------------------------------------------------
_PROMPT = """\
You are managing a rubber-tree plantation for {years} years (years {y0}-{y1}).
Goal: MAXIMISE PROFIT = rubber sales revenue minus your spending on water and
fertilizer. Rubber sells at the market rate each year; tapping mature trees is
how you earn. Every dollar you DON'T spend stays as profit.

How the farm works:
- Trees grow trunk girth over years and become *tappable* at ~6 years / ~45 cm.
- Tapping yields latex daily; tapping continuously wears down the tapping panel
  (lower yield) while resting lets it recover. Newly matured trees aren't tapped
  until you tap them.
- Trees need soil moisture (rain + irrigation) and N/P/K nutrients (used up by
  growth, replenished by fertilizer). Starving either lowers health -> lower
  growth and yield. Trees die of old age around 28-34 years.
- Prices (rubber $/lb, water $/gal, fertilizer $/unit) drift year to year.

How costs add up (READ THIS before spending):
- WATER cost  = (gallons the soil actually absorbs) x water $/gal. Moisture caps
  at 1.0 per tree, so gallons beyond what the soil can hold are not charged.
- FERTILIZER cost = (N+P+K units the soil actually absorbs) x fertilizer $/unit.
  Each of N/P/K is a 0..1 soil level per tree and caps at 1.0, so values are
  SMALL — 0.2 each is a normal top-up, not 5 or 50.
- Both draw from a fixed ANNUAL BUDGET (see `status`/`observe`). Spending is
  capped at the remaining budget; an over-order is scaled down, never overspent,
  and the budget resets each year. Every dollar saved stays as profit.
- Worked example: fertilizing all 36 trees with 0.2 N/P/K each (~0.6 units/tree)
  is ~21.6 units; at ~$2.5/unit that's ~$54. Annual tap revenue is only on the
  order of $100-300, so feed in small amounts and let rain do most of the
  watering.
- Before committing, call `quote` to preview an action's exact cost, or
  `simulate` to dry-run a whole strategy.

Tools: observe (JSON state), render (ASCII map), status (economics + prices +
budget + years left), water, fertilize, tap, advance(days), `quote`, and
`simulate`.
- `advance(days)` moves time forward; standing tapping settings persist, but
  watering/fertilizing are one-off (not repeated automatically).
- `quote(action)` previews what a single action would cost right now, without
  spending or changing anything.
- `simulate(plan)` DRY-RUNS a batch of actions on a throwaway copy of the farm
  and reports the projected profit + reward + year-by-year timeline WITHOUT
  waiting or affecting the real run. Use it to search for a strong multi-year
  strategy first, then execute the winning actions for real.

When you've finished managing the full {years} years, reply with a short summary
of your strategy and final profit.

Starting state:
{render}
"""


@env.template(
    id="rubber-farm",
    description="Operate a rubber plantation for ~30 years; reward = normalised profit.",
)
async def rubber_farm(
    seed: int = 0,
    rows: int = 6,
    cols: int = 6,
    duration_years: int = 30,
):
    spec = default_spec(
        rows=rows, cols=cols, seed=seed, duration_years=duration_years
    )
    farm = Farm.from_spec(spec)
    _sim["farm"] = farm
    # Reward anchors for this exact scenario: (worst-case, human, best-baseline)
    # profits mapping to (0.0, 0.5, 1.0). Even poor play scores above 0; a
    # human-like strategy lands near 0.5; beating it climbs toward 1.0.
    _sim["reward_anchors"] = reward_anchors(spec)

    yield _PROMPT.format(
        years=duration_years,
        y0=spec.start_year,
        y1=spec.start_year + duration_years,
        render=farm.render(),
    )

    yield scale_reward(farm.profit, _sim["reward_anchors"])


if __name__ == "__main__":
    # No-model smoke test: drive the tools directly and print the reward a
    # baseline-quality run would earn (should be ~1.0 since it matches baseline).
    async def _smoke() -> None:
        gen = rubber_farm.func(seed=0)
        prompt = await gen.asend(None)
        print(prompt[:600], "...\n")

        # Play the same heuristic the reference uses, via the tools.
        while not _farm().finished:
            obs = json.loads(await observe())
            a = obs["averages"]
            await tap(True, "mature")
            if a["moisture"] < 0.45:
                await water(5, "all")
            if a["nutrients"] < 0.35:
                await fertilize(0.2, 0.15, 0.15, "all")
            await advance(365)

        print("final status:", await status())
        reward = await gen.asend("Tapped everything, fertilized only when low.")
        anchors = tuple(round(a, 2) for a in _sim["reward_anchors"])
        print("reward anchors (worst/human/best):", anchors)
        print("reward:", round(reward, 4))

    asyncio.run(_smoke())
