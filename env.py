"""HUD v6 environment: run a rubber-tree farm for 30 years, maximise profit.

The agent acts on a living :class:`Farm.Farm` simulation through MCP tools
(observe / render / water / fertilize / tap / advance / status). The reward is
the farm's end-of-horizon **profit** (rubber revenue minus water + fertilizer
cost), normalised against a sensible baseline operator so it lands in ``0..1``.

Structure mirrors the blank template: tools are served from an in-process
FastMCP server started in ``@env.initialize`` and published as an ``mcp``
capability; tasks are ``@env.template`` async generators that prompt, let the
agent act, then yield the reward.

    hud eval env.py claude
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

from hud import Environment
from hud.capabilities import Capability

from Farm import Farm, default_spec, reference_profit
from Farm.actions import Targets

env = Environment(name="rubber-farm")

# ---------------------------------------------------------------------------
# Simulation state. HUD runs one container per evaluation, so a single
# module-global farm is safe (no in-process parallelism).
# ---------------------------------------------------------------------------
_sim: dict = {"farm": None, "reference": 0.0}


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
    """Return the running economics and how many years remain in the run."""
    farm = _farm()
    e = farm.observe()["economics"]
    return (
        f"year={farm.year} years_left={_years_left(farm)} finished={farm.finished} | "
        f"revenue=${e['revenue']} cost=${e['cost']} profit=${e['profit']} "
        f"rubber={e['latex_lb']}lb"
    )


async def water(gallons_per_tree: float, target: str = "all") -> str:
    """Irrigate trees now (target: all|mature|immature|'row,col'). Costs water $/gal."""
    return json.dumps(_farm().water(gallons_per_tree, _parse_target(target)))


async def fertilize(
    nitrogen: float = 0.0,
    phosphorus: float = 0.0,
    potassium: float = 0.0,
    target: str = "all",
) -> str:
    """Add N/P/K nutrient-units to the soil now. Costs fertilizer $/unit (the big cost)."""
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
        for tool in (observe, render, status, water, fertilize, tap, advance):
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
how you earn. Watering and (especially) fertilizing cost money, so spend only
when it pays off.

How the farm works:
- Trees grow trunk girth over years and become *tappable* at ~6 years / ~45 cm.
- Tapping yields latex daily; tapping continuously wears down the tapping panel
  (lower yield) while resting lets it recover. Newly matured trees aren't tapped
  until you tap them.
- Trees need soil moisture (rain + irrigation) and N/P/K nutrients (used up by
  growth, replenished by fertilizer). Starving either lowers health -> lower
  growth and yield. Trees die of old age around 28-34 years.
- Prices (rubber $/lb, water $/gal, fertilizer $/unit) drift year to year, so
  timing your fertilizer purchases matters.

Tools: observe (JSON state), render (ASCII map), status (economics + years
left), water, fertilize, tap, advance(days). Use `advance` to move time forward
(e.g. one year at a time) and re-tap newly matured trees as you go. Any time you
don't explicitly play out continues with your current tapping settings but no
further watering/fertilizing.

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
    # Profit a sensible baseline operator earns on this exact scenario; used to
    # scale the reward so ~baseline -> 1.0 and losing money -> 0.0.
    _sim["reference"] = reference_profit(spec)

    yield _PROMPT.format(
        years=duration_years,
        y0=spec.start_year,
        y1=spec.start_year + duration_years,
        render=farm.render(),
    )

    profit = farm.profit
    reference = _sim["reference"]
    reward = profit / reference if reference > 0 else (1.0 if profit > 0 else 0.0)
    yield max(0.0, min(1.0, reward))


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
        print("reference profit:", round(_sim["reference"], 2))
        print("reward:", round(reward, 4))

    asyncio.run(_smoke())
