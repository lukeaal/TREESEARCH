"""HUD environment: optimise a 30-year rubber-farm plan, maximise profit.

This is a *plan-submission* (offline-optimisation) task. Instead of stepping a
live simulation, the agent writes a **schedule of actions** — a day-indexed
list-of-lists — and submits it to be scored:

* the outer index is the **day** (0-based) of the run,
* the inner list is the **chronological actions to take on that day**.

`submit_plan` runs that schedule through the simulator from the pristine starting
state and returns the resulting profit plus the normalised reward. The agent can
submit as many times as it likes during a single rollout and inspect the farm at
any day with `observe_at`, iterating to discover the best schedule. The episode
reward is the BEST reward across all submissions.

Profit is rescaled to ``0..1`` against baselines on the same scenario: doing
nothing (or losing money) = 0.0, a sensible human operator = 0.5, and the best
simple baseline = 1.0. Not submitting a plan also scores 0.0.

    hud eval env.py claude
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

from hud import Environment
from hud.capabilities import Capability

from Farm import Farm, default_spec, reward_anchors, scale_reward
from Farm.actions import Targets

env = Environment(name="rubber-farm")

# ---------------------------------------------------------------------------
# Scenario state. HUD runs one container per evaluation, so module-globals are
# safe (no in-process parallelism). There is no *live* farm any more: every
# observation and score is computed by replaying a plan on a fresh farm built
# from the immutable spec, so tool calls never interfere with each other.
# ---------------------------------------------------------------------------
_sim: dict = {
    "spec": None,
    "reward_anchors": (0.0, 0.0, 0.0),
    "best": {"reward": 0.0, "profit": None, "submissions": 0, "plan": None},
}


def _spec():
    spec = _sim["spec"]
    if spec is None:
        raise RuntimeError("No scenario is running. (A task initialises it.)")
    return spec


def _starting_farm() -> Farm:
    """A fresh farm at day 0 built from the scenario spec (no actions applied)."""
    return Farm.from_spec(_spec())


def _parse_target(target: str) -> Targets:
    """Turn an agent-supplied string into a Farm target selector."""
    target = (target or "all").strip().lower()
    if target in ("all", "mature", "immature"):
        return target  # type: ignore[return-value]
    if "," in target:
        r, c = target.split(",", 1)
        return [(int(r), int(c))]
    raise ValueError(f"bad target {target!r}: use all|mature|immature|'row,col'")


def _price_forecast(spec) -> list[dict]:
    """Per-year prices for the whole horizon, so timing decisions are plannable."""
    out: list[dict] = []
    for i in range(spec.duration_years):
        y = spec.start_year + i
        out.append(
            {
                "year": y,
                "rubber_per_lb": round(spec.market_rate.get(y), 4),
                "water_per_gallon": round(spec.water_cost.get(y), 4),
                "fertilizer_per_unit": round(spec.fertilizer_cost.get(y), 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Plan format
# ---------------------------------------------------------------------------
# A plan is a schedule of actions keyed by day. The agent may submit it either
# as a JSON list (index = day) or a JSON object (day-number -> actions):
#
#   [ [ {"tool":"tap","on":true,"target":"mature"} ],   # day 0
#     [],                                               # day 1 (nothing)
#     ... ]
#
#   { "0":   [ {"tool":"tap","target":"mature"} ],
#     "365": [ {"tool":"fertilize","n":0.2,"p":0.15,"k":0.15,"target":"all"} ] }
#
# Each action is a dict with a "tool" key: tap | untap | water | fertilize.
# Tapping is a *standing* setting (persists across days until changed); watering
# and fertilizing are one-off on the day they appear. Days you don't mention
# simply pass with the current tapping settings and no spending.


def _as_action_list(actions) -> list:
    if actions is None:
        return []
    if isinstance(actions, dict):  # a lone action for that day
        return [actions]
    if not isinstance(actions, list):
        raise ValueError(f"each day's value must be a list of actions, got {actions!r}")
    return actions


def _parse_plan(plan: str) -> dict[int, list]:
    """Parse a plan into ``{day: [action, ...]}``. Accepts a list or an object."""
    data = json.loads(plan)
    mapping: dict[int, list] = {}
    if isinstance(data, list):
        for day, actions in enumerate(data):
            mapping[day] = _as_action_list(actions)
    elif isinstance(data, dict):
        for key, actions in data.items():
            try:
                day = int(key)
            except (TypeError, ValueError):
                raise ValueError(f"plan keys must be day numbers, got {key!r}")
            if day < 0:
                raise ValueError(f"day must be >= 0, got {day}")
            mapping[day] = _as_action_list(actions)
    else:
        raise ValueError(
            "plan must be a JSON list (index = day) or object (day -> [actions])"
        )
    return mapping


def _apply_day_action(farm: Farm, action: dict) -> None:
    """Apply one scheduled action to ``farm`` on the current day."""
    if not isinstance(action, dict):
        raise ValueError(f"each action must be an object, got {action!r}")
    tool = str(action.get("tool", "")).strip().lower()
    if tool in ("tap", "untap"):
        on = bool(action.get("on", tool == "tap"))
        farm.tap(on, _parse_target(action.get("target", "mature")))
    elif tool == "water":
        farm.water(float(action.get("gallons", 0.0)), _parse_target(action.get("target", "all")))
    elif tool == "fertilize":
        farm.fertilize(
            float(action.get("n", 0.0)),
            float(action.get("p", 0.0)),
            float(action.get("k", 0.0)),
            _parse_target(action.get("target", "all")),
        )
    elif tool in ("wait", "none", ""):
        return
    else:
        raise ValueError(f"unknown action tool {tool!r}: use tap|untap|water|fertilize")


def _run_plan(
    plan_map: dict[int, list], *, until_day: int | None = None, collect_timeline: bool = False
) -> tuple[Farm, list[dict]]:
    """Replay ``plan_map`` on a fresh farm up to ``until_day`` (default: the end)."""
    spec = _spec()
    farm = Farm.from_spec(spec)
    horizon = spec.duration_years * 365
    end = horizon if until_day is None else max(0, min(int(until_day), horizon))
    timeline: list[dict] = []
    for day in range(end):
        for action in plan_map.get(day, []):
            _apply_day_action(farm, action)
        farm.step(1)
        if collect_timeline and farm.day % 365 == 0:
            e = farm.observe()["economics"]
            timeline.append(
                {
                    "years_elapsed": farm.day // 365,
                    "profit": e["profit"],
                    "revenue": e["revenue"],
                    "cost": e["cost"],
                    "latex_lb": e["latex_lb"],
                }
            )
    return farm, timeline


# ---------------------------------------------------------------------------
# Agent-facing tools
# ---------------------------------------------------------------------------
async def observe() -> str:
    """JSON of the STARTING farm (day 0) plus the full per-year price forecast.

    This is your planning baseline: tree ages/girths/health at the start, soil
    levels, the annual budget, and how rubber/water/fertilizer prices drift over
    every year of the run (so you can time spending).
    """
    spec = _spec()
    snap = _starting_farm().observe()
    snap["horizon_days"] = spec.duration_years * 365
    snap["duration_years"] = spec.duration_years
    snap["price_forecast"] = _price_forecast(spec)
    return json.dumps(snap, indent=2)


async def render() -> str:
    """Human-readable ASCII map of the starting farm plus key stats."""
    return _starting_farm().render()


async def observe_at(day: int, plan: str = "[]") -> str:
    """Inspect the farm after ``day`` days have elapsed under an (optional) plan.

    Replays ``plan`` (same format as `submit_plan`) on a fresh farm for ``day``
    days and returns that snapshot — tree health/girth, soil, running economics
    and budget. Use it to see how a candidate schedule is playing out partway
    through (e.g. which trees have matured, whether soil is starving). Pass an
    empty plan ``[]`` to see the no-action trajectory. Nothing is scored here.
    """
    try:
        plan_map = _parse_plan(plan)
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({"error": f"bad plan: {exc}"})
    try:
        farm, _ = _run_plan(plan_map, until_day=int(day))
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": f"bad plan: {exc}"})
    snap = farm.observe()
    snap["days_elapsed"] = farm.day
    return json.dumps(snap, indent=2)


async def submit_plan(plan: str) -> str:
    """Score a full action schedule: run it through the sim and return profit + reward.

    ``plan`` is day-indexed (the reframed task format):
    * a JSON LIST whose index is the day, each element a list of that day's
      actions, e.g. ``[[{"tool":"tap","target":"mature"}], [], ...]``; or
    * a JSON OBJECT mapping day-number -> actions, e.g.
      ``{"0": [{"tool":"tap","target":"mature"}],
         "365": [{"tool":"fertilize","n":0.2,"p":0.15,"k":0.15}]}``.

    Each action is ``{"tool": ...}`` where tool is one of:
    * ``tap``       — start tapping (``on`` defaults true), ``target`` mature|all|'r,c'
    * ``untap``     — stop tapping (``target`` ...)
    * ``water``     — ``gallons`` per tree, ``target`` ...
    * ``fertilize`` — ``n``/``p``/``k`` (0..1 each), ``target`` ...

    Tapping persists day-to-day; watering/fertilizing apply only on their day.
    The plan runs from the pristine starting farm, so results are deterministic
    and fully comparable. You may submit as many times as you like — your
    EPISODE SCORE is the best reward across all your submissions — so iterate:
    submit, read the timeline, adjust, resubmit.

    Returns this plan's profit, its normalised reward, whether it's a new best,
    the best-so-far, and a year-by-year profit timeline.
    """
    try:
        plan_map = _parse_plan(plan)
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps({"error": f"bad plan: {exc}"})
    try:
        farm, timeline = _run_plan(plan_map, collect_timeline=True)
    except (ValueError, TypeError) as exc:
        return json.dumps({"error": f"bad plan: {exc}"})

    reward = scale_reward(farm.profit, farm.ledger.latex_lb, _sim["reward_anchors"])
    best = _sim["best"]
    best["submissions"] += 1
    is_new_best = reward > best["reward"]
    if is_new_best:
        best.update(reward=reward, profit=farm.profit, plan=plan_map)

    e = farm.observe()["economics"]
    return json.dumps(
        {
            "profit": e["profit"],
            "reward": round(reward, 4),
            "is_new_best": is_new_best,
            "best_reward_so_far": round(best["reward"], 4),
            "submission_number": best["submissions"],
            "economics": e,
            "timeline": timeline,
            "note": "Your episode score is the BEST reward over all submit_plan calls.",
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
        for tool in (observe, render, observe_at, submit_plan):
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
    _sim["spec"] = None


# ---------------------------------------------------------------------------
# Task: design the best {years}-year action schedule, graded on profit
# ---------------------------------------------------------------------------
_PROMPT = """\
You are designing the operating schedule for a rubber-tree plantation over
{years} years (years {y0}-{y1}). You do NOT step through time live; instead you
write a SCHEDULE OF ACTIONS, submit it to be simulated, read the resulting
profit, and refine it. Iterate as many times as you like — your score is the
BEST plan you submit.

Goal: MAXIMISE PROFIT = rubber sales revenue minus spending on water and
fertilizer. Tapping mature trees earns money; every dollar you don't spend stays
as profit.

The plan format (day-indexed):
- A JSON list whose INDEX is the day (0-based), each element a list of the
  actions to take that day; or a JSON object mapping a day-number to its
  actions. Days you omit just pass with no spending and your current tapping.
- Actions (objects with a "tool"):
  * {{"tool":"tap","target":"mature"}}      start tapping (persists day to day)
  * {{"tool":"untap","target":"all"}}       stop tapping
  * {{"tool":"water","gallons":5,"target":"all"}}
  * {{"tool":"fertilize","n":0.2,"p":0.15,"k":0.15,"target":"all"}}
- target is all|mature|immature|'row,col'. There are 365 days per year, so day
  365 is the start of year 2, 730 of year 3, etc.

How the farm works:
- Trees become tappable at ~6 years / ~45 cm girth. Tapping yields latex daily;
  tapping every day wears the panel down (lower yield), resting heals it. Newly
  matured trees are not tapped until you tap them, so re-tap as they mature.
- Trees need soil moisture (rain + irrigation) and N/P/K nutrients; starving
  either lowers health -> lower growth and yield. Trees die of old age ~28-34y.
- Prices drift year to year (see the price_forecast in `observe`).

How costs add up:
- WATER cost = (gallons the soil absorbs) x water $/gal; FERTILIZER cost =
  (N+P+K units absorbed) x fertilizer $/unit. Moisture and each of N/P/K cap at
  1.0 per tree, so amounts are SMALL (0.2 each is a normal feed, not 5 or 50)
  and over-ordering is not charged for what the soil can't hold.
- A fixed ANNUAL BUDGET caps yearly spending; over-orders are scaled down, never
  overspent, and it resets each year. Annual tap revenue is on the order of
  $100-300, so feed sparingly and let rain do most of the watering.

Tools:
- `observe` — starting farm + full per-year price forecast (your planning data).
- `render` — ASCII map of the starting farm.
- `observe_at(day, plan)` — replay a candidate plan for `day` days and inspect
  the farm then (tree maturity, soil, running economics). Nothing is scored.
- `submit_plan(plan)` — simulate a full schedule and get its profit + reward +
  year-by-year timeline. Call it repeatedly; the best result is your score.

Workflow: read `observe`, draft a schedule, `submit_plan`, study the timeline,
adjust (re-tap newly matured trees, feed when soil is low, stop wasteful spend),
and resubmit until the reward stops improving. Finish with a short summary of
your best strategy and its profit.

Starting state:
{render}
"""


@env.template(
    id="rubber-farm",
    description="Design a ~30-year rubber-farm action schedule; reward = normalised profit.",
)
async def rubber_farm(
    seed: int = 0,
    rows: int = 6,
    cols: int = 6,
    duration_years: int = 30,
):
    spec = default_spec(rows=rows, cols=cols, seed=seed, duration_years=duration_years)
    _sim["spec"] = spec
    _sim["reward_anchors"] = reward_anchors(spec)
    _sim["best"] = {"reward": 0.0, "profit": None, "submissions": 0, "plan": None}

    yield _PROMPT.format(
        years=duration_years,
        y0=spec.start_year,
        y1=spec.start_year + duration_years,
        render=_starting_farm().render(),
    )

    # Episode reward = the best plan the agent submitted this rollout. Never
    # submitting scores a hard 0.0 — the same as submitting a do-nothing plan
    # (profit 0 maps to reward 0.0), so credit requires actually harvesting.
    best = _sim["best"]
    yield best["reward"] if best["submissions"] > 0 else 0.0


if __name__ == "__main__":
    # No-model smoke test: build a sensible schedule, submit it, print the reward.
    async def _smoke() -> None:
        gen = rubber_farm.func(seed=0)
        prompt = await gen.asend(None)
        print(prompt[:500], "...\n")

        # A "tend & tap" schedule: tap mature trees at the start of every year and
        # feed/water a little each year. (Re-tapping each year picks up trees that
        # have newly matured.)
        plan: dict[str, list] = {}
        for yr in range(30):
            day = yr * 365
            plan[str(day)] = [
                {"tool": "tap", "target": "mature"},
                {"tool": "water", "gallons": 5, "target": "all"},
                {"tool": "fertilize", "n": 0.2, "p": 0.15, "k": 0.15, "target": "all"},
            ]

        result = json.loads(await submit_plan(json.dumps(plan)))
        print("submit_plan ->", {k: result[k] for k in ("profit", "reward", "is_new_best")})
        print("timeline tail:", result["timeline"][-2:])

        # observe_at midway through the same plan
        mid = json.loads(await observe_at(365 * 10, json.dumps(plan)))
        print("year 10 snapshot: tappable=%s health=%s profit=%s" % (
            mid["trees"]["tappable"], mid["averages"]["health"], mid["economics"]["profit"]))

        reward = await gen.asend("Submitted a tend-and-tap schedule.")
        anchors = tuple(round(a, 2) for a in _sim["reward_anchors"])
        print("reward anchors (loss_floor/ceil):", anchors)
        print("episode reward (best submitted):", round(reward, 4))

    asyncio.run(_smoke())
