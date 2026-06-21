# Rubber Farm — a long-horizon planning RL environment

**This environment trains and evaluates an agent's ability to run an autonomous
business: plan a long sequence of actions, over a long horizon, toward a goal.**
The agent operates an autonomous farm — writing a 30-year operating schedule for
a rubber plantation, submitting it to a fast simulator, reading the resulting
profit, and iterating to find the best schedule it can. It is a **high degree of
freedom** task: thousands of days × many possible actions per day, the kind of
open-ended control space you'd face when **controlling agent swarms to operate
the farm**. The skill under test is *long-horizon offline optimization* — reason
about delayed consequences, search over candidate plans, and converge on a
profitable strategy.

The broader thesis: autonomous farming is a concrete proving ground for
**autonomous business** — a self-contained operation with revenue, costs,
capital that ages, and decisions whose payoff lands years later.

---

## The task

You manage a grid of rubber trees for ~30 years. Trees mature (~6 years), can be
*tapped* for latex (the only revenue), need water and N/P/K nutrients to stay
healthy, and age out around 28–34 years. You don't step through time live —
instead you submit a **schedule of actions** and the simulator plays out all 30
years instantly. Maximize **profit = rubber revenue − spending on water &
fertilizer**.

### Plan format (day-indexed)

A plan is a schedule keyed by day. Submit it either as a JSON **object**
(`day → actions`, the practical form) or a **list** (index = day):

```json
{
  "0":   [{"tool": "tap", "target": "mature"},
          {"tool": "fertilize", "n": 0.2, "p": 0.15, "k": 0.15, "target": "all"}],
  "365": [{"tool": "tap", "target": "mature"},
          {"tool": "water", "gallons": 5, "target": "all"}]
}
```

Each day holds a list of actions, applied in order. Action tools:

| tool | fields | notes |
|------|--------|-------|
| `tap` | `target` | start tapping (persists across days) |
| `untap` | `target` | stop tapping |
| `water` | `gallons`, `target` | one-off irrigation |
| `fertilize` | `n`, `p`, `k`, `target` | one-off feed; each 0..1 per tree |

`target` ∈ `all | mature | immature | "row,col"`. Tapping is a standing setting;
watering/fertilizing apply only on the day they appear; omitted days just pass.

### Tools

- **`submit_plan(plan)`** — the only scoring tool. Simulates the full schedule
  and returns profit, reward, and a year-by-year timeline. Call it repeatedly;
  your score is your **best** submission.
- `observe()` — starting farm state + the full per-year price forecast.
- `render()` — ASCII map of the starting farm.
- `observe_at(day, plan)` — inspect a plan's physical state at a given day (tree
  maturity, soil, health). Does **not** report profit and does **not** score.

## Reward

Profit is normalized to `0..1`, keyed on rubber harvested *and* profitability:

| outcome | reward |
|---------|--------|
| no submission, or a plan that harvests no rubber | **0.0** |
| harvests rubber but loses money | **0.1 → 0.5** (rises toward break-even) |
| break-even (profit 0, with rubber) | **0.5** |
| profitable | **0.5 → 1.0** (1.0 at the best simple baseline) |

So harvesting earns partial credit, profitability earns strictly more, and a
full **1.0 requires being genuinely profitable**.

## Simulation notes

- **Forgiving, bounded spending.** You only pay for water/nutrients the soil
  actually absorbs (each caps at 1.0 per tree), and a fixed **annual budget**
  caps yearly spend (over-orders are scaled down, never overspent). This removes
  the "over-fertilize into bankruptcy" trap and keeps the optimization tractable.
- **Tapping dynamics.** Continuous tapping wears the panel down (lower yield);
  resting heals it — nudging toward sustainable, alternate-style tapping.
- **Deterministic per seed.** Weather and tree genetics are seeded, so a plan's
  outcome is reproducible and submissions are directly comparable.

## Running it

```bash
# Evaluate an agent locally against the tasks
hud eval tasks.py claude --task-ids rubber-farm-seed-0 -v

# No-model smoke test (drives the tools directly, prints a reward)
python env.py

# Explore the simulator / baseline strategies
python test_run_simulation.py

# Deploy to the HUD platform, then sync the taskset
hud build && hud push
hud sync tasks <taskset-name>
```

## Repo layout

| path | what it is |
|------|------------|
| `env.py` | HUD environment: MCP tools, the plan-submission task, reward wiring |
| `tasks.py` | Task variants (seeds, grid sizes, horizons) for `hud eval`/`hud sync` |
| `Farm/` | The simulation package (pure Python, no HUD dependency) |
| `Farm/objects.py` | The engine: `RubberTree`, `Farm` (growth, tapping, soil, economics) |
| `Farm/spec.py` | `FarmSpec` / `default_spec` — a serializable scenario description |
| `Farm/actions.py` | Action data types (`Water`, `Fertilize`, `Tap`) |
| `Farm/strategies.py` | Baseline policies + reward anchors and `scale_reward` |
| `Farm/textsim.py` | A tiny interactive/`--demo` text front-end for the sim |
| `test_run_simulation.py` | A runnable walkthrough comparing strategies |
