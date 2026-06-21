"""Tasks for the rubber-farm environment.

Run locally:   hud eval tasks.py claude
Sync remote:   hud sync tasks <taskset-name>

Each task is the same 30-year management problem on a different randomly
generated plantation (a different ``seed`` -> different tree genetics, ages, and
weather/price trends). Reward is the farm's final profit, normalised so a
sensible baseline operator scores ~1.0.
"""

# env is re-exported so `hud eval tasks.py` can resolve the Environment.
from env import env, rubber_farm  # noqa: F401

# Generous step budget: managing ~30 years a year at a time takes many turns.
_AGENT_CONFIG = {"max_steps": 80}

# A spread of scenarios (different plantations) on the default 6x6 / 30-year farm.
tasks = []
for _seed in range(6):
    _t = rubber_farm(seed=_seed)
    _t.slug = f"rubber-farm-seed-{_seed}"
    _t.agent_config = _AGENT_CONFIG
    tasks.append(_t)

# A couple of variants to widen the difficulty/shape distribution.
_small = rubber_farm(seed=0, rows=3, cols=3)
_small.slug = "rubber-farm-3x3"
_small.agent_config = _AGENT_CONFIG
tasks.append(_small)

_short = rubber_farm(seed=1, duration_years=15)
_short.slug = "rubber-farm-15yr"
_short.agent_config = _AGENT_CONFIG
tasks.append(_short)
