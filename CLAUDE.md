# Bellman Capital — Context for Claude Code

## What this project is
Academic Deep Reinforcement Learning project that builds a DQN agent to allocate capital across three risky assets and cash using hourly OHLCV data. The agent is trained via walk-forward validation across 5 temporal folds and evaluated on Sortino ratio. The only file the team submits is `agent.py`; all infrastructure in `src/` is read-only.

## Key design decisions (already implemented)

- **Discrete action menu (10 portfolios):** `agent.py, _WEIGHT_MENU, lines 29–40`. Weights `[a0, a1, a2, cash]` sum to 1. Actions 0/1/4 are pinned to HoldCash/HoldAsset0/EqualWeight to match `src/baselines.py` contracts.
- **Observation vector (22-dim):** `agent.py, TradingEnv._obs(), lines 84–93`. 18 scaled market features (6 per asset: log_ret, vol_21, mom_20, atr_14, vol_ratio, taker_buy_ratio) + 4 current portfolio weights.
- **Reward R3:** `agent.py, TradingEnv._reward(), lines 106–121`. `log_ret − 0.1×turnover − 0.5×drawdown`. LAMBDA=0.1, MU=0.5 as class constants. R1 and R2 commented out above for easy swapping.
- **Turnover capture:** `agent.py, _weights_from_action(), lines 98–103`. Captured here because `BaseTradingEnv.step()` calls this method while `self._weights` still holds old weights.
- **Terminal step clamp:** `agent.py, _obs(), line 88`. `t = min(self._t, len(self.data)-1)` prevents IndexError when `step()` increments `_t` past the last row.
- **Double DQN:** `agent.py, Agent.update(), lines 218–222`. Online net selects action, target net evaluates it. Target synced every 1 000 steps.
- **Lookahead barrier:** `agent.py, train_fold(), line 350`. `scaler.fit()` only on `train_data`; eval uses the same scaler without re-fitting.
- **Persistence:** checkpoints saved every 10 000 steps to `models/checkpoints/fold_{id}_step_N.pt`; loss logged every 500 steps to `models/metrics/fold_{id}_loss.csv`; eval metrics to `models/metrics/fold_{id}_metrics.csv`. Resume is automatic if `fold_{id}_latest.pt` exists.

## What is already built

| File | What it does |
|------|-------------|
| `agent.py` (445 lines) | TradingEnv, QNetwork, Agent (Double DQN), train_fold(), _run_episode(), persistence |
| `GUIA_EQUIPO.md` | Design document: state/action/reward/algorithm decisions with financial justification |
| `IMPLEMENTACION.md` | Plain-language technical explanation in Spanish, 10 sections including troubleshooting |

## Current status

**Done:** all 26 tests in `tests/test_submission.py` pass. Full training pipeline functional end-to-end. Persistence (checkpoints, loss CSV, metrics CSV) implemented and documented. IMPLEMENTACION.md complete through Section 10.

**Pending:** run all 5 folds and compare agent Sortino vs EqualWeight baseline per fold. Tune LAMBDA/MU if agent parks in cash or churns.

**Next action for the team:** run `uv run python agent.py` to verify smoke test passes, then run all 5 folds and fill in README.md Sections 0, 9, 10, 11 with actual results.

## Rules for every session

- Do not run training
- Do not modify `src/` files
- Do not execute processes longer than 5 seconds
- Reference file and line numbers when modifying `agent.py`
- If asked to verify something, read the file — do not run it
- All 26 tests must pass after every change to `agent.py`
- Never call `scaler.fit()` inside `TradingEnv` or on eval data
