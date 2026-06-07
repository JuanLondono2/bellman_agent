# Bellman Capital — Context for Claude Code

## What this project is
Academic Deep Reinforcement Learning project that builds a DQN agent to allocate capital across three risky assets and cash using hourly OHLCV data. The agent is trained via walk-forward validation across 5 temporal folds and evaluated on Sortino ratio. The only file the team submits is `agent.py`; all infrastructure in `src/` is read-only.

## Key design decisions (already implemented)

- **Discrete action menu (10 portfolios):** `agent.py, _WEIGHT_MENU, lines 29–40`. Weights `[a0, a1, a2, cash]` sum to 1. Actions 0/1/4 are pinned to HoldCash/HoldAsset0/EqualWeight to match `src/baselines.py` contracts.
- **Observation vector (22-dim):** `agent.py, TradingEnv._obs(), lines 84–93`. 18 scaled market features (6 per asset: log_ret, vol_21, mom_20, atr_14, vol_ratio, taker_buy_ratio) + 4 current portfolio weights.
- **Reward R3:** `agent.py, TradingEnv._reward(), lines 107–122`. `log_ret − 0.01×turnover − 0.05×downside` where `downside = max(0, −log_ret)` (differential Sortino). LAMBDA=0.01, MU=0.05 as class constants. R1 and R2 commented out above for easy swapping.
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

**V1 (2026-06-07):** sortino=−5.49, cum_ret=−0.9999, max_dd=−0.9999 (fold 5 only). Diagnosed in `ENTRENAMIENTO_V1.md`. Root cause: drawdown penalty created a structural trap where switching to cash cost more than staying in a losing position.

**V2 (2026-06-07):** sortino=−3.67, cum_ret=−1.0, max_dd=−1.0 (fold 5 only). Diagnosed in `ENTRENAMIENTO_V2.md`. Fixes applied improved Sortino but structural trap persisted: `MU × drawdown` is identical for all actions, so it cannot distinguish between staying and escaping a losing position.

**V3 (2026-06-07):** Full 5-fold run with differential Sortino reward (Fix 5). Diagnosed in `ENTRENAMIENTO_V3.md`.

| Fold | train_end | eval_end | sortino | cum_ret | max_dd |
|------|-----------|----------|---------|---------|--------|
| 1 | 2019-12-31 | 2020-12-31 | **+1.8580** | **+127.8567** | **−0.2333** |
| 2 | 2020-12-31 | 2021-12-31 | −1.5286 | −0.9978 | −0.9985 |
| 3 | 2021-12-31 | 2022-12-31 | −3.5462 | −1.0000 | −1.0000 |
| 4 | 2022-12-31 | 2023-12-31 | −4.9175 | −0.9998 | −0.9998 |
| 5 | 2023-12-31 | 2025-12-31 | −3.8967 | −1.0000 | −1.0000 |

**Feature engineering limit (2026-06-07):** Con 200k pasos fijos, no existe combinación de features que compense la insuficiencia de episodios en folds 2-5 ni el distribution shift; el feature engineering es la herramienta correcta solo después de resolver el presupuesto de pasos.

Root cause (V3): distribution shift between training and evaluation regimes. Fold 1 succeeded because eval period (2020) was a strongly trending bull market aligned with the training regime. Folds 2-5 failed due to: (a) market regime mismatch in eval, and (b) insufficient training episodes per fold — larger training sets (26k–52k rows) complete only 3.8–7.6 full episodes in 200k steps vs 11.5 episodes for fold 1. Differential Sortino is the correct reward but did not solve these two root causes.

**Fixes applied to `agent.py` for V3:**
- Fix 5 (`agent.py:107–122`): `_reward()` replaced with differential Sortino — `downside = max(0, −log_ret)` (per-step, action-dependent) instead of `(peak − value) / peak` (cumulative, action-independent). Breaks the structural trap proved in V2 analysis.

**Fixes applied to `agent.py` for V4:**
- Fix 6 (`agent.py:159`): `TRAIN_STEPS = 600_000` (was 200_000) — fold 5 now completes ~11 episodes instead of ~3.8, matching fold 1's convergence budget in V3.
- Fix 7 (`agent.py:157`): `BUFFER_SIZE = 50_000` (was 100_000) — prevents a single large-fold episode from dominating 52% of the replay buffer, improving temporal diversity of sampled transitions.

**Data interval currently in use:** `load_prices("1h")` (`agent.py:346`) → `data/raw/prices_1h.parquet`. Available intervals: 15m, 30m, 1h.

**Periodicity analysis conclusion (from `ENTRENAMIENTO_V3.md` Section 8):** Do NOT change the interval for V4. With 600k training steps, finer intervals (30m, 15m) still reduce episodes per fold (17 and 8.5 respectively for fold 1 vs ~34 with 1h), and the 1h interval provides a better signal/noise ratio for the rolling features (vol_21, mom_20 = ~21 and ~20 hours). The failure of folds 2-5 is caused by market regime mismatch, not data granularity.

**Pending for V4 training:**
1. Run `uv run pytest tests/test_submission.py -v` — all 26 must pass.
2. Delete V3 checkpoints, run full 5-fold training, document in `ENTRENAMIENTO_V4.md`.
3. Success criterion: sortino > 0 in ≥ 3 folds. If only fold 1 positive again, add regime-detection features to the 22-dim observation vector.

## Rules for every session

- Do not run training
- Do not modify `src/` files
- Do not execute processes longer than 5 seconds
- Reference file and line numbers when modifying `agent.py`
- If asked to verify something, read the file — do not run it
- All 26 tests must pass after every change to `agent.py`
- Never call `scaler.fit()` inside `TradingEnv` or on eval data
