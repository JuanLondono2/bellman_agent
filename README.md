# BELLMAN CAPITAL — Autonomous Portfolio Allocation

## Teams

Work in groups of up to 4. One submission per team. Every team member is expected to understand and be able to defend the full implementation.

## Overview

Markets move $5 trillion a day. Every price is an argument between a buyer and a seller, and one of them is wrong. This workshop asks you to build a reinforcement learning agent that systematically decides how to allocate capital across a portfolio of assets, evaluated on real historical market data.

## Market Primer

A **return** is the percentage change in price between two timesteps. A **portfolio** is an allocation of capital across assets: holding 50% in Asset A and 50% in cash with a 20% rise in Asset A yields a 10% portfolio return.

**Transaction costs** are fees paid on every rebalance. At 10 basis points per trade, an agent that trades excessively destroys its own returns. **Drawdown** is the peak-to-trough loss before recovery; a 50% drawdown requires a 100% gain just to break even.

## Project Objectives

Build an agent that allocates capital across four assets (three risky, one cash) to maximize risk-adjusted return. The agent trains on historical data and is evaluated on a held-out period it cannot observe during development.

**Non-negotiable constraints:**
1. No lookahead. Features at time `t` may only use data from `t` and earlier.
2. Transaction costs of at least 10 basis points per trade must be modeled.
3. Results must be fully reproducible from the submitted codebase.

Grading rewards rigorous methodology, not high returns. An agent that fails on the held-out period but demonstrates disciplined evaluation can earn full marks.

## 0. Team

- **Agent codename:**
- **Researchers:**
- **Inception date:**
- **Thesis:** What does your agent believe about these markets, and how does that belief shape its design?

## 1. Problem Formulation

### State Space

The agent's observation at each timestep is its state. Define it carefully: what information is available at time `t`, and what form does it take?

For every feature in your state, justify its inclusion in one sentence. Then answer:
1. What is in your observation at time `t`? List every component.
2. Raw prices are not Markov: past volatility and momentum matter. How does your observation account for this?
3. How much history do you include, and why?

### Action Space

The action is the allocation decision the agent makes at each step.

**Short positions are allowed.** Risky asset weights can be negative (e.g. -0.5 means you are short 50% of that asset: you profit if it falls, lose if it rises). Cash weight must remain non-negative. All weights must sum to 1, and each risky asset weight is capped at [-1.0, 1.0] by the framework — no leverage. Note that this model does not charge a funding rate on short positions, which is a simplification of how shorting works in practice.

| Design | Tradeoff |
|---|---|
| Discrete menu of named allocations | Pre-define ~10–15 fixed portfolios; agent selects one per step. Compatible with DQN. Interpretable. Limited to predefined combinations. Can include short positions in the menu. |
| Continuous portfolio weights | Agent outputs weights directly. More expressive, but incompatible with DQN; requires policy gradient methods (PPO, SAC), which are significantly harder to tune. |
| Adjustment actions | Agent shifts current weights incrementally. Natural for gradual rebalancing; state must include current holdings. |

Answer:
1. What is your action space? If discrete, list every action and its economic interpretation.
2. Why did you choose this representation?
3. What does this choice prevent your agent from doing?

## 2. Data and Exploratory Analysis

```
data/raw/prices_{interval}.parquet    # OHLCV + taker buy ratio, columns: asset_0_*, asset_1_*, asset_2_*, cash
```

Data is provided in three candle intervals: `15m`, `30m`, `1h`. Asset names are anonymized; identities are revealed at the final evaluation. `src/data.py` provides `load_prices(interval)`, `split()`, and `build_features()`. Temporal splits are fixed in `configs/default.yaml` and must not be altered.

**Before writing any code, run the EDA notebook:**

```bash
uv run jupyter notebook notebooks/eda.ipynb
```

It walks through price history, return distributions, volatility regimes, correlations, and the pre-computed feature set. Your state space and reward function decisions should be grounded in what you observe there.

**Lookahead.** Rows are indexed by candle close time, so all values in row `T` are observable at `T`. Lookahead cannot occur on raw data, but can appear in feature engineering. Common violations: fitting a scaler before the train/eval split, or rolling operations that borrow from future rows. `build_features()` is lookahead-safe; audit any custom features.

## 3. Environment Design

```python
obs, _ = env.reset()
while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

The base class in `src/env.py` handles portfolio valuation and transaction costs. Subclass it in `agent.py` and implement three methods:

```python
def _obs(self) -> np.ndarray           # observation vector at each step
def _weights_from_action(self, action)  # action index to portfolio weights (sum to 1, all >= 0)
def _reward(self, prev, curr)           # scalar reward signal
```

## 4. Reward Design

Document the full iteration. State each reward formulation, train, and report observed behavior. Did the agent park in cash? Trade excessively? For the final formulation, compare at least two alternatives: log return, differential Sharpe, drawdown-penalized, turnover-penalized. Identify one plausible exploit of your reward and report whether the agent discovered it.

## 5. Algorithm

DQN with a discrete action space is the standard starting point. Reasonable extensions: Double DQN, Dueling DQN, Prioritized Experience Replay. Policy gradient methods (PPO, SAC) are permitted if justified by the problem formulation. Report all hyperparameters. Any tuning on the held-out evaluation window is grounds for disqualification.

## 6. Baselines

All baselines are implemented in `src/baselines.py` and must be evaluated under identical conditions to your agent.

| Baseline | Role |
|---|---|
| Random policy | Sanity floor |
| Hold cash | Passive benchmark |
| Hold Asset 0 | Single-asset benchmark |
| Equal weight, rebalanced | Diversification benchmark |
| SMA crossover | Trend-following heuristic |

## 7. Training Protocol

Report: total environment steps, wall-clock time, hardware, logged metrics, and stopping criterion.

## 8. Evaluation

Split your data into train and test using `split()` from `src/data.py`. Train your agent on the training period, evaluate on the held-out test period without any further tuning. The final held-out window is evaluated once, with no parameter changes afterward.

Report against all baselines: cumulative return, Sortino ratio (primary metric — penalizes only downside volatility), max drawdown, and total fees paid. Run at 0 bps and 10 bps to show your agent is robust to transaction costs.

## 9. Results

Equity curves against all baselines with seed spread visible. Per-window metrics table. Allocation plot over time with economic interpretation. At minimum one figure documenting failure or anomalous behavior.

## 10. Discussion

One paragraph per item on how it manifested in your project and how you addressed it:

- Reward design and reward hacking
- Sample efficiency: market regimes are long and observations are not independent
- Train-to-deploy distribution shift
- Non-stationarity and regime change
- Long-horizon credit assignment

## 11. Reflection

- Three results that surprised you
- Two methodological changes you would make given more time
- One aspect of your agent's behavior you cannot explain
- The most significant gap between DRL theory and this applied problem

## 12. Submission

Submit a single file: `agent.py`. It must define `TradingEnv` (your environment) and `Agent` (your model). Class names must not be changed.

```bash
uv run pytest tests/test_submission.py -v
```

All tests must pass. A submission with failing tests will not be graded.

## Rubric

| Component | Weight |
|---|---|
| Problem formulation: state and action design (Section 1) | 20% |
| Environment implementation (Section 3) | 15% |
| Reward design with documented iteration (Section 4) | 15% |
| Evaluation protocol: walk-forward, ablation, metrics (Section 8) | 20% |
| Baseline comparison (Section 6) | 10% |
| Discussion and reflection (Sections 10–11) | 15% |
| Submission passes all tests | 5% |

The held-out evaluation result is not on this rubric. Rigorous methodology on a failing agent outscores a lucky result with no methodology.
