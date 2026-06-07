# =============================================================================
# YOUR FILE — this is the only file you submit.
# Implement TradingEnv and Agent below. Do not modify anything in src/.
# =============================================================================

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

import os
import csv
from datetime import datetime

from src.env import BaseTradingEnv
from src.base import BaseAgent
from src.data import load_prices, split, build_features
from src.metrics import compute_metrics


# ── Environment ───────────────────────────────────────────────────────────────

# Discrete action menu — 10 portfolios (GUIA_EQUIPO.md Section 3).
# Constraints enforced by BaseTradingEnv.step():
#   weights.sum() == 1, w[:3] in [-1, 1], w[3] >= 0
_WEIGHT_MENU = np.array([
    [ 0.00,  0.00,  0.00,  1.00],  # 0: All Cash
    [ 1.00,  0.00,  0.00,  0.00],  # 1: Long A0
    [ 0.00,  1.00,  0.00,  0.00],  # 2: Long A1
    [ 0.00,  0.00,  1.00,  0.00],  # 3: Long A2
    [ 1/3,   1/3,   1/3,   0.00],  # 4: Equal Weight (required by EqualWeight baseline)
    [ 0.25,  0.25,  0.25,  0.25],  # 5: Conservative (25% each)
    [-0.50,  0.50,  0.50,  0.50],  # 6: Short A0, long A1+A2
    [ 0.50, -0.50,  0.50,  0.50],  # 7: Short A1, long A0+A2
    [ 0.50,  0.50, -0.50,  0.50],  # 8: Short A2, long A0+A1
    [ 0.00,  0.50,  0.50,  0.00],  # 9: Long A1+A2 only
], dtype=np.float32)

N_ACTIONS = len(_WEIGHT_MENU)


class TradingEnv(BaseTradingEnv):

    # Reward penalty coefficients (GUIA_EQUIPO.md Section 4, R3).
    LAMBDA = 0.1   # turnover penalty: discourages excessive rebalancing
    MU     = 0.5   # drawdown penalty: discourages large peak-to-trough losses

    def __init__(self, prices, features, scaler,
                 transaction_cost_bps=10.0, initial_cash=10_000.0):
        super().__init__(prices, transaction_cost_bps, initial_cash)

        # features: pre-scaled DataFrame from build_features(train, fit=True).
        # scaler:   stored for reference only — fit() was called by the caller.
        self.features = features
        self.scaler   = scaler

        # rolling(21) in build_features() makes first valid feature row = data.index[21].
        self._lookback = 21

        # Mirror the initial all-cash position set by BaseTradingEnv.reset(), so
        # _weights_from_action() can compute turnover even before reset() is called.
        self._weights = np.array([0., 0., 0., 1.], dtype=np.float32)

        # Reward state — reset() reinitialises both on every episode.
        self._peak_value    = float(initial_cash)
        self._last_turnover = 0.0

        self.action_space = spaces.Discrete(N_ACTIONS)

        # 18 scaled features (6 per asset × 3 assets) + 4 current portfolio weights.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._peak_value    = float(self.initial_cash)
        self._last_turnover = 0.0
        return obs, info

    def _obs(self) -> np.ndarray:
        # BaseTradingEnv.step() increments _t THEN calls _obs(), so at the terminal
        # step _t == len(self.data) which is out of bounds. Clamp to last valid index;
        # the terminal obs is never used in the Double DQN target (done flag zeros it).
        t         = min(self._t, len(self.data) - 1)
        timestamp = self.data.index[t]
        feat = self.features.loc[timestamp].values.astype(np.float32)  # (18,)
        obs  = np.concatenate([feat, self._weights])                    # (22,)
        assert not np.isnan(obs).any(), (
            f"NaN in obs at t={t} ({timestamp}): "
            f"feat_nan={np.isnan(feat).sum()}, weight_nan={np.isnan(self._weights).sum()}"
        )
        return obs

    def _weights_from_action(self, action: int) -> np.ndarray:
        w = _WEIGHT_MENU[action].copy()
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"action {action} weights sum to {w.sum()}"
        # Capture turnover here: base class step() calls us while self._weights
        # still holds the OLD weights, before it overwrites them.
        self._last_turnover = float(np.abs(w - self._weights).sum())
        return w

    def _reward(self, prev_value: float, curr_value: float) -> float:
        # Track running peak for drawdown calculation.
        self._peak_value = max(self._peak_value, curr_value)

        log_ret  = float(np.log(curr_value / (prev_value + 1e-8) + 1e-8))
        turnover = self._last_turnover
        drawdown = max(0.0, (self._peak_value - curr_value) / (self._peak_value + 1e-8))

        # R1: pure log-return — agent discovers churn exploit (trades every step)
        # return log_ret

        # R2: log-return minus turnover penalty — agent may park permanently in cash
        # return log_ret - self.LAMBDA * turnover

        # R3: full reward — active formulation
        return log_ret - self.LAMBDA * turnover - self.MU * drawdown


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Two-hidden-layer MLP: obs_dim → 256 → 256 → n_actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent(BaseAgent):

    # Hyperparameters — values match configs/default.yaml exactly.
    HIDDEN_DIMS         = (256, 256)
    LR                  = 1e-4
    GAMMA               = 0.99
    EPSILON_START       = 1.0
    EPSILON_END         = 0.05
    EPSILON_DECAY_STEPS = 50_000
    BATCH_SIZE          = 64
    BUFFER_SIZE         = 100_000
    TARGET_UPDATE_FREQ  = 1_000
    TRAIN_STEPS         = 200_000

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__(obs_dim, n_actions)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(obs_dim, n_actions, self.HIDDEN_DIMS).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, self.HIDDEN_DIMS).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.LR)
        self.buffer    = deque(maxlen=self.BUFFER_SIZE)
        self._step     = 0   # incremented by the training loop

    @property
    def epsilon(self) -> float:
        """Current exploration rate: linearly decayed from EPSILON_START to EPSILON_END."""
        frac = min(self._step / self.EPSILON_DECAY_STEPS, 1.0)
        return self.EPSILON_START + frac * (self.EPSILON_END - self.EPSILON_START)

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action. Default epsilon=0 gives greedy (eval) behaviour."""
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online_net(obs_t).argmax(dim=1).item())

    def store(self, obs: np.ndarray, action: int, reward: float,
              next_obs: np.ndarray, done: bool) -> None:
        """Push one transition onto the replay buffer."""
        self.buffer.append((obs, int(action), float(reward), next_obs, float(done)))

    def update(self, batch: list) -> float:
        """
        Double DQN update on a pre-sampled list of transitions.

        Double DQN (van Hasselt et al., 2016):
          - online_net selects the greedy next action
          - target_net evaluates that action's Q-value
        This decoupling reduces the maximisation bias of standard DQN.

        Returns the scalar MSE loss.
        """
        obs_b, act_b, rew_b, nobs_b, done_b = zip(*batch)

        obs_t  = torch.tensor(np.array(obs_b),  dtype=torch.float32).to(self.device)
        act_t  = torch.tensor(act_b,            dtype=torch.long).to(self.device)
        rew_t  = torch.tensor(rew_b,            dtype=torch.float32).to(self.device)
        nobs_t = torch.tensor(np.array(nobs_b), dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done_b,           dtype=torch.float32).to(self.device)

        # Q(s, a) for the actions actually taken
        q_curr = self.online_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Online net picks action; target net scores it (Double DQN)
            next_act = self.online_net(nobs_t).argmax(dim=1, keepdim=True)
            q_next   = self.target_net(nobs_t).gather(1, next_act).squeeze(1)
            target   = rew_t + self.GAMMA * q_next * (1.0 - done_t)

        loss = nn.functional.mse_loss(q_curr, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        """Hard-copy online → target network (called every TARGET_UPDATE_FREQ steps)."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train(self, env, n_steps: int = 200_000, fold_id=None) -> None:
        """
        DQN training loop with epsilon-greedy exploration and Double DQN updates.
        Resets env automatically at each episode boundary.

        fold_id: string label used for checkpoint/log file names (e.g. "20231231").
                 Pass None to skip all persistence (default — backward compatible).
        """
        # ── Persistence setup ──────────────────────────────────────────────
        if fold_id is not None:
            ckpt_dir      = "models/checkpoints"
            metrics_dir   = "models/metrics"
            os.makedirs(ckpt_dir,    exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)
            latest_path   = f"{ckpt_dir}/fold_{fold_id}_latest.pt"
            loss_csv_path = f"{metrics_dir}/fold_{fold_id}_loss.csv"

            # ── Resume if a checkpoint exists ─────────────────────────────
            if os.path.exists(latest_path):
                ckpt = torch.load(latest_path, map_location=self.device)
                self.online_net.load_state_dict(ckpt["online_net"])
                self.target_net.load_state_dict(ckpt["target_net"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self._step = ckpt["step"]
                print(f"    Resumed from checkpoint: step={self._step}")
            else:
                with open(loss_csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(["step", "loss", "epsilon", "buffer_size"])

        # ── Training loop ──────────────────────────────────────────────────
        obs, _ = env.reset()

        for global_step in range(self._step + 1, n_steps + 1):
            eps    = self.epsilon
            action = self.act(obs, epsilon=eps)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.store(obs, action, reward, next_obs, done)
            self._step += 1

            loss = None
            if len(self.buffer) >= self.BATCH_SIZE:
                batch = random.sample(self.buffer, self.BATCH_SIZE)
                loss  = self.update(batch)

            if self._step % self.TARGET_UPDATE_FREQ == 0:
                self.sync_target()

            obs = env.reset()[0] if done else next_obs

            # ── Loss curve: one row every 500 steps ───────────────────────
            if fold_id is not None and loss is not None and global_step % 500 == 0:
                with open(loss_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([global_step, loss, eps, len(self.buffer)])

            # ── Checkpoint every 10 000 steps ─────────────────────────────
            if fold_id is not None and global_step % 10_000 == 0:
                ckpt = {
                    "online_net": self.online_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "step":       self._step,
                }
                torch.save(ckpt, f"{ckpt_dir}/fold_{fold_id}_step_{global_step}.pt")
                torch.save(ckpt, f"{ckpt_dir}/fold_{fold_id}_latest.pt")

            if global_step % 20_000 == 0 or global_step == n_steps:
                print(f"    step {global_step:>7}/{n_steps}  "
                      f"eps={eps:.3f}  buf={len(self.buffer)}")


# ── Walk-forward training ─────────────────────────────────────────────────────

def _run_episode(agent: Agent, env: TradingEnv):
    """Greedy episode. Returns (portfolio_values array, total_reward float)."""
    values       = [env.initial_cash]
    total_reward = 0.0
    obs, _       = env.reset()
    done         = False
    while not done:
        action = agent.act(obs, epsilon=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(info["portfolio_value"])
        total_reward += reward
        done = terminated or truncated
    return np.array(values, dtype=np.float64), total_reward


def train_fold(fold_config: dict, fold_id: str = None) -> dict:
    """
    Train and evaluate one walk-forward fold. No hyperparameter changes between folds.

    fold_config: {"train_end": "YYYY-MM-DD", "eval_end": "YYYY-MM-DD"}
    fold_id:     label for checkpoint/log files; derived from train_end if not given.
    Returns: compute_metrics() dict evaluated on the held-out eval period.
    """
    train_end = fold_config["train_end"]
    eval_end  = fold_config["eval_end"]
    if fold_id is None:
        fold_id = train_end.replace("-", "")   # e.g. "20231231"
    print(f"[train_fold]  train_end={train_end}  eval_end={eval_end}  fold_id={fold_id}")

    # 1. Load raw prices (1h candles)
    data = load_prices("1h")

    # 2. Temporal split — eval starts strictly after train_end (src/data.py:split)
    train_data, eval_data = split(data, train_end, eval_end)
    print(f"  train rows : {len(train_data):>6}  eval rows : {len(eval_data):>6}")

    # 3. LOOKAHEAD BOUNDARY — scaler.fit() on TRAIN DATA ONLY.
    #    Calling fit() on eval data leaks future statistics: disqualification risk.
    train_features, scaler = build_features(train_data, fit=True)  # <-- fit here only

    # 4. Transform eval with the train-fitted scaler (no re-fit)
    eval_features, _ = build_features(eval_data, scaler=scaler)

    # 5. Build environments — each uses its own feature slice and the same scaler
    train_env = TradingEnv(train_data, train_features, scaler)
    eval_env  = TradingEnv(eval_data,  eval_features,  scaler)

    # 6. Train (SEED=42 fixed inside Agent.__init__; no tuning on eval split)
    print(f"  training Agent for {Agent.TRAIN_STEPS:,} steps...")
    agent = Agent(obs_dim=22, n_actions=N_ACTIONS)
    agent.train(train_env, n_steps=Agent.TRAIN_STEPS, fold_id=fold_id)

    # 7. Evaluate on the held-out eval period (greedy, epsilon=0)
    print("  evaluating on eval split...")
    portfolio_values, episode_reward = _run_episode(agent, eval_env)

    metrics = compute_metrics(portfolio_values)
    print(f"  sortino={metrics['sortino']:>8.4f}  "
          f"cum_ret={metrics['cum_ret']:>8.4f}  "
          f"max_dd={metrics['max_dd']:>8.4f}")

    # ── Persistence: metrics CSV ───────────────────────────────────────────
    metrics_dir = "models/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv = f"{metrics_dir}/fold_{fold_id}_metrics.csv"
    write_header = not os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "episode_reward", "sortino", "cum_ret",
                        "max_dd", "epsilon", "timestamp"])
        w.writerow([
            Agent.TRAIN_STEPS,
            round(episode_reward, 6),
            round(metrics["sortino"], 6),
            round(metrics["cum_ret"], 6),
            round(metrics["max_dd"], 6),
            Agent.EPSILON_END,
            datetime.utcnow().isoformat(),
        ])

    return metrics


# ── Sanity test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data             = load_prices("1h")
    features, scaler = build_features(data, fit=True)   # fit on full data (demo only)
    env              = TradingEnv(data, features, scaler)

    obs, _ = env.reset()
    assert not np.isnan(obs).any(), "NaN in initial observation"
    assert obs.shape == (22,), f"Expected obs shape (22,), got {obs.shape}"

    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        assert not np.isnan(obs).any(),              f"NaN in obs at step {i}"
        assert not np.isnan(reward),                 f"NaN in reward at step {i}"
        assert not np.isnan(info["portfolio_value"]), f"NaN in portfolio_value at step {i}"

    print("Sanity test passed: reset() + step(0) x5, no NaN.")
    print(f"  obs_shape     : {obs.shape}  OK (22,)")
    print(f"  action_space  : {env.action_space}")
    print(f"  _lookback     : {env._lookback}")
    print(f"  portfolio_val : {info['portfolio_value']:.4f}")
    for a in range(N_ACTIONS):
        w = env._weights_from_action(a)
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"action {a} weights don't sum to 1"
    print(f"  All {N_ACTIONS} actions have weights summing to 1. OK.")

    agent = Agent(obs_dim=22, n_actions=10)
    print(f"\nAgent sanity: act(zeros, epsilon=1.0) = {agent.act(np.zeros(22), epsilon=1.0)}")
    print(f"  device      : {agent.device}")
    print(f"  buffer size : {len(agent.buffer)} / {agent.BUFFER_SIZE}")
    print(f"  epsilon     : {agent.epsilon:.4f}  (step=0, decays to {agent.EPSILON_END})")

    print("\nReward values — 10 steps with action=4 (equal weight):")
    print(f"  {'step':>4}  {'reward':>10}  {'turnover':>9}  {'value':>12}  {'peak':>12}")
    env.reset()
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(4)
        print(f"  {i+1:>4}  {reward:>10.6f}  {info['turnover']:>9.4f}"
              f"  {info['portfolio_value']:>12.4f}  {env._peak_value:>12.4f}")
        assert not np.isnan(reward), f"NaN reward at step {i+1}"
        if terminated or truncated:
            break

    print("\n--- Fold 5 smoke test (train: start-2023, eval: 2024-2025) ---")
    fold5_config = {"train_end": "2023-12-31", "eval_end": "2025-12-31"}
    metrics = train_fold(fold5_config)
    print(f"  cum_ret  : {metrics['cum_ret']:.4f}")
    print(f"  ann_ret  : {metrics['ann_ret']:.4f}")
    print(f"  ann_vol  : {metrics['ann_vol']:.4f}")
    print(f"  sortino  : {metrics['sortino']:.4f}  (primary metric)")
    print(f"  max_dd   : {metrics['max_dd']:.4f}")
    print("Fold 5 smoke test complete.")
