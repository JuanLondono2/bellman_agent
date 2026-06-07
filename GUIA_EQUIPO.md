# GUIA_EQUIPO — Bellman Capital

> Documento de referencia interno. Toda decisión de diseño está anclada al código fuente real. **No modificar los splits ni el código de infraestructura** (`src/env.py`, `src/data.py`, `src/baselines.py`, `src/metrics.py`).

---

## 1. Project Understanding

### ¿Qué hace el proyecto?

El proyecto entrena un agente de RL para asignar capital entre tres activos de riesgo (anónimos) y cash, usando datos OHLCV históricos de velas de 1h, 30m ó 15m.

**`src/env.py`** expone `BaseTradingEnv(gym.Env)`, una clase abstracta con tres métodos que el equipo **debe** implementar en `agent.py`:

| Método | Responsabilidad |
|--------|----------------|
| `_obs(self) -> np.ndarray` | Vector de observación en el timestep actual |
| `_weights_from_action(self, action: int) -> np.ndarray` | Mapeo de índice de acción a pesos de portafolio `(4,)` |
| `_reward(self, prev_value, curr_value) -> float` | Señal de recompensa escalar |

La clase base maneja: stepping de precios (`self.prices = prices[CLOSE_COLS].values`), seguimiento de valor (`self._value`), pesos actuales (`self._weights`), y cálculo de costos de transacción. La fórmula exacta del step es:

```python
turnover = np.abs(w - self._weights).sum()
ret      = self.prices[self._t] / self.prices[self._t - 1]   # shape (4,)
self._value = self._value * np.dot(w, ret) - self._value * turnover * self.tc
```

El subclase también debe definir `self._lookback` (número de filas de calentamiento que `reset()` salta con `self._t = self._lookback`).

**`configs/default.yaml`** define el split temporal con walk-forward de 5 folds:

| Fold | Train hasta | Eval |
|------|-------------|------|
| 1 | 2019-12-31 | 2020 completo |
| 2 | 2020-12-31 | 2021 completo |
| 3 | 2021-12-31 | 2022 completo |
| 4 | 2022-12-31 | 2023 completo |
| 5 | 2023-12-31 | 2024-2025 |
| Held-out | — | hasta 2026-12-31 → **nunca tocar** |

**`src/metrics.py`** calcula con `compute_metrics(portfolio_values, freq=252)`:

| Métrica | Fórmula | Rol |
|---------|---------|-----|
| `cum_ret` | `v[-1]/v[0] - 1` | Retorno total |
| `ann_ret` | Compuesto anualizado | Retorno ajustado por tiempo |
| `ann_vol` | `std(r) * sqrt(freq)` | Riesgo total |
| `sharpe` | `ann_ret / ann_vol` | Retorno por unidad de riesgo |
| `sortino` | `ann_ret / downside_std` | **Métrica primaria** — penaliza solo volatilidad negativa |
| `max_dd` | `min((v - peak) / peak)` | Peor caída desde máximo |

> **Nota sobre `freq=252`**: el parámetro asume días de trading. Con datos de 1h, la anualización correcta sería `freq=24*365=8760`. Sin embargo, dado que `compute_metrics` es infraestructura fija, el equipo debe ser consistente y no cambiarlo; interpretar los valores absolutos con cautela.

---

## 2. State Space Design

### Datos disponibles en `prices_1h.parquet`

Shape: **(70005, 16)** — ~8 años de velas horarias (2018-01-01 a ~2026-01-01).

Columnas: `asset_{0,1,2}_{close, high, low, volume, taker_buy_ratio}` + `cash` (constante = 1.0).

### Features producidas por `build_features()` en `src/data.py`

`build_features()` genera **18 features** (6 por activo × 3 activos), luego aplica `StandardScaler`:

| Feature | Cálculo | Justificación financiera |
|---------|---------|--------------------------|
| `asset_i_log_ret` | `log(close_t / close_{t-1})` | Aproximación simétrica del retorno; estacionaria; input natural del agente de RL |
| `asset_i_vol_21` | `rolling(21).std(log_ret)` | Captura *volatility clustering* (ARCH/GARCH) — la volatilidad actual predice la futura |
| `asset_i_mom_20` | `log(close_t / close_{t-20})` | Momentum de 20 períodos; evidencia empírica robusta de persistencia de tendencias a medio plazo |
| `asset_i_atr_14` | `ATR(14) / close` | Rango verdadero normalizado; proxy del riesgo de mercado intradía sin sesgo de escala |
| `asset_i_vol_ratio` | `volume / rolling(21).mean(volume)` | Volumen relativo señala actividad inusual; picos predicen ruptura de rango (teoría de microestructura) |
| `asset_i_tbr` | Fracción de volume iniciada por compradores | Señal de presión direccional a corto plazo; TBR > 0.5 indica sesgo comprador neto |

**Riesgo de lookahead en `build_features()`:** Todas las operaciones usan `.shift(1)` o `.rolling(W)` con índices pasados — son lookahead-safe. **Riesgo real:** si el scaler se ajusta (`fit=True`) usando datos que incluyen el período de eval. Ver Sección 6.

### Vector de observación propuesto

```
obs = [
    features_t[0:18],       # 18 features escaladas del timestep actual
    self._weights[0:4],      # pesos actuales del portafolio (4 valores)
]
# Dimensión total: 22
```

**Justificación de incluir `self._weights`:** El agente necesita saber su posición actual para calcular el costo de transacción implícito de cualquier acción futura. Sin esto, no puede razonar sobre el *path dependency* del portafolio — violación del supuesto de Markov.

**Alternativa con ventana de tiempo (recomendada para capturar secuencia):** Incluir los últimos `L` timesteps de log-returns para los 3 activos (`L * 3` valores). El parámetro `lookback_window: 30` en `configs/default.yaml` sugiere `L = 30`. Esto es compatible con el baseline `SMA`, cuyo método `act(obs)` espera `obs[:lookback * 3]` estructurado como `(lookback, 3)` de retornos.

```python
# Diseño compatible con SMA baseline (src/baselines.py línea 26-31):
#   n, lookback = 3, len(obs) // 3
#   rets = obs[:lookback * n].reshape(lookback, n)
obs = np.concatenate([
    log_returns_window.flatten(),   # shape (30 * 3,) = 90
    current_weights,                # shape (4,)
])
# Dimensión total: 94
```

**Flag de riesgo — NaN en rolling features:** `build_features()` descarta con `.dropna()` las primeras ~20 filas (max(21, 20) = 21 períodos de calentamiento). Asegurarse de que `self._lookback >= 21` para evitar indexar features vacías durante `reset()`.

---

## 3. Action Space Design

### Decisión: Acción Discreta (compatible con DQN)

El método abstracto `_weights_from_action(self, action: int)` en `src/env.py:75` espera un índice entero y devuelve pesos `(4,)`. El diseño de menú discreto es el único compatible con DQN sin cambiar el algoritmo.

**Las posiciones cortas están soportadas** por la clase base: `w[:3]` puede estar en `[-1.0, 1.0]` con `cash >= 0` y `sum(w) == 1.0`.

### Menú de 10 acciones propuesto

| ID | Nombre | Pesos `[a0, a1, a2, cash]` | Interpretación económica |
|----|--------|---------------------------|--------------------------|
| 0 | All Cash | `[0, 0, 0, 1]` | Refugio: sin exposición a riesgo de mercado |
| 1 | Long A0 | `[1, 0, 0, 0]` | Apuesta concentrada en asset_0 |
| 2 | Long A1 | `[0, 1, 0, 0]` | Apuesta concentrada en asset_1 |
| 3 | Long A2 | `[0, 0, 1, 0]` | Apuesta concentrada en asset_2 |
| 4 | Equal Weight | `[1/3, 1/3, 1/3, 0]` | Diversificación máxima sin leverage |
| 5 | Conservative | `[0.25, 0.25, 0.25, 0.25]` | Diversificado con 25% cash como amortiguador |
| 6 | Short A0 | `[-0.5, 0.5, 0.5, 0.5]` | Apuesta bajista en asset_0 con cobertura long |
| 7 | Short A1 | `[0.5, -0.5, 0.5, 0.5]` | Apuesta bajista en asset_1 con cobertura long |
| 8 | Short A2 | `[0.5, 0.5, -0.5, 0.5]` | Apuesta bajista en asset_2 con cobertura long |
| 9 | Long A1+A2 | `[0, 0.5, 0.5, 0]` | Par-trading: excluye asset_0 |

> **Restricción de los baselines:** `HoldCash` retorna acción 0, `HoldAsset0` retorna acción 1, `EqualWeight` retorna acción 4. Estos mapeos están hardcodeados en `src/baselines.py:12,15,18`. El menú de acciones **debe** respetar este contrato.

**¿Qué impide este diseño?** El menú discreto no puede expresar pesos continuos arbitrarios. Un agente óptimo que quisiera, por ejemplo, 47% en asset_0 y 53% en asset_2 no puede hacerlo. Esto limita la capacidad expresiva pero es la restricción correcta para DQN.

**Soporte de posiciones cortas:** Sí, acciones 6-8 incluyen shorting. El entorno **no cobra tasa de financiamiento** por las posiciones cortas (`src/env.py:32-34`), lo que es una simplificación — en mercados reales habría un costo de borrow.

---

## 4. Reward Design

### R1: Log-retorno puro (baseline — se espera que falle)

```python
def _reward(self, prev_value: float, curr_value: float) -> float:
    return np.log(curr_value / prev_value + 1e-8)
```

**Por qué falla:** El agente maximiza retorno esperado sin penalización por riesgo ni por trading. El exploit más frecuente es **churn**: el agente aprende a hacer flip entre posiciones opuestas en cada step porque el lookback corto no le permite atribuir el costo de transacción al retorno futuro (problema de crédito diferido). También tiende a tomar posiciones concentradas sin amortiguador de drawdown.

**Cómo detectarlo:** `info["turnover"]` promedio > 1.5 por step; curvas de equity con volatilidad extrema en entrenamiento pero colapso en eval.

### R2: Log-retorno menos penalización de turnover

```python
LAMBDA = 0.1  # penalización de turnover; tunable

def _reward(self, prev_value: float, curr_value: float) -> float:
    log_ret  = np.log(curr_value / prev_value + 1e-8)
    # turnover ya calculado en step() y disponible como self._last_turnover
    penalty  = LAMBDA * self._last_turnover
    return log_ret - penalty
```

> **Nota de implementación:** `env.step()` calcula `turnover = np.abs(w - self._weights).sum()` en `src/env.py:57` *antes* de actualizar `self._weights`. El subclase puede almacenarlo como `self._last_turnover = turnover` al inicio de `_weights_from_action()` o capturarlo de otra forma.

**Turnover exacto:** suma de cambios absolutos en pesos, `|w_new - w_old|.sum()`. Para una rotación total de portafolio (0% → 100%), turnover = 2.0.

**Exploit esperado:** Con `LAMBDA` muy alto, el agente aprende a no moverse nunca (acción repetida = turnover 0). Se queda permanentemente en la acción del timestep inicial. **Detección:** acción única en > 90% de los steps.

### R3: R2 más penalización de drawdown

```python
LAMBDA = 0.1
MU     = 0.5  # penalización de drawdown

def reset(self, *, seed=None, options=None):
    obs, info = super().reset(seed=seed, options=options)
    self._peak_value = self.initial_cash   # tracking del pico
    return obs, info

def _reward(self, prev_value: float, curr_value: float) -> float:
    self._peak_value = max(self._peak_value, curr_value)
    log_ret   = np.log(curr_value / prev_value + 1e-8)
    turnover  = self._last_turnover
    drawdown  = max(0.0, (self._peak_value - curr_value) / self._peak_value)
    return log_ret - LAMBDA * turnover - MU * drawdown
```

**Tracking de `peak_value`:** inicializado a `initial_cash` en `reset()`; actualizado a cada step como `max(self._peak_value, curr_value)`.

**Exploit esperado:** El agente puede aprender a mantener cash (turnover=0, drawdown=0) para obtener recompensa constante ~0 en lugar de explorar. **Detección:** `sortino` del agente < `sortino` de `HoldCash` baseline.

---

## 5. Algorithm

### DQN con Double DQN Extension

**Justificación del Double DQN:** El DQN estándar sobreestima los valores-Q (sesgo de maximización en la selección del argmax). El Double DQN desacopla selección (red online) y evaluación (red target), reduciendo este sesgo sin costo computacional significativo. En mercados financieros con alta varianza de retornos, este sesgo es especialmente dañino.

### Hiperparámetros (valores en `configs/default.yaml`)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `hidden_dims` | `[256, 256]` | Capacidad suficiente para 22-94 features; dos capas ocultas capturan interacciones no lineales sin sobreajuste severo |
| `learning_rate` | `1e-4` | Estándar para Adam en problemas de RL financiero; LR más alto causa inestabilidad en Q-targets |
| `gamma` | `0.99` | Horizonte largo (efectivo ~100 steps); importa para capturar los efectos de drawdown diferidos |
| `epsilon_start` | `1.0` | Exploración total al inicio para llenar el replay buffer con transiciones diversas |
| `epsilon_end` | `0.05` | Exploración residual del 5%; evita que el agente quede atrapado en política subóptima |
| `epsilon_decay_steps` | `50000` | Decae linealmente desde 1.0 hasta 0.05 en 50k steps; ~25% del entrenamiento total |
| `batch_size` | `64` | Balance entre gradiente estable y eficiencia; tamaños más grandes no ayudan sin PER |
| `replay_buffer_size` | `100000` | Suficiente para ~4 años de datos 1h; evita olvidar regímenes pasados |
| `target_update_freq` | `1000` | Red target actualizada cada 1000 steps; estabiliza el entrenamiento sin lag excesivo |
| `train_steps` | `200000` | ~2 épocas sobre datos de entrenamiento; suficiente para convergencia inicial |

> **Hiperparámetros que NUNCA se pueden tunear en el split de test:** `gamma`, `epsilon_end`, cualquier hiperparámetro de red. Sólo pueden ajustarse en el split de validación (walk-forward folds). Hacerlo en el test es **causa de descalificación**.

---

## 6. Normalization and Lookahead Safety

Esta sección es **riesgo de descalificación #1** del proyecto. La regla es simple: el scaler solo ve datos de entrenamiento.

### Flujo correcto

```python
from src.data import load_prices, split, build_features

data = load_prices("1h")

# Fold 5 como ejemplo (config: train_end="2023-12-31", eval_end="2025-12-31")
train_raw, eval_raw = split(data, train_end="2023-12-31", eval_end="2025-12-31")

# fit=True SOLO en train
train_features, scaler = build_features(train_raw, fit=True)

# scaler se pasa sin re-fit al eval
eval_features, _       = build_features(eval_raw, scaler=scaler)
```

### Dónde vive el scaler en el entorno

```python
class TradingEnv(BaseTradingEnv):
    def __init__(self, prices, features, scaler, ...):
        super().__init__(prices, ...)
        self.features = features   # DataFrame ya escalado
        self.scaler   = scaler     # guardado para inspección/debugging
        self._lookback = max(21, lookback_window)   # ≥ 21 para evitar NaN de rolling

    def _obs(self) -> np.ndarray:
        feat = self.features.loc[self.data.index[self._t]].values  # 18 features
        weights = self._weights                                      # 4 pesos
        return np.concatenate([feat, weights]).astype(np.float32)
```

### Dónde NO debe ir `scaler.fit()`

- Nunca en `build_features(eval_raw, fit=True)` → lookahead.
- Nunca sobre el DataFrame completo antes de hacer `split()` → lookahead.
- Nunca re-fit al inicio de cada episodio → lookahead implícito en eval.

**Violación más común:** llamar `build_features(data, fit=True)` donde `data` es el DataFrame completo (train + eval). Esto filtra información futura hacia el scaler, inflando artificialmente los resultados de evaluación.

---

## 7. Baseline Evaluation

Todos los baselines están en `src/baselines.py` y deben ejecutarse bajo condiciones idénticas al agente.

| Clase | Acción devuelta | Rol diagnóstico | ¿Qué significa si el agente pierde? |
|-------|-----------------|-----------------|--------------------------------------|
| `RandomPolicy` | Aleatoria | Sanity floor mínimo | El agente no aprendió nada; revisar reward, bugs en entrenamiento |
| `HoldCash` | 0 (all cash) | Costo de oportunidad de no invertir | El agente genera retornos negativos netos de fees; reward design rompe incentivos |
| `HoldAsset0` | 1 (100% a0) | Benchmark de un activo single-asset | El agente no puede superar buy-and-hold; diversificación no ayuda o fees son excesivos |
| `EqualWeight` | 4 (1/3 c/activo) | Benchmark de diversificación pasiva | El agente no aprende timing; su signal no supera el ruido del rebalanceo periódico |
| `SMA` | 4 si momentum>0 else 0 | Heurística de tendencia (short=5, long=20) | El agente no aprende seguimiento de tendencia — la señal más elemental del mercado |

> **Contrato de acción fijo:** `HoldCash→0`, `HoldAsset0→1`, `EqualWeight→4` están hardcodeados en baselines.py. El menú de `_weights_from_action()` **no puede reasignar** estos índices.

**Jerarquía esperada de rendimiento (Sortino en eval):**

`Agente` > `SMA` > `EqualWeight` ≥ `HoldAsset0` > `HoldCash` > `RandomPolicy`

Si el agente queda por debajo de `EqualWeight` en todos los folds de walk-forward, revisar lookahead o reward hacking antes de reportar.

---

## 8. Team Task Split

### Persona 1 — EDA y Datos
**Branch:** `feature/eda-and-data`

| Entregable | Detalle | Dependencia |
|------------|---------|-------------|
| `notebooks/eda.ipynb` completo | Todas las celdas ejecutadas; análisis de distribuciones, regímenes de volatilidad, correlación, drawdown histórico, y features de `build_features()` | Ninguna |
| Informe de features | Para cada una de las 18 features: media, std, autocorrelación a lag-1, y correlación con el retorno del siguiente período | Ninguna |
| Selección de `INTERVAL` | Justificar si usar 1h, 30m ó 15m basado en ruido vs señal (bias-variance tradeoff del estado) | Ninguna |
| Recomendación de lookback | Cuántos timesteps de historia incluir en `_obs()`, basado en autocorrelación de retornos | Ninguna |
| Análisis de splits | Confirmar que cada fold de `configs/default.yaml` incluye al menos un régimen alcista y uno bajista | Ninguna |

**Riesgo para esta persona:** interpretar el TBR sin entender que es un average del período de la vela, no una observación en tiempo real.

---

### Persona 2 — TradingEnv Implementation
**Branch:** `feature/env-and-features`

| Entregable | Detalle | Dependencia |
|------------|---------|-------------|
| `agent.py::TradingEnv` | Subclase de `BaseTradingEnv` con los tres métodos implementados | Diseño de estado y acción de Persona 1 |
| `_weights_from_action()` | Menú de 10 acciones del Sección 3, con assertions que validen suma=1 | Ninguna |
| `_obs()` | Vector de features escaladas + pesos actuales; `self.scaler` almacenado correctamente | Scaler de `build_features(train, fit=True)` |
| `_reward()` | Implementación de R3 (log-ret - turnover - drawdown), con `self._peak_value` inicializado en `reset()` | Ninguna |
| Tests de sanity | Script que verifica: `env.reset()` no lanza NaN, `env.step(action)` mantiene `sum(weights)==1`, un episodio completo termina sin crash | Ninguna |

**Riesgo para esta persona:** olvidar actualizar `self._lookback` en `__init__()` → `reset()` empieza en índice 0 donde los rolling features son NaN.

---

### Persona 3 — Agent, Training Loop, Evaluation
**Branch:** `feature/agent-and-eval`

| Entregable | Detalle | Dependencia |
|------------|---------|-------------|
| `agent.py::Agent` | DQN con Double DQN; red neuronal `[obs_dim → 256 → 256 → n_actions]`; replay buffer; epsilon-greedy | TradingEnv de Persona 2 |
| Training loop | Walk-forward sobre los 5 folds de `configs/default.yaml`; scaler entrenado solo en train split por fold | TradingEnv de Persona 2 |
| Logging | Curva de episodic return durante entrenamiento; `compute_metrics()` al final de cada fold en train y eval | `src/metrics.py` |
| Ablación de `transaction_costs` | Evaluación a 0, 10, 25 bps según `eval.transaction_costs_ablation` en `configs/default.yaml` | TradingEnv de Persona 2 |
| Comparación de baselines | Tabla de Sortino, max_dd, cum_ret para el agente y los 5 baselines en cada fold | `src/baselines.py` |
| Tests de submission | `uv run pytest tests/test_submission.py -v` debe pasar antes de entregar | Ambas personas |

**Riesgo para esta persona:** tunear hiperparámetros mirando las métricas de eval. Toda búsqueda de hiperparámetros debe hacerse en los folds de walk-forward, nunca en el held-out.

---

## Apéndice — Checklist de Riesgos Críticos

| Riesgo | Síntoma | Fix |
|--------|---------|-----|
| **Lookahead en scaler** | Métricas de eval infladas ~20-40% vs train | `build_features(train, fit=True)` → `build_features(eval, scaler=scaler)` |
| **NaN en rolling features** | `_obs()` retorna NaN; Q-network diverge | `self._lookback >= 21`; verificar con `np.isnan(obs).any()` |
| **Turnover hacking** | Agente hace flip en cada step; fees devoran retorno | Reducir `LAMBDA` si el agente se paraliza; aumentar si hace churn |
| **Cash parking** | Agente siempre retorna acción 0; `sortino < HoldCash` | Revisar escala de recompensa; agregar penalización de inactividad |
| **Tunear en test** | Overfitting al held-out; descalificación | Toda búsqueda de hiperparámetros solo en walk-forward folds |
| **Shorting sin funding rate** | Agente sobreusa posiciones cortas porque son "gratis" | Documentar la simplificación; comparar con y sin acciones short |
| **Índice temporal en `_obs()`** | Usar `self._t` para indexar `self.features.values[self._t]` sin verificar que los índices de fecha coincidan | Indexar por fecha: `self.features.loc[self.data.index[self._t]]` |
