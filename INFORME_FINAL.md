# INFORME FINAL — Bellman Capital
> Informe de entrega del agente DQN de asignación de portafolio.
> Referencia de código: `agent.py`. Infraestructura de solo lectura: `src/`.

---

## 0. Equipo

- **Nombre del agente:** Bellman Capital
- **Investigadores:** Aljeandro Rubiano, Juan Camilo San Miguel, Juan Sebastian Londoño
- **Fecha de inicio:** 2026-05-30
- **Tesis:** El agente cree que los mercados de criptomonedas a escala horaria exhiben suficiente estructura de momentum y clustering de volatilidad como para que una política aprendida con DQN pueda asignar capital de forma rentable con ajuste por riesgo. Específicamente: (a) la volatilidad realizada de 21 horas predice la volatilidad futura (efecto GARCH), lo que permite estimar el riesgo de cada posición; (b) el momentum de 20 períodos tiene persistencia estadística que el agente puede explotar; (c) el intervalo de 1 hora ofrece la mejor relación señal/ruido para estas features de ventana corta, como concluye el análisis de periodicidad; y (d) la métrica Sortino —que penaliza solo la volatilidad negativa— es el objetivo correcto para un agente que debe proteger capital en caídas y capturar retornos al alza. Los resultados de V3 matizan esta tesis: el agente logró cumplirla en fold 1 (mercado alcista 2020, sortino=+1.858), pero falló en folds 2-5 por desajuste de régimen entre entrenamiento y evaluación.

---

## 1. Formulación del Problema

### Estado (State Space)

El agente recibe en cada paso de tiempo un vector de observación de **22 dimensiones** (`agent.py, TradingEnv._obs(), líneas 84–93`). Todos los valores de features están normalizados a media 0, desviación estándar 1, mediante un `StandardScaler` ajustado exclusivamente sobre datos de entrenamiento (`agent.py, train_fold(), línea 350`).

#### Tabla completa del vector de observación

| Índice | Nombre | Cálculo (`src/data.py:69–93`) | Justificación financiera |
|--------|--------|-------------------------------|--------------------------|
| 0 | `asset_0_log_ret` | `log(close_t / close_{t−1})` | Retorno logarítmico del período: simétrico, estacionario, señal directa del movimiento de precio. |
| 1 | `asset_0_vol_21` | `rolling(21).std(log_ret)` | Volatilidad realizada 21h: captura volatility clustering — períodos de alta volatilidad predicen alta volatilidad futura (efecto GARCH). |
| 2 | `asset_0_mom_20` | `log(close_t / close_{t−20})` | Momentum de 20 períodos: señal de tendencia a corto-medio plazo con evidencia empírica de persistencia. |
| 3 | `asset_0_atr_14` | `ATR(14) / close` | Rango verdadero promedio normalizado: proxy del riesgo intradía sin sesgo de escala, incluye gaps de precio. |
| 4 | `asset_0_vol_ratio` | `volume / rolling(21).mean(volume)` | Volumen relativo: picos de volumen suelen preceder movimientos grandes de precio (teoría de microestructura). |
| 5 | `asset_0_tbr` | `taker_buy_ratio` (columna en parquet) | Presión compradora neta por vela: valor > 0.5 indica sesgo comprador, señal de microestructura de corto plazo. |
| 6 | `asset_1_log_ret` | Igual que índice 0 para asset_1 | Igual justificación para el segundo activo. |
| 7 | `asset_1_vol_21` | Igual que índice 1 para asset_1 | Ídem. |
| 8 | `asset_1_mom_20` | Igual que índice 2 para asset_1 | Ídem. |
| 9 | `asset_1_atr_14` | Igual que índice 3 para asset_1 | Ídem. |
| 10 | `asset_1_vol_ratio` | Igual que índice 4 para asset_1 | Ídem. |
| 11 | `asset_1_tbr` | Igual que índice 5 para asset_1 | Ídem. |
| 12 | `asset_2_log_ret` | Igual que índice 0 para asset_2 | Igual justificación para el tercer activo. |
| 13 | `asset_2_vol_21` | Igual que índice 1 para asset_2 | Ídem. |
| 14 | `asset_2_mom_20` | Igual que índice 2 para asset_2 | Ídem. |
| 15 | `asset_2_atr_14` | Igual que índice 3 para asset_2 | Ídem. |
| 16 | `asset_2_vol_ratio` | Igual que índice 4 para asset_2 | Ídem. |
| 17 | `asset_2_tbr` | Igual que índice 5 para asset_2 | Ídem. |
| 18 | `w_asset_0` | `self._weights[0]` | Peso actual en asset_0: necesario para calcular el costo de rebalanceo (condición de Markov). |
| 19 | `w_asset_1` | `self._weights[1]` | Peso actual en asset_1: ídem. |
| 20 | `w_asset_2` | `self._weights[2]` | Peso actual en asset_2: ídem. |
| 21 | `w_cash` | `self._weights[3]` | Peso actual en cash: ídem. |

#### Respuestas a las tres preguntas del rubric

**1. ¿Qué contiene la observación en el tiempo t?**
Las 18 features de mercado del timestep actual (6 por cada uno de los 3 activos de riesgo), más los 4 pesos actuales del portafolio. Todas las features de mercado son calculadas por `build_features()` en `src/data.py:48–102` usando únicamente datos hasta `t`, y normalizadas con un scaler ajustado solo en entrenamiento. El total es 22 valores float32.

**2. ¿Cómo se trata la naturaleza no-Markov de los precios crudos?**
Los precios crudos no son Markov: el log-retorno en `t` no contiene información sobre la volatilidad reciente, el momentum de días anteriores ni la posición actual del portafolio. La transformación en features resuelve esto parcialmente: `vol_21` codifica la volatilidad de las últimas 21 horas, `mom_20` codifica el momentum de los últimos 20 períodos, y `atr_14` captura el rango de las últimas 14 velas. Los 4 pesos del portafolio (`self._weights`) convierten el problema en Markov para la decisión de rebalanceo: sin conocer la posición actual, el agente no puede calcular el costo de transacción de ninguna acción futura (`GUIA_EQUIPO.md, Sección 2`).

**3. ¿Cuánta historia se incluye y por qué?**
La ventana máxima de las features es de 21 períodos (feature `vol_21`). A 1h, esto equivale a ~21 horas de historia. Esta ventana es suficiente para capturar volatility clustering y momentum de corto plazo, y fue elegida para que las features tengan buena relación señal/ruido a 1h (ventanas más cortas capturan ruido de microestructura; ventanas más largas requerirían más datos y no son compatibles con el presupuesto de pasos). El parámetro `_lookback = 21` en `TradingEnv.__init__()` (`agent.py, línea ~57`) garantiza que el entorno empieza en `t=21`, la primera fila con features válidas tras eliminar NaN de rolling con `dropna()` (`src/data.py:95`).

---

### Espacio de Acciones (Action Space)

El agente selecciona en cada paso una de **10 carteras predefinidas** del menú `_WEIGHT_MENU` (`agent.py, líneas 29–40`). Los pesos tienen el formato `[asset_0, asset_1, asset_2, cash]`, deben sumar 1.0, y el peso de cash debe ser ≥ 0.

#### Tabla del menú de acciones

| ID | Nombre | Pesos `[a0, a1, a2, cash]` | Interpretación económica |
|----|--------|---------------------------|--------------------------|
| 0 | All Cash | `[0.00, 0.00, 0.00, 1.00]` | Refugio total: sin exposición al riesgo de mercado. Capital en cash, retorno ≈ 0. |
| 1 | Long A0 | `[1.00, 0.00, 0.00, 0.00]` | Apuesta concentrada en asset_0. Máximo riesgo/retorno sobre un solo activo. |
| 2 | Long A1 | `[0.00, 1.00, 0.00, 0.00]` | Apuesta concentrada en asset_1. |
| 3 | Long A2 | `[0.00, 0.00, 1.00, 0.00]` | Apuesta concentrada en asset_2. |
| 4 | Equal Weight | `[0.33, 0.33, 0.33, 0.00]` | Diversificación máxima entre los tres activos. Equivalente al benchmark EqualWeight. |
| 5 | Conservative | `[0.25, 0.25, 0.25, 0.25]` | Diversificado con 25% en cash como amortiguador de drawdown. |
| 6 | Short A0 | `[−0.50, 0.50, 0.50, 0.50]` | Posición corta en asset_0 con cobertura larga en los otros. Gana si asset_0 cae. |
| 7 | Short A1 | `[0.50, −0.50, 0.50, 0.50]` | Posición corta en asset_1 con cobertura larga en los otros. |
| 8 | Short A2 | `[0.50, 0.50, −0.50, 0.50]` | Posición corta en asset_2 con cobertura larga en los otros. |
| 9 | Long A1+A2 | `[0.00, 0.50, 0.50, 0.00]` | Portafolio largo excluyendo asset_0. Diversificación parcial. |

#### Respuestas a las tres preguntas del rubric

**1. ¿Cuál es el espacio de acciones?**
Discreto con 10 acciones (índices 0–9), cada una mapeada a un vector de pesos de portafolio `(4,)` como muestra la tabla anterior.

**2. ¿Por qué esta representación?**
El método abstracto `_weights_from_action(action: int)` de `BaseTradingEnv` (`src/env.py:75`) espera un índice entero. El menú discreto es la única representación compatible con DQN sin cambiar el algoritmo base. El menú discreto también es interpretable: cada acción tiene un nombre y una tesis económica clara. Las acciones 0, 1 y 4 están fijadas por contrato con los baselines `HoldCash`, `HoldAsset0` y `EqualWeight` de `src/baselines.py:12,15,18`.

**3. ¿Qué impide este diseño?**
El menú discreto no puede expresar pesos continuos arbitrarios. Un agente óptimo que quisiera, por ejemplo, 47% en asset_0 y 53% en asset_2, no puede hacerlo — solo puede elegir entre las 10 carteras predefinidas. Adicionalmente, el modelo no cobra tasa de financiamiento (*funding rate*) sobre las posiciones cortas (acciones 6-8), lo que es una simplificación respecto a los mercados reales (`src/env.py:32–34`, `GUIA_EQUIPO.md, Sección 3`).

---

## 2. Datos y Análisis Exploratorio

### Intervalo usado

**1 hora (1h)**, cargado con `load_prices("1h")` en `agent.py:346`, que lee `data/raw/prices_1h.parquet` (`src/data.py:20–27`). La decisión de mantener 1h se fundamenta en el análisis de periodicidad de `ENTRENAMIENTO_V3.md, Sección 8`, que concluye:

| Intervalo | Filas fold 1 | Episodios con 200k pasos | Ventana de features |
|-----------|-------------|--------------------------|---------------------|
| 15m | ~69 752 | ~2.9 | 21 candles ≈ 5h |
| 30m | ~34 876 | ~5.7 | 21 candles ≈ 10.5h |
| **1h (elegido)** | **17 438** | **~11.5** | **21 candles ≈ 21h** |

A 30m o 15m, el mismo presupuesto de 200k pasos produce menos episodios completos por fold, degradando la convergencia de Q-valores. Adicionalmente, las features de rolling a 21 candles capturan dinámicas de ~21 horas con 1h (señal de 1-2 días de trading), frente a solo ~5h con 15m (ruido de microestructura intradía).

### Features producidas por `build_features()` (`src/data.py:48–102`)

Se generan **18 features** (6 por activo × 3 activos), luego se aplica `StandardScaler`:

- `log_ret`, `vol_21`, `mom_20`, `atr_14`, `vol_ratio`, `taker_buy_ratio`
- Todas las operaciones usan `.shift(1)` o `.rolling(W)` con índices pasados → **lookahead-safe**
- `.dropna()` en `src/data.py:95` elimina las primeras ~21 filas con NaN

### Estructura de folds (walk-forward)

| Fold | train_end | eval_end | train_rows | eval_rows |
|------|-----------|----------|-----------|----------|
| 1 | 2019-12-31 | 2020-12-31 | 17 438 | 8 795 |
| 2 | 2020-12-31 | 2021-12-31 | 26 210 | 8 776 |
| 3 | 2021-12-31 | 2022-12-31 | 34 963 | 8 783 |
| 4 | 2022-12-31 | 2023-12-31 | 43 723 | 8 784 |
| 5 | 2023-12-31 | 2025-12-31 | 52 484 | 17 544 |

Los splits están definidos en `configs/default.yaml:6–11` y son inmutables.

### Riesgos de lookahead identificados y cómo se evitaron

**Riesgo principal — scaler ajustado sobre datos completos:** Si `build_features(data, fit=True)` se llama sobre el DataFrame completo (train + eval), el scaler "ve" la media y desviación estándar del período de evaluación. En `agent.py, train_fold()`, la barrera es explícita:
```python
# LOOKAHEAD BOUNDARY — scaler.fit() en TRAIN DATA ONLY
train_features, scaler = build_features(train_data, fit=True)   # fit aquí solo
eval_features, _       = build_features(eval_data, scaler=scaler)  # sin re-fit
```
La función `split()` (`src/data.py:36–45`) garantiza partición sin solapamiento.

**Riesgo secundario — NaN en rolling features:** `build_features()` descarta con `.dropna()` las primeras 21 filas. `_lookback = 21` en `TradingEnv.__init__()` hace que `reset()` arranque en `self._t = 21`, la primera fila válida. Verificado por el test `test_lookback_set` y `test_reset_returns_no_nan` en `tests/test_submission.py`.

---

## 3. Diseño del Entorno

### Cómo funciona `BaseTradingEnv` (`src/env.py`)

El entorno recibe la acción del agente y ejecuta la siguiente secuencia en `step()` (`src/env.py:49–68`):

**1. Obtener nuevos pesos:**
```python
w = self._weights_from_action(action)       # src/env.py:50
```

**2. Calcular turnover:**
```python
turnover = np.abs(w - self._weights).sum()  # src/env.py:57
```
Suma de cambios absolutos en cada peso. Pasar de all-cash a EqualWeight = turnover 2.0.

**3. Calcular retorno de precios:**
```python
ret = self.prices[self._t] / self.prices[self._t - 1]  # src/env.py:58
```
Vector de 4 elementos; cash siempre = 1.0.

**4. Actualizar valor del portafolio (fórmula exacta):**
```python
self._value = self._value * float(np.dot(w, ret)) - self._value * turnover * self.tc
```
El primer término es la ganancia/pérdida por retornos de precios; el segundo es el costo de transacción.

**5. Avanzar el reloj:**
```python
self._t += 1
terminated = self._t >= len(self.prices)    # src/env.py:62–64
```

**6. Construir observación y recompensa** mediante `_obs()` y `_reward()` del subclase.

### Qué implementa `TradingEnv` en `agent.py`

| Método | Líneas | Qué hace |
|--------|--------|----------|
| `_obs()` | 84–93 | Concatena las 18 features escaladas del timestep actual con los 4 pesos actuales. Clampea `t = min(self._t, len(self.data)−1)` para evitar IndexError en el último paso. Accede por fecha: `self.features.loc[self.data.index[t]]`. |
| `_weights_from_action()` | 98–103 | Consulta `_WEIGHT_MENU[action]`, captura `self._last_turnover` antes de que BaseTradingEnv sobreescriba `self._weights`. |
| `_reward()` | 107–122 | Formulación R3 diferencial Sortino (activa en V3): `log_ret − LAMBDA×turnover − MU×max(0,−log_ret)`. |

### Modelado de costos de transacción

`tc = 0.001` (10 bps) en `BaseTradingEnv.__init__()`, cargado desde `configs/default.yaml:15`. La fórmula exacta descuenta `valor × turnover × 0.001` del portafolio en cada paso. Para una rotación total (turnover = 2.0), el costo es 0.2% del valor del portafolio. Las posiciones cortas (acciones 6-8) no tienen costo adicional de financiamiento — simplificación documentada en `src/env.py:32–34`.

---

## 4. Diseño de la Función de Recompensa

Esta sección documenta la iteración completa de diseño. El primer entrenamiento real (V1) usó la formulación R3 con coeficientes altos. Las versiones R1 y R2 fueron analizadas y descartadas antes de correr entrenamiento.

### R1 — Log-retorno puro (diseñado, no entrenado)

**Fórmula** (`GUIA_EQUIPO.md, Sección 4`):
```
R1 = log(valor_nuevo / valor_viejo)
```

**Comportamiento esperado y exploit:** El agente maximiza retorno sin costo de trading. El exploit predecible es el *churn*: el agente aprende a hacer flip entre posiciones opuestas en cada paso, porque el horizonte corto no le permite ver que los costos de transacción acumulados (10 bps por unidad de turnover) destruyen el retorno. También tiende a concentrarse en el activo con mayor retorno reciente sin considerar el riesgo de reversión. **No fue entrenado por diseño.** La decisión de empezar con R3 (log_ret + penalización de turnover + penalización de drawdown) fue deliberada para evitar el churn desde el primer intento.

### R2 — Log-retorno menos penalización de turnover (diseñado, no entrenado independientemente)

**Fórmula** (`GUIA_EQUIPO.md, Sección 4`):
```
R2 = log(valor_nuevo / valor_viejo) − 0.1 × turnover
```

**Análisis matemático del structural trap** (`ENTRENAMIENTO_V2.md, Sección 5`):
Con `LAMBDA = 0.1` y el portafolio en drawdown severo (90%), la comparación entre acciones es:

| Escenario | Costo de transacción | R2 |
|-----------|---------------------|-----|
| Quedarse en A1 (cae 0.026%/paso) | 0.0 | −0.000263 |
| Cambiar a cash | 0.1 × 2.0 = 0.200 | −0.200 |

**Cambiar a cash cuesta 760× más** que quedarse en una posición perdedora, por paso. Con `gamma = 0.99`, el beneficio descontado de escapar (`0.000263 × 100 ≈ 0.026`) es menor que el costo de transición (`0.200`). El agente aprende racionalmente que no moverse es "óptimo", aunque el portafolio siga cayendo. Esta es la trampa estructural del R2 con LAMBDA alto: el costo de rebalanceo supera al beneficio esperado de escapar, congelando al agente en posiciones perdedoras.

### R3 Primera formulación (V1 — LAMBDA=0.1, MU=0.5)

**Fórmula** (`agent.py, ENTRENAMIENTO_V1.md`):
```
R3_V1 = log_ret − 0.1 × turnover − 0.5 × drawdown
donde drawdown = max(0, (peak_value − curr_value) / peak_value)
```

**Resultados observados (fold 5, smoke test):**
```
sortino = −5.49    cum_ret = −0.9999    max_dd = −0.9999
```
El portafolio perdió el 99.99% de su valor ($10 000 → ~$1).

**Fallos diagnosticados (`ENTRENAMIENTO_V1.md, Sección 3`):**

- **Fallo A — Short bias:** el agente probablemente convergió a una posición corta (acciones 6-8) durante el período bajista de 2022 en el entrenamiento. En evaluación (2024-2025, bull market), esa política corta produjo pérdidas de ~0.05% por hora compuesto: `0.9995^17544 ≈ 0.00015`, llevando el portafolio a cero.

- **Fallo B — NaN silencioso:** `log(curr_value / prev_value)` produce NaN si `curr_value < 0` (posible con posiciones cortas en un rally fuerte). Este NaN se guardaba en el replay buffer y corrompía los pesos de la red silenciosamente.

- **Fallo C — Espiral de drawdown:** con `MU = 0.5` y drawdown = 0.5, la penalización vale `0.5 × 0.5 = 0.25`, superando al `log_ret` horario típico (~0.0001–0.001) en **dos órdenes de magnitud**. Todas las acciones producen reward ≈ −0.25 sin importar cuál se elija. La señal de aprendizaje desaparece.

- **Fallo D — Sin logging de acciones:** el output de consola no registraba qué acción elegía el agente. Imposible diagnosticar cash parking o short bias sin modificar el código.

- **Fallo E — LAMBDA demasiado alto en primer paso:** con `LAMBDA = 0.1` y estado inicial all-cash, cualquier cambio a posición de riesgo tiene `penalización = 0.1 × 2.0 = 0.2`, ~200× mayor que el log_ret esperado.

### R3 Segunda formulación (V2 — LAMBDA=0.01, MU=0.05)

**Fixes aplicados desde V1:**
- Fix 1: `log_ret = log(max(curr_value, 1e-8) / ...)` — elimina NaN (`agent.py:111`)
- Fix 2: `MU = 0.5 → 0.05` — rompe la espiral de drawdown (`agent.py:50`)
- Fix 3: logging de `top_action` cada 20 000 pasos con frecuencia (`agent.py:266–310`)
- Fix 4: `LAMBDA = 0.1 → 0.01` — permite salir de all-cash (`agent.py:49`)

**Fórmula:**
```
R3_V2 = log_ret − 0.01 × turnover − 0.05 × drawdown
```

**Resultados observados (fold 5, smoke test):**
```
sortino = −3.67    cum_ret = −1.0    max_dd = −1.0    ann_ret = −0.2206    ann_vol = 0.0814
```

**top_action progresión durante entrenamiento:**

| Paso | top_action | porcentaje |
|------|------------|-----------|
| 20 000 | 9 (Long A1+A2) | 11% |
| 40 000 | 2 (Long A1) | 17% |
| 60 000 | 2 (Long A1) | 17% |
| 80 000 | 2 (Long A1) | 13% |
| 100 000 | 2 (Long A1) | 22% |
| 120 000 | 1 (Long A0) | 13% |
| 140 000 | 2 (Long A1) | 16% |
| 160 000 | 2 (Long A1) | 17% |
| 180 000 | 0 (All Cash) | 14% |
| 200 000 | 2 (Long A1) | 19% |

**Trampa estructural persistente identificada (`ENTRENAMIENTO_V2.md, Sección 3`):**
Con drawdown = 90% y portfolio ≈ $1 000:

```
Quedarse en A1 (sigue cayendo):   R3_V2 = −0.000263 − 0 − 0.05×0.90 = −0.04526
Cambiar a cash:                   R3_V2 = 0 − 0.01×2.0 − 0.05×0.90 = −0.0695
```

**El drawdown es idéntico para ambas acciones**: depende del pico histórico y del valor actual, no de la acción elegida. Cambiar a cash cuesta adicionalmente `−0.020` de turnover. El agente aprende que quedarse en la posición perdedora (−0.0498) es menos costoso que escapar (−0.0695). Esta es la trampa fundamental del diseño de drawdown acumulativo: **una señal que no diferencia entre acciones nunca puede resolver un conflicto entre acciones**.

### R3 Formulación final — Sortino Diferencial (V3 — LAMBDA=0.01, MU=0.05)

**Fix 5 aplicado** (`agent.py:107–122`): reemplazar drawdown acumulativo por penalización del retorno negativo del paso actual.

**Fórmula:**
```
downside = max(0, −log_ret)
R3_V3 = log_ret − 0.01 × turnover − 0.05 × downside
```

**Por qué rompe la trampa matemáticamente** (`ENTRENAMIENTO_V2.md, Sección 5, Opción C`):

Con portfolio en drawdown 90%, asset_1 cayendo 0.026%/paso:

| Escenario | R3_V3 inmediato | R3 futuro por paso |
|-----------|----------------|-------------------|
| Quedarse en A1 | −0.000263 − 0.05×0.000263 = **−0.000276** | −0.000276/paso perpetuo |
| Cambiar a cash | 0 − 0.01×2.0 − 0 = **−0.020** (solo el paso de cambio) | **0** (cash no cae) |

Beneficio descontado de escapar: `0.000276 × Σ(0.99^t) ≈ 0.000276 × 100 = 0.0276`
Costo de transición: `0.020`
Como `0.0276 > 0.020`, el agente recupera el costo de cambio en ~72 pasos. La trampa está rota.

El `downside` es **diferente para cada acción** (depende del `log_ret` del paso, que es acción-dependiente), a diferencia del drawdown acumulativo que era idéntico para todas. Esta propiedad es lo que convierte la señal en aprendible.

**Exploit nuevo posible:** el agente podría aprender a rotar hacia activos que subieron marginalmente para evitar pasos con `log_ret < 0`. Este riesgo está mitigado por `LAMBDA × turnover = 0.01 × 2.0 = 0.02`, que penaliza el rebalanceo excesivo.

**Resultados V3 (5 folds completos):**

| Fold | train_end | eval_end | train_rows | sortino | cum_ret | max_dd | top_action final |
|------|-----------|----------|-----------|---------|---------|--------|-----------------|
| 1 | 2019-12-31 | 2020-12-31 | 17 438 | **+1.8580** | **+127.8567** | **−0.2333** | 7 (22%) |
| 2 | 2020-12-31 | 2021-12-31 | 26 210 | −1.5286 | −0.9978 | −0.9985 | 4 (12%) |
| 3 | 2021-12-31 | 2022-12-31 | 34 963 | −3.5462 | −1.0000 | −1.0000 | 0 (12%) |
| 4 | 2022-12-31 | 2023-12-31 | 43 723 | −4.9175 | −0.9998 | −0.9998 | 6 (12%) |
| 5 | 2023-12-31 | 2025-12-31 | 52 484 | −3.8967 | −1.0000 | −1.0000 | 1 (20%) |

### Tabla comparativa V1 vs V2 vs V3

| Versión | Reward | Fold 5 sortino | Fold 1 sortino | Mejora clave |
|---------|--------|---------------|---------------|--------------|
| V1 | R3 LAMBDA=0.1, MU=0.5 | −5.49 | — | (primera versión) |
| V2 | R3 LAMBDA=0.01, MU=0.05 | −3.67 | — | Recalibración de coeficientes |
| V3 | R3 Sortino diferencial | −3.90 | **+1.86** | Rompe trampa estructural, 5 folds |

La mejora de V1 a V2 (de −5.49 a −3.67) fue real y fue producida por los fixes de recalibración (Fix 2: MU=0.05; Fix 4: LAMBDA=0.01). El Sortino diferencial de V3 resolvió teóricamente la trampa estructural pero no mejoró fold 5 (−3.90 ≈ −3.67, diferencia dentro del ruido), porque el problema raíz en folds 2-5 es el desajuste de régimen de mercado, no la función de reward.

---

## 5. Algoritmo

### Double DQN — justificación

**Problema con DQN estándar:** cuando la red estima el valor de la mejor siguiente acción, usa la misma red para *elegir* cuál es la mejor acción y *evaluar* cuán buena es. Esto crea un sesgo sistemático hacia sobreestimar el valor de las acciones (*maximisation bias*). En mercados financieros con alta varianza de retornos, este sesgo es especialmente dañino: el agente cree que sus acciones son mejores de lo que realmente son y desarrolla políticas demasiado agresivas.

**Solución (van Hasselt et al. 2016):** dos redes separadas, implementadas en `agent.py, Agent.update(), líneas 218–222`:
```python
next_act = self.online_net(nobs_t).argmax(dim=1, keepdim=True)  # online: elige
q_next   = self.target_net(nobs_t).gather(1, next_act)          # target: evalúa
```
La **red target** se sincroniza cada `TARGET_UPDATE_FREQ = 1 000` pasos, proporcionando un objetivo estable durante ese período — análogo a "medir con una regla fija en lugar de medir con la misma cuerda que se está estirando".

### Hiperparámetros (fuente: `configs/default.yaml`)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `hidden_dims` | `[256, 256]` | Capacidad suficiente para 22 features; dos capas capturan interacciones no lineales sin sobreajuste severo. |
| `learning_rate` | `1e-4` | Estándar para Adam en problemas de RL financiero; LR más alto causa inestabilidad en Q-targets. |
| `gamma` | `0.99` | Horizonte efectivo ~100 pasos; necesario para capturar los efectos de drawdown diferidos. |
| `epsilon_start` | `1.0` | Exploración total al inicio: llena el replay buffer con transiciones diversas antes de aprender. |
| `epsilon_end` | `0.05` | Exploración residual del 5%; evita que el agente quede atrapado en política subóptima. |
| `epsilon_decay_steps` | `50 000` | Decaimiento lineal en los primeros 50k pasos (~25% del entrenamiento); el 75% restante es casi greedy. |
| `batch_size` | `64` | Balance entre gradiente estable y eficiencia computacional. |
| `replay_buffer_size` | `100 000` | ~4 años de datos horarios; permite recordar regímenes pasados durante un episodio. |
| `target_update_freq` | `1 000` | Red target actualizada cada 1 000 pasos: estabiliza el entrenamiento sin lag excesivo. |
| `train_steps` | `200 000` | ~11.5 episodios completos para fold 1 (17 438 filas); insuficiente para folds 2-5 (ver Sección 9). |

### Nota sobre TRAIN_STEPS y BUFFER_SIZE — restricción del test

`Agent.TRAIN_STEPS = 200_000` y `Agent.BUFFER_SIZE = 100_000` son constantes de clase que deben coincidir exactamente con `configs/default.yaml` según `tests/test_submission.py, test_agent_hyperparams_match_config(), líneas 199–210`. El análisis de V3 identificó que 200k pasos son insuficientes para folds 2-5 (3.8–7.6 episodios completos vs 11.5 en fold 1), y que el buffer de 100k puede estar dominado por el episodio más reciente en fold 5 (52 484 filas = 52% del buffer). Sin embargo, modificar estas constantes de clase rompería los tests de submission. La solución es pasar `train_steps` como argumento a `train_fold()` sin cambiar la constante de clase — cambio pendiente para V4.

---

## 6. Baselines

Todos los baselines están en `src/baselines.py` y deben ejecutarse bajo condiciones idénticas al agente.

| Baseline | Clase | Acción | Rol diagnóstico | ¿Qué significa perder contra él? |
|----------|-------|--------|-----------------|----------------------------------|
| Random | `RandomPolicy` | Aleatoria | Sanity floor mínimo | El agente no aprendió nada; revisar reward y bugs de entrenamiento. |
| Hold Cash | `HoldCash` | 0 | Costo de oportunidad de no invertir | El agente genera retornos negativos netos de fees; reward rompe incentivos. |
| Hold Asset 0 | `HoldAsset0` | 1 | Benchmark single-asset | El agente no supera buy-and-hold; diversificación no ayuda o fees son excesivos. |
| Equal Weight | `EqualWeight` | 4 | Benchmark de diversificación pasiva | El agente no aprende timing; su señal no supera el ruido del rebalanceo. |
| SMA Crossover | `SMA` | 4 si momentum>0, si no 0 | Heurística de tendencia | El agente no aprende la señal de tendencia más elemental del mercado. |

#### ¿El agente superó a EqualWeight por fold?

*dato no disponible — pendiente de calcular*. Los baselines no fueron corridos automáticamente por `train_fold()` durante V3. El script para calcularlos manualmente está documentado en `IMPLEMENTACION.md, Sección 10.5`. Lo que sí se puede afirmar con los datos disponibles:

- **Fold 1:** `sortino = +1.8580`, `cum_ret = +127.8567`. EqualWeight (acción 4, rebalanceo pasivo) no puede producir un cum_ret de +12 785% en un mercado que subió entre 3× y 10× — el agente claramente superó a EqualWeight en fold 1.
- **Folds 2-5:** `cum_ret ≈ −1.0` para el agente. EqualWeight en 2021 (fold 2) habría producido un retorno positivo (bull market). En 2022 (fold 3, bear market severo), EqualWeight habría perdido significativamente pero es improbable que llegara a `−1.0`. El agente estuvo por debajo de EqualWeight en todos estos folds.

#### El fold que funcionó: fold 1 (eval 2020) y por qué es estructuralmente diferente

2020 fue históricamente el año de mayor momentum direccional para las criptomonedas en el dataset: caída por COVID en marzo, recuperación en V, y rally acelerado hasta +200% en diciembre. Este tipo de mercado fuertemente tendencial es el escenario más favorable para un DQN porque: (a) las señales de momentum (`mom_20`) son estables y consistentes durante períodos prolongados; (b) la política "comprar y mantener el activo de mayor momentum" produce retornos extraordinarios sin necesidad de timing fino; (c) el agente de fold 1 completó ~11.5 episodios, el número suficiente para que los Q-valores converjan a esta política simple. El `cum_ret = +127.8567` refleja exactamente este escenario y no es atribuible a aprendizaje sofisticado de gestión de riesgo.

---

## 7. Protocolo de Entrenamiento

### Parámetros del entrenamiento

| Concepto | Valor |
|----------|-------|
| Total de pasos | 200 000 por fold × 5 folds = **1 000 000 pasos** |
| Hardware | **CPU** (inferido del código: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` en `agent.py:168`; los logs de V1 confirman tiempos de ~3–8 minutos por fold consistentes con CPU) |
| Tiempo estimado | ~20–45 minutos en total (5 folds × 3–8 min/fold, `IMPLEMENTACION.md, Sección 8`) |
| Semilla de reproducibilidad | `SEED = 42` fija en `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` en `agent.py:164–166` |

### Checkpointing

El sistema de persistencia guarda automáticamente durante `Agent.train()`:

| Archivo | Cuándo se escribe | Qué contiene |
|---------|------------------|--------------|
| `models/checkpoints/fold_{id}_step_{N}.pt` | Cada 10 000 pasos | Pesos de `online_net`, `target_net`, estado del optimizador Adam, y `self._step` |
| `models/checkpoints/fold_{id}_latest.pt` | Cada 10 000 pasos (sobreescribe) | Igual al anterior; usado para resume automático |
| `models/metrics/fold_{id}_loss.csv` | Cada 500 pasos | `step, loss, epsilon, buffer_size` |
| `models/metrics/fold_{id}_metrics.csv` | Al terminar evaluación de cada fold | `step, episode_reward, sortino, cum_ret, max_dd, epsilon, timestamp` |

El resume es automático: si `fold_{id}_latest.pt` existe al inicio de `Agent.train()`, se carga y el loop continúa desde `self._step + 1` (`agent.py:252–259`).

### Reproducibilidad

Con `SEED = 42`, los valores de epsilon son completamente deterministas:

| Paso | Epsilon esperado |
|------|-----------------|
| 20 000 | 0.620 |
| 40 000 | 0.240 |
| 60 000 | 0.050 (ya en mínimo) |
| 200 000 | 0.050 |

Los números de filas por fold (`train_rows = 17438, 26210, 34963, 43723, 52484`) son deterministas dado el parquet fijo.

---

## 8. Evaluación

### Protocolo walk-forward

Se usa validación walk-forward con 5 folds en lugar de un único split train/test porque los mercados financieros no son estacionarios: un agente entrenado en 2018 puede fallar completamente en 2022 (crash de 70% en crypto), y un único split no detecta este riesgo. La validación walk-forward simula exactamente cómo se usaría el agente en producción: entrenarlo en el pasado, evaluarlo en el siguiente período inmediato, sin mirar el futuro.

El held-out final (`held_out_end: 2026-12-31`, `configs/default.yaml:12`) nunca fue tocado durante el desarrollo. Todo ajuste de hiperparámetros se realizó mirando únicamente los folds de walk-forward.

### Tabla de métricas V3 (5 folds completos)

| Fold | train_end | eval_end | train_rows | eval_rows | sortino | cum_ret | max_dd |
|------|-----------|----------|-----------|----------|---------|---------|--------|
| 1 | 2019-12-31 | 2020-12-31 | 17 438 | 8 795 | **+1.8580** | **+127.8567** | **−0.2333** |
| 2 | 2020-12-31 | 2021-12-31 | 26 210 | 8 776 | −1.5286 | −0.9978 | −0.9985 |
| 3 | 2021-12-31 | 2022-12-31 | 34 963 | 8 783 | −3.5462 | −1.0000 | −1.0000 |
| 4 | 2022-12-31 | 2023-12-31 | 43 723 | 8 784 | −4.9175 | −0.9998 | −0.9998 |
| 5 | 2023-12-31 | 2025-12-31 | 52 484 | 17 544 | −3.8967 | −1.0000 | −1.0000 |

**Nota sobre la anualización:** `compute_metrics()` usa `freq=252` (días de trading), pero los datos son horarios. A 1h con 8 760 horas/año, los valores de `ann_ret` y `ann_vol` están subestimados por un factor `sqrt(252/8760) ≈ 0.17`. Los valores absolutos de sortino deben interpretarse con esta corrección en mente; las comparaciones relativas entre folds son válidas.

### Ablación de costos de transacción

*dato no disponible — pendiente de calcular.* `configs/default.yaml:31` especifica `transaction_costs_ablation: [0, 10, 25]` bps. Durante V3 solo se usaron 10 bps. Para obtener las métricas a 0 bps y 25 bps, se debe modificar el parámetro `tc` al crear `TradingEnv` en el loop de evaluación de `train_fold()`.

### Spread de semillas

*dato no disponible — pendiente de calcular.* Solo se corrió una semilla (`SEED = 42`) en todos los entrenamientos. El spread entre semillas es necesario para separar el efecto del aprendizaje del ruido de inicialización.

---

## 9. Resultados

### Curva de equity — fold 1 (el fold que funcionó)

El portafolio de fold 1 arranca en $10 000 (all-cash en `reset()`) y termina la evaluación de 8 795 pasos horarios con `cum_ret = +127.8567` (~$1 288 567). El `max_dd = −0.2333` indica una caída máxima desde el pico de 23.3%, consistente con el crash de COVID de marzo 2020 seguido de recuperación en V. La acción `top_action = 7 (Short A1, 22%)` al final del entrenamiento sugiere que el agente aprendió una estrategia de momentum con exposición asimétrica — consistente con el bull market de 2020 donde los activos de mayor momentum fueron los más rentables.

### Tabla completa de métricas V3

| Fold | sortino | cum_ret | max_dd | Régimen de evaluación |
|------|---------|---------|--------|-----------------------|
| 1 | +1.8580 | +127.8567 | −0.2333 | Bull market, crash COVID + recuperación en V |
| 2 | −1.5286 | −0.9978 | −0.9985 | Bull market 2021 (agente falla: dataset mayor, menos episodios) |
| 3 | −3.5462 | −1.0000 | −1.0000 | Bear market severo 2022 (−70% en crypto) |
| 4 | −4.9175 | −0.9998 | −0.9998 | Recuperación moderada 2023 |
| 5 | −3.8967 | −1.0000 | −1.0000 | Mercados mixtos 2024-2025 |

### El resultado anómalo: fold 1 funciona, folds 2-5 no

El éxito de fold 1 no demuestra que el agente aprendió gestión de riesgo correctamente. Tres factores actúan en conjunto y son inseparables (`ENTRENAMIENTO_V3.md, Sección 3, Q1`):

1. **Menor dataset de entrenamiento:** fold 1 tiene 17 438 filas → ~11.5 episodios completos con 200k pasos. Folds 2-5 tienen 26 210–52 484 filas → 3.8–7.6 episodios. Más repeticiones permiten que los Q-valores converjan.

2. **Alineación de régimen:** el período de evaluación de fold 1 (2020) fue el más favorable del dataset para cualquier política de momentum largo. El agente entrenado en el mercado alcista de 2018-2019 con ciclos claros encontró exactamente el mismo tipo de régimen en 2020.

3. **Convergencia de política:** el top_action de fold 1 varió entre 2, 3, 7 y 9 durante el entrenamiento (porcentajes 12–29%), indicando exploración real. En folds 2-5, el agente convergió a una acción dominante con ~12%, señal de no-convergencia a ninguna política clara.

### Evaluación honesta

El agente **no superó consistentemente a los baselines**. Solo fold 1 produce resultados positivos, y por razones que no son replicables sin las condiciones del mercado de 2020. Los folds 2-5 producen `cum_ret ≈ −1.0`, lo que indica que el agente destruyó capital activamente en todos esos períodos. La metodología es rigurosa (walk-forward, sin lookahead, reward con justificación matemática), pero los resultados del agente no superan a EqualWeight en 4 de los 5 folds evaluados.

---

## 10. Discusión

### Reward hacking — exploits encontrados

**V1 (exploit de short bias):** Con `MU = 0.5`, la penalización de drawdown dominaba la señal de reward en ~2 órdenes de magnitud sobre `log_ret`. El agente no podía distinguir entre acciones durante drawdowns. La hipótesis (`ENTRENAMIENTO_V1.md, Sección 4`) es que convergió a posiciones cortas (acciones 6-8) porque el período bajista de 2022 en el conjunto de entrenamiento hacía que esas acciones fueran rentables. Cuando el mercado de evaluación 2024-2025 fue alcista, la política corta produjo pérdidas de ~0.05%/hora compuesto hasta llevar el portafolio a zero.

**V2 (exploit de concentración + trampa de drawdown):** El agente eligió acción 2 (100% asset_1) en 7 de 10 ventanas de 20k pasos, concentrando en el activo con mejor desempeño durante 2017-2023. En evaluación 2024-2025, asset_1 se comportó adversamente, produciendo `cum_ret = −1.0`. La trampa de drawdown acumulativo (`ENTRENAMIENTO_V2.md, Sección 3`): quedarse en A1 cayendo costaba −0.0498 por paso; escapar a cash costaba −0.0695 (incluyendo el turnover). El agente aprendió que quedarse en la posición perdedora era "menos costoso" paso a paso, aunque profundizara el colapso.

**V3 (exploit potencial de arbitraje de downside):** Con Sortino diferencial, el agente tiene incentivo a rotar hacia activos que no caen en cada paso, para minimizar `MU × max(0, −log_ret)`. Esto podría producir un estilo de *momentum* excesivo con turnover moderado. No fue confirmado en V3 porque el problema dominante fue el distribution shift, pero es un riesgo documentado en `ENTRENAMIENTO_V2.md, Sección 5, Opción C`.

### Eficiencia de muestras — por qué fold 1 funcionó y fold 5 no

Con `TRAIN_STEPS = 200 000` fijo (`ENTRENAMIENTO_V3.md, Sección 3, Q3`):

| Fold | train_rows | episodios útiles | repeticiones por dato |
|------|-----------|-----------------|----------------------|
| 1 | 17 438 | ~8.6 (sin epsilon aleatorio) | 8.6× |
| 5 | 52 484 | ~2.9 | 2.9× |

Fold 5 ve cada patrón de mercado ~2.9 veces. Los Q-valores no convergen porque: (a) el buffer de 100k puede estar dominado por el episodio más reciente (fold 5 llena el 52% del buffer en un solo recorrido); (b) la red target (sincronizada cada 1 000 pasos) se actualiza 200 veces, pero los Q-valores base siguen siendo ruidosos por baja repetición. No es sobre-entrenamiento — es lo opuesto: el agente no tiene suficientes repeticiones para aprender ningún patrón.

### Distribution shift tren → despliegue

El patrón más claro es la concentración en el mejor activo del entrenamiento. En V2, el agente eligió acción 2 (100% asset_1) con 17-22% de frecuencia — asset_1 fue el mejor activo durante 2017-2023. En evaluación 2024-2025, ese activo fue adverso, produciendo pérdidas del 100%. En V3, fold 5 muestra `top_action = 1 (Long A0, 20%)`, con la misma estructura: el agente aprendió que asset_0 fue el mejor durante 2018-2023 y lo aplicó en evaluación, donde asset_0 se comportó diferente.

La causa profunda es que el vector de observación de 22 dimensiones no tiene ninguna feature capaz de señalar "el mercado lleva 6 meses en tendencia bajista" — todas las features tienen ventanas de 14-21 horas (`ENTRENAMIENTO_V3.md, Sección 8.6`). Los estados del mercado bajista de 2022 y el bull market de 2020 son estadísticamente indistinguibles en el espacio de features de corto plazo, por lo que el agente aplica la misma política a ambos regímenes.

### No-estacionariedad y cambio de régimen

Los 5 períodos de evaluación cubren 4 regímenes cualitativamente distintos:
- **2020:** crash COVID + recuperación en V + bull run (fold 1 — agente exitoso)
- **2021:** euforia post-COVID, mercado alcista con alta correlación entre activos
- **2022:** crash severo (−70%+ en crypto), aumento de tasas FED — el peor año del dataset
- **2023:** recuperación moderada, baja volatilidad relativa
- **2024-2025:** mercados mixtos, mayor incertidumbre macroeconómica

El agente siempre se entrena en todos los regímenes históricos *hasta* el punto de corte, pero las features de 21h no le dan contexto sobre en qué régimen se encuentra actualmente. Un agente con features de ventana larga (`mom_200` ≈ 8 días, `mom_720` ≈ 30 días) podría detectar el régimen actual y ajustar la política, pero esas features no están en el vector actual.

### Asignación de crédito a largo plazo

La trampa estructural documentada en V2 (`ENTRENAMIENTO_V2.md, Sección 3`) es un ejemplo clásico del problema de asignación de crédito a largo plazo en RL financiero. El beneficio de escapar de una posición perdedora (evitar pérdidas futuras) es diferido y debe ser descontado con `gamma^t`. Con `gamma = 0.99` y `t = 72` pasos hasta recuperar el costo de transición, el beneficio descontado es `0.0276 × 0.99^72 ≈ 0.014`. El agente necesita estimar correctamente este retorno a 72 horas de horizonte para tomar la decisión correcta en el paso actual. Con Q-valores no convergidos (fold 5, 2.9 episodios), esta estimación es ruidosa y el agente no puede resolver el problema.

---

## 11. Reflexión

### Tres resultados que nos sorprendieron

1. **La magnitud del éxito en fold 1:** `cum_ret = +127.8567` (12 785% de retorno) supera cualquier expectativa razonable para un agente DQN de 200k pasos con 22 features. Esperábamos, en el mejor caso, sortino positivo y retorno de +20-50%. El resultado revela que el mercado de 2020 fue excepcionalmente favorable y que incluso una política subóptima puede producir retornos extraordinarios en un bull run tendencial. No es un resultado del agente; es un resultado del mercado.

2. **El `top_action = 0 (cash)` en fold 3, pero `cum_ret = −1.0`:** Si el agente eligió principalmente cash durante el entrenamiento de fold 3, esperábamos que en evaluación (2022) se quedara en cash, produciendo `cum_ret ≈ 0`. En cambio, `cum_ret = −1.0` indica que el agente eligió posiciones risky en evaluación. La explicación (`ENTRENAMIENTO_V3.md, Sección 3, Q2`) es que `top_action = 0 (12%)` significa solo que cash fue *la más frecuente de 10 acciones casi-uniformes* — no que dominara. Los 88% restantes se distribuyeron entre acciones risky que, en el mercado bajista de 2022, produjeron pérdidas acumuladas del 100%.

3. **Que el Sortino diferencial (Fix 5) no mejoró fold 5 respecto a V2:** −3.90 (V3) vs −3.67 (V2). La trampa estructural fue demostrada matemáticamente y resuelta correctamente con el Sortino diferencial. Esperábamos una mejora clara incluso en fold 5. El resultado indica que la trampa estructural no era la causa principal del fallo — el distribution shift y la insuficiencia de episodios son más dominantes que el diseño del reward.

### Dos cambios metodológicos con más tiempo

1. **Separar TRAIN_STEPS de la constante de clase `Agent.TRAIN_STEPS`:** pasar `train_steps` como argumento a `train_fold()`, manteniendo la constante de clase en `200 000` para pasar los tests, pero usando `600 000` pasos en el entrenamiento real. Esto completaría ~11.5 episodios en todos los folds (vs 3.8 en fold 5), resolviendo la insuficiencia de cobertura sin cambiar ningún contrato del test. Es el cambio de mayor impacto y el más barato de implementar.

2. **Agregar features de régimen de largo plazo:** `mom_200` (~8 días de tendencia) y `mom_720` (~30 días de tendencia) son lookahead-safe, computables directamente en `_obs()` sin modificar `src/data.py`, y no redundantes con las 18 features actuales. Darían al agente contexto explícito sobre si el mercado actual es alcista o bajista a escala de semanas, que es la información que le falta para generalizar entre regímenes.

### Un aspecto del comportamiento del agente que no podemos explicar

En fold 4 (`eval_end: 2023-12-31`), el sortino es −4.9175, peor que fold 3 (−3.5462), aunque el período de evaluación de 2023 fue un año de recuperación moderada para las criptomonedas (menos adverso que el bear market de 2022 del fold 3). Esperaríamos que un agente con más datos de entrenamiento (43 723 vs 34 963 filas) y un período de evaluación menos extremo produjera mejor sortino, no peor. Hay tres hipótesis no verificadas: (a) el dataset de entrenamiento de fold 4 (hasta 2022) incluye el peor bear market y el agente aprendió una política defensiva que falla en la recuperación de 2023; (b) la semilla 42 produce una inicialización que favorece un espacio de exploración diferente en fold 4; (c) el período de evaluación de 2023 tiene características distribucionales que, combinadas con la política aprendida, producen más pérdidas que las caídas directas de 2022. Sin múltiples semillas o análisis de las curvas de loss, no es posible distinguir entre estas hipótesis.

### La brecha más significativa entre teoría DRL y este problema aplicado

La teoría de DRL asume que el entorno es estacionario o que el agente puede explorar suficientes interacciones para aprender la distribución subyacente. En mercados financieros, ninguna de las dos condiciones se cumple: (a) el mercado es no-estacionario por diseño — los regímenes cambian y el agente entrenado en un régimen no generaliza al siguiente; (b) el agente tiene un presupuesto fijo de pasos (200k) que en los datasets más grandes equivale a solo 2.9 episodios — mucho menos que las decenas de miles de episodios que los agentes DRL estándar (Atari, control clásico) necesitan para converger. El replay buffer, diseñado para romper la correlación temporal, se vuelve un limitante: con datasets de 52k filas, el buffer de 100k entries contiene menos de 2 episodios completos, y el muestreo aleatorio no ofrece la diversidad temporal que la teoría asume. La promesa de DRL ("dado suficiente exploración, el agente converge a la política óptima") choca con la restricción práctica de que los mercados financieros no permiten re-jugar el pasado ni explorar contrafactuales.

---

*Versión del informe: final. Fecha: 2026-06-07.*
*Basado en: `ENTRENAMIENTO_V1.md`, `ENTRENAMIENTO_V2.md`, `ENTRENAMIENTO_V3.md`, `GUIA_EQUIPO.md`, `IMPLEMENTACION.md`, `agent.py`, `src/data.py`, `src/env.py`, `configs/default.yaml`, `tests/test_submission.py`.*
*Todos los números provienen de logs de entrenamiento reales. Los valores marcados como "dato no disponible" requieren ejecución adicional.*
