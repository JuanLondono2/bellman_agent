# ENTRENAMIENTO V2 — Bellman Capital

> Registro de la segunda ejecución de entrenamiento (smoke test, fold 5).
> Documentación de resultados, diagnóstico y plan de acción para V3.

---

## 1. Configuración del entrenamiento

**Fecha de ejecución:** 2026-06-07

**Fixes aplicados desde V1** (ver diagnóstico completo en `ENTRENAMIENTO_V1.md`):

| Fix | Archivo y línea | Cambio |
|-----|-----------------|--------|
| Fix 1 | `agent.py:111` | `np.log(curr_value / ...)` → `np.log(max(curr_value, 1e-8) / ...)` — protege contra NaN |
| Fix 2 | `agent.py:50`  | `MU = 0.5` → `MU = 0.05` — rompe la espiral de drawdown |
| Fix 3 | `agent.py:266–310` | Contador `action_counts` agregado al loop; imprime `top_action=N (X%)` cada 20 000 pasos |
| Fix 4 | `agent.py:49`  | `LAMBDA = 0.1` → `LAMBDA = 0.01` — permite que el agente salga de all-cash |

**Hiperparámetros usados** (fuente: `configs/default.yaml`):

| Parámetro | Valor |
|-----------|-------|
| Algoritmo | Double DQN |
| Arquitectura | Linear(22→256→256→10) + ReLU |
| `learning_rate` | 1.0e-4 |
| `gamma` | 0.99 |
| `epsilon_start` | 1.0 |
| `epsilon_end` | 0.05 |
| `epsilon_decay_steps` | 50 000 |
| `batch_size` | 64 |
| `replay_buffer_size` | 100 000 |
| `target_update_freq` | 1 000 |
| `train_steps` | 200 000 |
| `transaction_cost_bps` | 10 |
| `initial_cash` | 10 000.0 |

**Versión de reward usada:** R3 (activa en `agent.py:122`):
```
recompensa = log_ret − 0.01 × turnover − 0.05 × drawdown
```
Con `LAMBDA = 0.01` (`agent.py:49`) y `MU = 0.05` (`agent.py:50`).

**Comando exacto ejecutado:**
```bash
uv run python agent.py
```
Esto ejecuta el bloque `__main__` de `agent.py`, que corre el smoke test del fold 5
con `train_end=2023-12-31, eval_end=2025-12-31`.

---

## 2. Resultados del smoke test (Fold 5)

| Fold | train_end | eval_end | sortino | cum_ret | max_dd | ann_ret | ann_vol |
|------|-----------|----------|---------|---------|--------|---------|---------|
| 5 | 2023-12-31 | 2025-12-31 | **−3.67** | **−1.0** | **−1.0** | **−0.2206** | **0.0814** |

**Progresión de `top_action` durante el entrenamiento (consola):**

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

**Observaciones:**
- El `top_action` ya no es 100% una sola acción (mejora respecto a V1 donde el agente era ciego).
- Acción 2 (100% asset_1) domina en 7 de los 10 intervalos de 20 000 pasos.
- Los porcentajes nunca superan el 22%, lo que indica exploración genuina.
- `cum_ret = −1.0` y `max_dd = −1.0` persisten: el portafolio llegó a cero.

---

## 3. Diagnóstico

### Q1 — ¿Qué es asset_1 y qué retorna `load_prices()`?

**Respuesta basada exclusivamente en lectura de `src/data.py`:**

`load_prices()` (`src/data.py:20–27`) lee el archivo `data/raw/prices_1h.parquet` y retorna
un `pd.DataFrame` con todas las columnas OHLCV en crudo tal como están en el parquet.
La función no filtra ni renombra: devuelve el DataFrame completo directamente con
`pd.read_parquet(path)`.

La columna `asset_1_close` se origina en `src/data.py:70`:
```python
close = data[f"{asset}_close"]   # asset = "asset_1"
```
y luego se extrae explícitamente en `close_prices()` (`src/data.py:32–33`):
```python
cols = [f"{a}_close" for a in RISKY_ASSETS] + ["cash"]
return data[cols].rename(columns={f"{a}_close": a for a in RISKY_ASSETS})
```
Es decir, `asset_1_close` se renombra a `asset_1` en el DataFrame de precios que recibe
`BaseTradingEnv`.

**Lo que el código no permite determinar:** el nombre real del activo de mercado que
corresponde a `asset_1`. La constante `RISKY_ASSETS = ["asset_0", "asset_1", "asset_2"]`
(`src/data.py:15`) usa nombres genéricos. Sin leer el archivo parquet o el script
`scripts/download_data.py`, no es posible saber si asset_1 es BTC, ETH u otro instrumento.

**Lo que sí revela el patrón de entrenamiento:** el agente eligió acción 2 (100% asset_1)
como acción dominante en 7 de los 10 intervalos. Esto significa que asset_1 produjo los
mejores rewards durante el período de entrenamiento (datos hasta 2023-12-31). El resultado
`cum_ret = −1.0` en el período de evaluación (2024-2025) indica que asset_1 tuvo un
comportamiento muy negativo en ese período, lo opuesto a lo aprendido.

---

### Q2 — Cálculo numérico de R3 con acción 2 constante y asset_1 cayendo 99%

**Supuestos:**
- Portfolio inicial: $10 000
- Asset_1 cae 99% en 17 544 pasos de evaluación (hourly)
- Factor por paso: `exp(ln(0.01) / 17544) = exp(−0.000263) ≈ 0.999737`
- Acción elegida siempre: 2 → pesos `[0.00, 1.00, 0.00, 0.00]`
- Estado inicial de pesos: `[0, 0, 0, 1]` (all-cash)
- `tc = 10 bps = 0.001`, `LAMBDA = 0.01`, `MU = 0.05`

---

**Paso 1** (primera acción: transición de all-cash a 100% asset_1):

```
turnover   = |0−0| + |1−0| + |0−0| + |0−1| = 2.0
curr_value = 10000 × 0.999737 − 10000 × 2.0 × 0.001
           = 9997.37 − 20.00 = 9977.37
log_ret    = ln(max(9977.37, 1e-8) / (10000 + 1e-8) + 1e-8)
           = ln(0.997737 + 1e-8) ≈ −0.002266
peak_value = max(10000, 9977.37) = 10000
drawdown   = (10000 − 9977.37) / 10000 = 0.002263

R3 = −0.002266 − 0.01 × 2.0 − 0.05 × 0.002263
   = −0.002266 − 0.020000 − 0.000113
   = −0.022379
```

El costo de transacción domina en el primer paso.

---

**Paso 2** (misma acción, sin rebalanceo, pesos ya son `[0,1,0,0]`):

```
turnover   = |0−0| + |1−1| + |0−0| + |0−0| = 0.0
curr_value = 9977.37 × 0.999737 = 9974.75
log_ret    = ln(9974.75 / 9977.37 + 1e-8) ≈ −0.000263
peak_value = 10000  (no cambia: 9974.75 < 10000)
drawdown   = (10000 − 9974.75) / 10000 = 0.002525

R3 = −0.000263 − 0.01 × 0 − 0.05 × 0.002525
   = −0.000263 − 0.000000 − 0.000126
   = −0.000389
```

---

**Paso ~8 772** (mitad del episodio, portfolio ≈ $1 000, drawdown ≈ 90%):

```
curr_value ≈ 10000 × exp(−0.000263 × 8772) ≈ 10000 × 0.1 = 1000
log_ret    ≈ −0.000263
turnover   = 0
drawdown   = (10000 − 1000) / 10000 = 0.90

R3 = −0.000263 − 0 − 0.05 × 0.90
   = −0.000263 − 0.04500
   = −0.045263
```

La penalización de drawdown supera al log_ret en un factor de **171×**.

---

**Paso ~17 500** (fin del episodio, portfolio ≈ $100, drawdown ≈ 99%):

```
curr_value ≈ 100
log_ret    ≈ −0.000263
turnover   = 0
drawdown   = (10000 − 100) / 10000 = 0.99

R3 = −0.000263 − 0 − 0.05 × 0.99
   = −0.000263 − 0.04950
   = −0.049763
```

La penalización de drawdown supera al log_ret en un factor de **189×**.

**Conclusión del cálculo:** aun con `MU = 0.05`, una vez que el portafolio entra en drawdown
severo, el término `MU × drawdown` domina la función de reward en ~2 órdenes de magnitud
sobre el `log_ret` horario. El agente no puede distinguir entre acción 2 (que cae) y
acción 0 (all-cash, que detendría las pérdidas) porque **ambas reciben R3 ≈ −0.05**: acción 2
da `−0.0498` y cambiar a acción 0 da `−0.02 (turnover) − 0.0495 (drawdown) = −0.0695`.
Irónicamente, el costo de transacción de CAMBIAR a cash (`−0.02`) hace que quedarse en A1
(`−0.0498`) parezca mejor a corto plazo.

---

## 4. Hipótesis principal

La causa más probable del `cum_ret = −1.0` es que el agente aprendió durante el
entrenamiento (datos hasta 2023-12-31) que asset_1 producía los mejores retornos, y
convergió a una política sesgada hacia acción 2 (100% asset_1) con ~17-22% de frecuencia
greedy. Durante la evaluación en 2024-2025, asset_1 tuvo un comportamiento adverso que
llevó al portafolio a una pérdida acumulada del 100%. El problema estructural que persiste
tras los fixes de V2 es que la función R3, una vez que el portafolio está en drawdown
profundo (>50%), hace que el costo de rebalanceo (`LAMBDA × turnover = 0.01 × 2.0 = 0.02`)
supere temporalmente al beneficio esperado de escapar a cash (`log_ret ≈ 0 vs −0.02`),
por lo que el agente aprende a no salir de las posiciones perdedoras. El Fix 2 redujo
`MU` de 0.5 a 0.05, lo que mejoró el sortino de −5.49 a −3.67, pero el problema de fondo
— el término de drawdown que crece con el deterioro del portafolio — no fue eliminado.

---

## 5. Fixes propuestos para V3

> **Premisa matemática compartida por las tres opciones.**
> El término de drawdown es idéntico para *cualquier* acción en un paso dado, porque
> depende del pico histórico del portafolio y del valor actual, no de la acción elegida.
> Por tanto, **nunca puede resolver la trampa por sí solo**: solo el término de turnover
> (`LAMBDA × turnover`) y el `log_ret` determinan si el agente prefiere quedarse o cambiar.
> La trampa estructural se rompe únicamente cuando el beneficio descontado de escapar
> supera el costo de transacción de la acción de escape.

---

### Opción A — Cambio mínimo: `MU = 0.0`, `LAMBDA = 0.1` (riesgo: medio)

**Archivos y líneas:** `agent.py:49` y `agent.py:50`.

```python
# Antes (V2):
LAMBDA = 0.01
MU     = 0.05

# Después (Opción A):
LAMBDA = 0.1
MU     = 0.0
```

**Comportamiento esperado:** al eliminar el término de drawdown, el reward queda como
`R3 = log_ret − 0.1 × turnover`. La señal de aprendizaje es proporcional al retorno
del portafolio en cada paso, sin distorsión acumulativa por el pico histórico.

**Verificación de la trampa estructural con drawdown = 0.90, portfolio ≈ $1 000:**

```
Quedarse en A1:
  log_ret  = −0.000263
  turnover = 0
  R3 = −0.000263 − 0.1 × 0 − 0 = −0.000263

Cambiar a cash:
  log_ret  ≈ 0
  turnover = 2.0
  R3 = 0 − 0.1 × 2.0 − 0 = −0.200
```

**La trampa PERSISTE:** cambiar a cash cuesta −0.200 vs quedarse en A1 con −0.000263.
El costo de transacción inicial es ~760× mayor que el beneficio por paso de abandonar A1.
Con `gamma = 0.99`, el beneficio descontado de salir es `0.000263 × 100 = 0.0263 << 0.200`.

**Nuevo exploit que puede aparecer:** cash parking. Con `LAMBDA = 0.1`, salir de all-cash
a cualquier posición de riesgo cuesta `0.1 × 2.0 = 0.2` en el paso de transición, valor
inalcanzable por el log_ret horario típico (~0.0003). El agente aprende tempranamente que
moverse desde cash es costoso y puede converger a la política trivial de no hacer nada.
Este exploit fue observado en V1 con los mismos coeficientes.

**Riesgo medio**: elimina la espiral de drawdown, pero sustituye una trampa por otra.

---

### Opción B — Rebalanceo de penalizaciones: `MU = 0.1`, `LAMBDA = 0.05` (riesgo: alto)

**Archivos y líneas:** `agent.py:49` y `agent.py:50`.

```python
# Antes (V2):
LAMBDA = 0.01
MU     = 0.05

# Después (Opción B):
LAMBDA = 0.05
MU     = 0.1
```

**Verificación de la trampa estructural con drawdown = 0.90, portfolio ≈ $1 000:**

```
Quedarse en A1:
  log_ret  = −0.000263
  turnover = 0
  drawdown = 0.90
  R3 = −0.000263 − 0.05 × 0 − 0.1 × 0.90
     = −0.000263 − 0.000 − 0.090
     = −0.090263

Cambiar a cash:
  log_ret  ≈ 0
  turnover = 2.0
  drawdown = 0.90  ← idéntico: el pico es 10 000, el portfolio es $1 000 sin importar la acción
  R3 = 0 − 0.05 × 2.0 − 0.1 × 0.90
     = −0.100 − 0.090
     = −0.190
```

**La trampa PERSISTE y empeora:** cambiar a cash cuesta −0.190 vs −0.090 de quedarse.
El margen de diferencia es 0.100 (solo el turnover). El beneficio descontado de salir
es `0.000263 × 100 = 0.0263`, muy inferior al costo de transición de 0.100.

Comparando con V2 (`LAMBDA = 0.01`): el costo de escape era 0.020; aquí sube a 0.100.
**La Opción B es estrictamente peor que V2** para resolver la trampa estructural.

**Riesgo alto**: agrava el problema en lugar de resolverlo.

---

### Opción C — Rediseño de reward: Sortino diferencial (riesgo: bajo) ✓ RECOMENDADA

**Archivos y líneas:** `agent.py:50` (eliminar MU como constante de clase) y
`agent.py:107–122` (reemplazar la función `_reward()` completa).

**Principio:** reemplazar la penalización de drawdown (señal acumulativa, no estacionaria,
idéntica para todas las acciones) por una penalización del retorno negativo del paso actual
(señal estacionaria, proporcional al daño real, diferente para cada acción).

**Fórmula:**
```
downside = max(0, −log_ret)          ← solo pasos con retorno negativo
R3 = log_ret − LAMBDA × turnover − MU × downside
```

Con `LAMBDA = 0.01` y `MU = 0.05` (mismos valores que V2, sin cambio de coeficientes).

**Implementación en `agent.py:107–122`:**
```python
def _reward(self, prev_value: float, curr_value: float) -> float:
    log_ret  = float(np.log(max(curr_value, 1e-8) / (prev_value + 1e-8) + 1e-8))
    turnover = self._last_turnover
    downside = max(0.0, -log_ret)   # penaliza solo retornos negativos

    # R1: pure log-return
    # return log_ret

    # R2: log-return minus turnover penalty
    # return log_ret - self.LAMBDA * turnover

    # R3: Sortino diferencial — penaliza retorno negativo del paso, no distancia al pico
    return log_ret - self.LAMBDA * turnover - self.MU * downside
```

Nota: eliminar `self._peak_value` del método (ya no se necesita). Si `BaseTradingEnv`
lo inicializa, no hay conflicto — simplemente deja de usarse.

**Verificación de la trampa estructural con drawdown = 0.90, portfolio ≈ $1 000:**

```
Quedarse en A1 (asset_1 sigue cayendo 0.026%/paso):
  log_ret  = −0.000263
  turnover = 0
  downside = max(0, 0.000263) = 0.000263
  R3 = −0.000263 − 0.01 × 0 − 0.05 × 0.000263
     = −0.000263 − 0.000013
     = −0.000276

Cambiar a cash (log_ret ≈ 0, cash no tiene retorno):
  log_ret  ≈ 0
  turnover = 2.0
  downside = max(0, 0) = 0
  R3 = 0 − 0.01 × 2.0 − 0.05 × 0
     = −0.020

Pasos siguientes en cash (ya no hay rebalanceo):
  log_ret  = 0,  turnover = 0,  downside = 0
  R3 = 0
```

**La trampa SE ROMPE.** Comparación:

| Escenario | R3 inmediato | R3 por paso (futuro) | Costo de transición |
|-----------|-------------|---------------------|---------------------|
| Quedarse en A1 | −0.000276 | −0.000276/paso (perpetuo) | — |
| Cambiar a cash | −0.020 (solo el paso de cambio) | 0 (cash estable) | −0.020 |

Beneficio descontado de escapar: `0.000276 × Σ(0.99^t) ≈ 0.000276 × 100 = **0.0276**`

Costo de transición: **0.020**

Como `0.0276 > 0.020`, el agente recupera el costo de cambio en ~72 pasos (`0.020 / 0.000276`).
Con `gamma = 0.99` y planificación de largo plazo, cambiar a cash es la acción racional.

**Por qué funciona:** el término `downside` penaliza cada paso con retorno negativo
proporcionalmente al daño real de ese paso. Si el agente cambia a cash, la penalización
de downside desaparece inmediatamente porque `log_ret = 0`. Si el agente se queda en A1,
acumula una penalización de 0.000276 cada hora indefinidamente. A diferencia del drawdown,
este término **es diferente para cada acción**: quedarse en A1 tiene downside > 0, cambiar
a cash tiene downside = 0.

**Nuevo exploit posible:** el agente podría aprender a hacer *arbitraje de downside* —
vender activos que han subido marginalmente para evitar steps con log_ret < 0. Este riesgo
es mitigado por el `LAMBDA × turnover` que penaliza el rebalanceo excesivo.

**Riesgo bajo:** no requiere cambiar los coeficientes (mismos LAMBDA=0.01 y MU=0.05 de V2),
solo reemplaza la fuente de la penalización. Retrocompatible con todos los tests existentes.

---

### Tabla comparativa de las tres opciones

| | Opción A | Opción B | Opción C (Recomendada) |
|--|---------|---------|----------------------|
| `LAMBDA` | 0.1 | 0.05 | 0.01 |
| `MU` | 0.0 | 0.1 | 0.05 |
| Tipo de penalización | Solo turnover | Turnover + drawdown | Turnover + downside diferencial |
| ¿Rompe la trampa? | No (turnover alto) | No (empeora) | **Sí** |
| Nuevo exploit | Cash parking | — | Downside arbitrage (mitigado) |
| Archivos modificados | `agent.py:49-50` | `agent.py:49-50` | `agent.py:107–122` |
| Riesgo | Medio | Alto | **Bajo** |

---

### Recomendación: Opción C para V3

La Opción C es la única que rompe la trampa estructural demostrada matemáticamente.
Las opciones A y B manipulan los mismos coeficientes sin cambiar la fuente del problema:
el drawdown acumulativo es siempre idéntico para cualquier acción, por lo que jamás puede
diferenciar entre quedarse en una posición perdedora y escapar de ella.

La Opción C preserva los coeficientes ya calibrados en V2 (`LAMBDA=0.01`, `MU=0.05`) y
cambia únicamente qué se penaliza: en lugar de la distancia al pico histórico (señal
no estacionaria), penaliza el retorno negativo del paso actual (señal estacionaria y
acción-dependiente). Este cambio alinea la reward con la definición del Sortino ratio
que se usa en evaluación, y hace que el agente tenga incentivo concreto para cortar
pérdidas en lugar de quedarse en posiciones en caída libre.

---

## 6. Próximos pasos

Orden de ejecución para el entrenamiento V3:

1. **Implementar Opción C** en `agent.py:107–122`: reemplazar `_reward()` con la
   formulación de Sortino diferencial. Eliminar el uso de `self._peak_value` del método.

2. **Verificar que los 26 tests siguen pasando:**
   ```bash
   uv run pytest tests/test_submission.py -v
   ```

3. **Correr el smoke test (solo fold 5)** y verificar en consola:
   - No aparece `nan`
   - `top_action` no supera el 50% en ningún intervalo
   - `cum_ret` mejora respecto a −1.0

4. **Eliminar checkpoints de V2** antes del entrenamiento V3 completo:
   ```powershell
   Remove-Item models\checkpoints\* -Force
   Remove-Item models\metrics\* -Force
   ```

5. **Correr los 5 folds completos:**
   ```bash
   uv run python agent.py
   ```

6. **Documentar resultados en `ENTRENAMIENTO_V3.md`** con la misma estructura,
   comparando sortino fold por fold con V1 y V2.

---

*Versión: V2 (actualizada con análisis de opciones V3). Autor: equipo. Fecha: 2026-06-07.*
*Código base analizado: `agent.py` (líneas 29–122, 264–310), `src/data.py` (líneas 15–33), `configs/default.yaml`.*
