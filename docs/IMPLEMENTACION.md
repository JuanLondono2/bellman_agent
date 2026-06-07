# IMPLEMENTACION — Bellman Capital

> Guía técnica para el equipo. Explica **qué se construyó, cómo funciona y por qué se tomó cada decisión**, con referencia exacta al código fuente. Nivel: alguien que entiende machine learning pero no ha leído el código.

---

## 1. Qué se construyó

El proyecto tiene dos capas: **infraestructura provista** (no se modifica) y **código del equipo** (el único archivo que se entrega).

**Infraestructura provista:**
- `src/env.py` — Clase `BaseTradingEnv`, que maneja la mecánica del portafolio: precios, pesos, costos de transacción. Es el "motor" del entorno. No se toca.
- `src/data.py` — Funciones `load_prices()`, `split()` y `build_features()`. Carga los datos históricos, los parte en train/eval sin solapamiento, y construye las 18 features por activo.
- `src/baselines.py` — Cinco estrategias de comparación simples: `RandomPolicy`, `HoldCash`, `HoldAsset0`, `EqualWeight`, `SMA`. Se usan para saber si el agente aprendió algo útil.
- `src/metrics.py` — Función `compute_metrics()`. Calcula seis métricas de desempeño sobre una curva de portafolio: retorno acumulado, retorno anualizado, volatilidad, Sharpe, **Sortino** (métrica primaria), y drawdown máximo.
- `src/base.py` — Clase abstracta `BaseAgent`. Define la interfaz mínima que todo agente debe implementar: `train()` y `act()`.
- `configs/default.yaml` — Todos los hiperparámetros del agente y la definición de los 5 folds de validación walk-forward.

**Código del equipo (`agent.py`):**
- Clase `TradingEnv` — subclase de `BaseTradingEnv`. Implementa qué ve el agente (`_obs`), qué hace (`_weights_from_action`) y qué recibe como recompensa (`_reward`).
- Clase `QNetwork` — red neuronal MLP de dos capas ocultas que estima el valor de cada acción.
- Clase `Agent` — agente DQN con extensión Double DQN. Incluye buffer de replay, red online, red target, y loop de entrenamiento.
- Funciones `_run_episode()` y `train_fold()` — orquestan la carga de datos, el split correcto del scaler, el entrenamiento y la evaluación de un fold completo. `_run_episode()` retorna el arreglo de valores del portafolio **y** la recompensa total acumulada del episodio. `train_fold()` guarda automáticamente checkpoints del modelo, curva de loss y métricas de evaluación en la carpeta `models/`.

**Tests y documentación:**
- `tests/test_submission.py` — 26 tests que deben pasar antes de entregar. Verifican contratos de acciones, forma del vector de observación, ausencia de NaN, y propiedades del agente.
- `GUIA_EQUIPO.md` — Documento de decisiones de diseño con justificación financiera y de RL para cada componente.

---

## 2. TradingEnv — Cómo funciona el entorno

El entorno es el "mundo" en el que vive el agente. En cada paso de tiempo, el agente recibe información del mercado, decide cómo distribuir su capital, y recibe una recompensa según qué tan bien le fue.

### ¿Qué pasa en `reset()`?

`reset()` reinicia el entorno al principio del período de datos. Se definen en `agent.py, clase TradingEnv, función reset(), línea ~75`:

| Variable | Valor inicial | Significado |
|----------|--------------|-------------|
| `self._t` | `21` (`self._lookback`) | Índice temporal de inicio. Salta las primeras 21 filas donde las features tienen NaN. |
| `self._value` | `10 000.0` | Capital inicial en dólares (definido en `configs/default.yaml`, línea 15). |
| `self._weights` | `[0, 0, 0, 1]` | Portafolio inicial: 100% en cash, 0% en cada activo de riesgo. |
| `self._peak_value` | `10 000.0` | Máximo histórico del portafolio, usado para calcular drawdown. |
| `self._last_turnover` | `0.0` | Cantidad de rebalanceo del último paso, usado en la recompensa. |

`BaseTradingEnv.reset()` (en `src/env.py, línea 42`) llama a `self._obs()` antes de retornar, así el agente recibe inmediatamente su primera observación.

### ¿Qué pasa en cada `step()`?

La secuencia completa ocurre en `src/env.py, función step(), líneas 49–68`:

**1. Obtener los pesos nuevos del agente**
```
w = self._weights_from_action(action)   # src/env.py, línea 50
```
En `agent.py, función _weights_from_action(), línea ~95`, se consulta la tabla `_WEIGHT_MENU` con el índice de acción. **Además**, aquí capturamos `self._last_turnover` (el cambio total de pesos), porque en este momento todavía tenemos los pesos viejos `self._weights`.

**2. Calcular el turnover (cuánto se rebalanceó)**
```
turnover = |w_nuevo - w_viejo|.sum()    # src/env.py, línea 57
```
*Turnover* es la suma de los cambios absolutos en cada peso. Pasar de 100% cash a 33%/33%/33% tiene turnover = 2.0 (el capital completo rota dos veces).

**3. Calcular el retorno de precios**
```
ret = precios[t] / precios[t-1]         # src/env.py, línea 58
```
Un vector de 4 elementos: cuánto subió cada activo (y el cash, que siempre es 1.0).

**4. Actualizar el valor del portafolio**
```
nuevo_valor = viejo_valor × dot(w, ret) − viejo_valor × turnover × tc
              ─────────────────────────   ──────────────────────────────
              ganancia/pérdida por precios    costo de transacción
```
En palabras: *"tu capital crece según el retorno ponderado de los activos que tienes, y luego se descuenta el costo de rebalancear"*. El costo es del 0.01% (10 bps) por unidad de turnover. Definición exacta en `src/env.py, línea 60`.

**5. Avanzar el reloj y verificar si terminó el episodio**
```
self._t += 1
terminated = self._t >= len(self.prices)   # src/env.py, líneas 62–64
```

**6. Construir la observación y calcular la recompensa**
Se llama a `_obs()` y `_reward()` del subclase (`agent.py`), y se retorna todo al agente.

### ¿Qué es `_lookback` y por qué importa?

`_lookback = 21` (definido en `agent.py, TradingEnv.__init__(), línea ~57`).

La función `build_features()` en `src/data.py, línea 87` calcula la volatilidad rodante de 21 períodos (`vol_21`). Esto requiere 21 log-retornos válidos. El primer log-retorno válido está en la fila 1 (la fila 0 es NaN porque no hay precio anterior). Por lo tanto, `vol_21` produce NaN en las filas 0–20 y su primer valor válido está en la fila 21. La función `build_features()` elimina estas filas con `.dropna()` en `src/data.py, línea 95`.

Si el agente empezara en `t=0`, intentaría leer una feature que no existe. Al iniciar en `t=21`, el primer dato que lee ya tiene sus 18 features calculadas correctamente.

---

## 3. El vector de observación — Qué ve el agente

El agente recibe en cada paso un vector de **22 números** (`agent.py, TradingEnv._obs(), línea ~81`). Todos los valores de features están **normalizados** (media 0, desviación estándar 1) por un `StandardScaler` entrenado sólo en datos de entrenamiento.

### Las 18 features de mercado (índices 0–17)

Se calculan en `src/data.py, función build_features(), líneas 69–93`, para cada uno de los 3 activos de riesgo:

| Índice | Nombre | Cálculo | Significado económico |
|--------|--------|---------|----------------------|
| 0 | `asset_0_log_ret` | `log(close_t / close_{t-1})` | Retorno logarítmico del período actual. Es la señal más directa del movimiento del precio. |
| 1 | `asset_0_vol_21` | `std(log_ret, ventana=21)` | Volatilidad realizada de las últimas 21 horas. Captura el fenómeno de *volatility clustering*: períodos de alta volatilidad tienden a seguirle períodos de alta volatilidad (efecto GARCH). |
| 2 | `asset_0_mom_20` | `log(close_t / close_{t-20})` | Momentum de 20 períodos. Señal de tendencia a medio plazo: precios que subieron tienden a seguir subiendo (momentum, evidencia empírica robusta). |
| 3 | `asset_0_atr_14` | `ATR(14) / close` | Rango verdadero promedio normalizado. Mide la volatilidad intradía considerando gaps de precio. Es un proxy del riesgo de mercado sin sesgo de escala. |
| 4 | `asset_0_vol_ratio` | `volume / media_rodante(volume, 21)` | Volumen relativo al promedio reciente. Picos de volumen suelen preceder movimientos grandes de precio (teoría de microestructura). |
| 5 | `asset_0_tbr` | `taker_buy_ratio` (directamente de los datos) | Fracción del volumen iniciada por compradores. Valor > 0.5 indica presión compradora neta. Es una señal de microestructura de corto plazo. |
| 6–11 | `asset_1_*` | Igual que 0–5 | Las mismas 6 features para asset_1. |
| 12–17 | `asset_2_*` | Igual que 0–5 | Las mismas 6 features para asset_2. |

### Los 4 pesos actuales del portafolio (índices 18–21)

| Índice | Nombre | Valor ejemplo |
|--------|--------|--------------|
| 18 | `w_asset_0` | 0.33 |
| 19 | `w_asset_1` | 0.33 |
| 20 | `w_asset_2` | 0.33 |
| 21 | `w_cash` | 0.00 |

### Por qué incluir los pesos actuales (condición de Markov)

*Condición de Markov*: el estado del sistema debe contener toda la información necesaria para tomar la decisión óptima, sin necesidad de conocer el historial.

Sin los pesos actuales, el agente no puede calcular el costo de rebalancear. Si está 100% en cash y quiere ir a 100% en asset_0, el turnover es 2.0 y paga 20 bps en costos. Si ya está 100% en asset_0, el costo es 0. La misma observación de mercado lleva a decisiones óptimas diferentes según la posición actual, por lo tanto la posición debe ser parte del estado.

### El riesgo de lookahead y cómo se evitó

*Lookahead* significa usar información del futuro durante el entrenamiento, lo que hace los resultados artificialmente buenos y luego falla en producción.

El riesgo principal es el `StandardScaler`. Si se llama `scaler.fit()` sobre datos que incluyen el período de evaluación, el scaler "ve" la media y desviación estándar futura y normaliza el pasado con esa información. Esto es trampa.

**Cómo se evita:** en `agent.py, función train_fold(), línea ~288` (marcada con un comentario explícito):
```python
# LOOKAHEAD BOUNDARY — scaler.fit() en TRAIN DATA ONLY.
train_features, scaler = build_features(train_data, fit=True)  # <-- fit aquí solo
eval_features, _       = build_features(eval_data, scaler=scaler)  # sin re-fit
```

El scaler se entrena sólo con el conjunto de entrenamiento (`fit=True`), y luego se aplica sin reentrenar al conjunto de evaluación. El parámetro `fit` está implementado en `src/data.py, build_features(), líneas 97–101`.

---

## 4. El menú de acciones — Qué puede hacer el agente

El agente elige en cada paso una de **10 carteras predefinidas**. Esta es la variable `_WEIGHT_MENU` en `agent.py, líneas 25–36`. Los pesos son `[asset_0, asset_1, asset_2, cash]` y deben sumar 1.

| ID | Nombre | Pesos `[a0, a1, a2, cash]` | Interpretación económica |
|----|--------|---------------------------|--------------------------|
| 0 | All Cash | `[0.00, 0.00, 0.00, 1.00]` | Refugio total. Sin exposición al mercado. El capital no crece ni decrece por precios. |
| 1 | Long A0 | `[1.00, 0.00, 0.00, 0.00]` | Apuesta concentrada en asset_0. Máximo riesgo/retorno en un solo activo. |
| 2 | Long A1 | `[0.00, 1.00, 0.00, 0.00]` | Apuesta concentrada en asset_1. |
| 3 | Long A2 | `[0.00, 0.00, 1.00, 0.00]` | Apuesta concentrada en asset_2. |
| 4 | Equal Weight | `[0.33, 0.33, 0.33, 0.00]` | Diversificación máxima entre los tres activos. Benchmark pasivo. |
| 5 | Conservative | `[0.25, 0.25, 0.25, 0.25]` | Diversificado con 25% en cash como amortiguador de drawdown. |
| 6 | Short A0 | `[-0.50, 0.50, 0.50, 0.50]` | Apuesta bajista en asset_0. Gana si asset_0 cae. |
| 7 | Short A1 | `[0.50, -0.50, 0.50, 0.50]` | Apuesta bajista en asset_1. |
| 8 | Short A2 | `[0.50, 0.50, -0.50, 0.50]` | Apuesta bajista en asset_2. |
| 9 | Long A1+A2 | `[0.00, 0.50, 0.50, 0.00]` | Portafolio largo excluyendo asset_0. |

**Contrato con los baselines (crítico):** las clases `HoldCash`, `HoldAsset0` y `EqualWeight` en `src/baselines.py, líneas 11–19` tienen hardcodeado que acción 0 = todo cash, acción 1 = todo asset_0, acción 4 = equal weight. Si se reordena el menú, los baselines comparan contra el portafolio equivocado y las métricas son inútiles.

### Por qué se incluyen posiciones cortas

Una posición corta (peso negativo) significa que el agente apuesta a que ese activo va a bajar: pide prestado el activo, lo vende, y espera comprarlo más barato para devolverlo.

Se incluyen porque amplían el espacio de estrategias posibles. Un mercado bajista genera oportunidades que un portafolio solo-largo no puede aprovechar.

**Simplificación importante:** en la realidad, las posiciones cortas tienen un costo de financiamiento diario (*funding rate*). Este modelo **no lo cobra** (documentado en `src/env.py, líneas 32–34`). Esto significa que el agente puede sobreutilizar posiciones cortas porque son "gratis". Al reportar resultados, se debe mencionar esta simplificación.

---

## 5. La función de recompensa — Qué optimiza el agente

La *recompensa* (reward) es la señal que le dice al agente qué tan bien hizo en cada paso. En `agent.py, TradingEnv._reward(), líneas ~100–113` están implementadas las tres versiones, con R1 y R2 comentadas y R3 activa.

### R1 — Log-retorno puro (baseline, se espera que falle)

```
recompensa = log(valor_nuevo / valor_viejo)
```

**Comportamiento esperado:** el agente maximiza retorno sin ningún costo. Aprende a hacer *churn*: intercambiar entre posiciones opuestas en cada paso, porque el horizonte corto no le permite ver que los costos de transacción acumulados destruyen el retorno. También tiende a concentrarse en el activo que subió más recientemente.

**Cómo detectarlo:** el campo `info["turnover"]` promedia > 1.5 por paso. Las curvas de capital tienen volatilidad extrema en entrenamiento pero colapsan en evaluación.

### R2 — Log-retorno menos penalización de turnover

```
recompensa = log(valor_nuevo / valor_viejo) − 0.1 × turnover
```

El factor `0.1` es `LAMBDA`, definido como constante de clase en `agent.py, línea ~44`.

**Comportamiento esperado:** el agente aprende a no rebalancear tanto. El problema es que si `LAMBDA` es muy alto, el agente aprende que la acción óptima es no hacer nada: se queda permanentemente en la acción del primer paso (turnover = 0, recompensa constante cercana a 0).

**Cómo detectarlo:** más del 90% de los pasos toman la misma acción. El `sortino` del agente es menor que el del baseline `HoldCash`.

### R3 — Formulación final (activa)

```
recompensa = log(valor_nuevo / valor_viejo) − 0.1 × turnover − 0.5 × drawdown
```

Los coeficientes son `LAMBDA = 0.1` y `MU = 0.5`, constantes de clase en `agent.py, líneas 44–45`.

El *drawdown* es cuánto estamos por debajo del máximo histórico del portafolio: `(pico - valor_actual) / pico`. Solo penaliza cuando el portafolio está en pérdida respecto a su máximo.

**Ejemplo concreto con números:**
- El portafolio pasó de 1 000 a 1 010 (subió 1%).
- El agente cambió su posición en 40% del capital total (turnover = 0.4).
- El pico histórico del portafolio fue 1 060. Actualmente estamos en 1 010, que es 4.7% por debajo.

Entonces:
```
log_ret  = log(1010 / 1000) ≈ 0.00995
turnover_penalty  = 0.1 × 0.4 = 0.040
drawdown = (1060 − 1010) / 1060 ≈ 0.0472
drawdown_penalty  = 0.5 × 0.0472 = 0.0236

R3 = 0.00995 − 0.040 − 0.0236 = −0.0537
```

A pesar de ganar 1%, la recompensa es negativa porque el agente rebalanceó demasiado y sigue en drawdown. Esto lo entrena a ser más conservador con los costos y a proteger el capital.

**Comportamiento esperado:** el agente aprende a explotar tendencias claras (retorno positivo), reducir rotación innecesaria, y evitar posiciones que llevan a drawdowns profundos. **Exploit posible:** puede aprender a quedarse 100% en cash, donde el drawdown es siempre 0 y el turnover es 0 si no cambia. Señal de alerta: `sortino` del agente < `sortino` de `HoldCash`.

---

## 6. El Agente — Cómo funciona DQN

### La Q-Network — La "memoria" del agente

Una *Q-Network* es una red neuronal que, dado el estado actual del mercado, estima cuán buena es cada acción posible. La arquitectura está en `agent.py, clase QNetwork, líneas ~120–132`:

```
Entrada: 22 números (el vector de observación)
   → Capa densa: 256 neuronas + ReLU
   → Capa densa: 256 neuronas + ReLU
   → Capa densa: 10 neuronas (una por acción)
Salida: 10 números (el valor estimado Q de cada acción)
```

*ReLU* (Rectified Linear Unit) es la función de activación: `max(0, x)`. Introduce no-linealidad y es eficiente de calcular. El agente escoge la acción con el mayor valor Q.

### El Replay Buffer — Romper la correlación temporal

El *replay buffer* (`agent.py, clase Agent.__init__(), línea ~167`) es una cola de los últimos 100 000 pasos `(observación, acción, recompensa, siguiente_observación, terminado)`.

**Por qué existe:** los datos de series de tiempo son temporalmente correlacionados. Si el agente aprende de pasos consecutivos (paso 1, 2, 3, ...), los gradientes de la red están sesgados por la correlación. Es como intentar entrenar un modelo de imágenes con 1000 fotos del mismo gato seguidas y luego 1000 de un perro — aprende secuencias, no patrones.

Al **muestrear aleatoriamente** 64 transiciones del buffer en cada actualización (`agent.py, Agent.train(), línea ~235`), los datos de entrenamiento son independientes entre sí y el aprendizaje es estable.

### Epsilon-greedy — Exploración vs explotación

El parámetro `epsilon` controla qué tan seguido el agente hace cosas aleatorias vs qué tan seguido hace la mejor acción conocida (`agent.py, propiedad epsilon, línea ~171`):

| Valor de epsilon | Comportamiento |
|-----------------|---------------|
| `1.0` (inicio) | 100% aleatorio. El agente prueba acciones al azar para descubrir qué funciona. |
| `0.05` (final) | 5% aleatorio, 95% greedy. El agente mayormente explota lo que aprendió. |
| `0.0` (evaluación) | 100% greedy. Nunca explora. Se usa al evaluar el agente entrenado. |

`epsilon` decae **linealmente** de 1.0 a 0.05 durante los primeros 50 000 pasos de entrenamiento, según `EPSILON_DECAY_STEPS` en `configs/default.yaml, línea 24`. Después de esos 50k pasos, `epsilon` se mantiene fijo en 0.05.

La analogía: un chef nuevo (epsilon=1.0) prueba todos los ingredientes al azar para aprender cuáles combinar. Con experiencia (epsilon=0.05), sigue mayormente sus mejores recetas pero ocasionalmente experimenta.

### Double DQN — El problema que resuelve

**El problema con DQN vanilla:** cuando la red estima el valor de la mejor siguiente acción, usa la misma red para *elegir* cuál es la mejor acción y *evaluar* cuán buena es. Esto crea un sesgo sistemático hacia sobreestimar el valor de las acciones (*maximisation bias*). El agente cree que sus acciones son mejores de lo que realmente son, y aprende políticas demasiado agresivas.

**La solución (Double DQN, van Hasselt et al. 2016):** se usan dos redes:
- La **red online** (`self.online_net`) elige qué acción tomar (`argmax`).
- La **red target** (`self.target_net`) evalúa cuán buena es esa acción.

Al separar selección y evaluación, se rompe el sesgo. Implementado en `agent.py, Agent.update(), líneas ~210–216`:
```python
next_act = self.online_net(nobs_t).argmax(dim=1, keepdim=True)  # online elige
q_next   = self.target_net(nobs_t).gather(1, next_act)          # target evalúa
```

### La red target — Por qué se necesitan dos redes

En DQN, el objetivo de entrenamiento (el "target") es:
```
target = recompensa + gamma × max_Q(siguiente_estado)
```

Si se usa la misma red para calcular `max_Q(siguiente_estado)` que se está actualizando, el objetivo cambia en cada paso de gradiente. Es como intentar medir una cuerda con la misma cuerda que se está estirando: la referencia no está fija.

La **red target** (`self.target_net`, `agent.py, línea ~163`) es una copia congelada de la red online. Se actualiza sólo cada 1 000 pasos con `sync_target()` (`agent.py, línea ~224`). Esto da un objetivo estable durante esos 1 000 pasos, y el entrenamiento converge.

---

## 7. El loop de entrenamiento — Qué pasa durante el entrenamiento

### Un paso de entrenamiento completo

Implementado en `agent.py, Agent.train(), líneas ~228–248`. Para cada uno de los 200 000 pasos:

1. **Epsilon-greedy:** con probabilidad `epsilon` (que va de 1.0 a 0.05), elige una acción aleatoria. Si no, usa `online_net.argmax()` para elegir la mejor acción conocida.

2. **Ejecutar en el entorno:** llama a `env.step(action)`. El entorno calcula el nuevo valor del portafolio, la observación siguiente y la recompensa.

3. **Guardar la transición:** almacena `(obs, action, reward, next_obs, done)` en el replay buffer.

4. **Actualizar la red (si el buffer tiene ≥ 64 transiciones):** samplera 64 transiciones aleatorias y ejecuta un paso de gradiente descendiente con la pérdida de Double DQN. Cada 500 pasos, si se proporcionó un `fold_id`, escribe una fila en `models/metrics/fold_{fold_id}_loss.csv` con el valor de la loss, el epsilon actual y el tamaño del buffer (`agent.py, líneas 287–289`).

5. **Sincronizar la red target (cada 1 000 pasos):** copia los pesos de `online_net` a `target_net`.

6. **Resetear si el episodio terminó:** si `done=True` (se llegó al final de los datos de entrenamiento), llama a `env.reset()`.

### Validación Walk-Forward — Por qué no un solo split

Un único split train/test asume que el mercado se comporta igual en todos los períodos. En mercados financieros esto es falso: hay regímenes de alta y baja volatilidad, períodos de bull y bear market. Un agente entrenado en datos del 2018 puede fallar completamente en 2022 (crisis post-COVID, aumento de tasas).

La **validación walk-forward** entrena el agente en un período histórico y lo evalúa en el período inmediatamente siguiente, simulando exactamente cómo se usaría en producción. La función `train_fold()` (`agent.py, línea ~270`) implementa un fold completo.

### Los 5 folds de `configs/default.yaml` (líneas 6–11)

| Fold | Datos de entrenamiento | Datos de evaluación | Qué prueba |
|------|----------------------|--------------------|----|
| 1 | Inicio – dic 2019 | Todo 2020 | Robustez ante el crash de COVID (feb-mar 2020) y la recuperación. |
| 2 | Inicio – dic 2020 | Todo 2021 | Desempeño en bull market post-COVID, alta euforia de mercados. |
| 3 | Inicio – dic 2021 | Todo 2022 | Resistencia al bear market más profundo (FED sube tasas, caídas > 70%). |
| 4 | Inicio – dic 2022 | Todo 2023 | Recuperación moderada, baja volatilidad relativa. |
| 5 | Inicio – dic 2023 | 2024–2025 | El más reciente: mercados actuales, el más relevante para el grader. |

El **held-out final** (`held_out_end: 2026-12-31` en `configs/default.yaml, línea 12`) nunca se toca hasta la evaluación final. Cualquier ajuste de hiperparámetros hecho mirando el held-out es causa de descalificación.

---

## 8. Cómo ejecutar el entrenamiento (para el equipo)

### Smoke test — Un solo fold para verificar que todo funciona

```bash
uv run python agent.py
```

Esto ejecuta la función `train_fold()` con el fold 5 (2023-2025) en el bloque `__main__` de `agent.py, líneas ~355–370`. Tarda aproximadamente 3–8 minutos en CPU, dependiendo del hardware.

### Para correr un fold específico desde Python

```python
from agent import train_fold

metrics = train_fold({"train_end": "2021-12-31", "eval_end": "2022-12-31"})
print(metrics)
```

### Para correr los 26 tests de submission

```bash
uv run pytest tests/test_submission.py -v
```

Todos los 26 deben pasar. Un solo test fallido implica un problema grave de implementación.

### Qué esperar durante el entrenamiento

El output del smoke test tiene esta forma:
```
[train_fold]  train_end=2023-12-31  eval_end=2025-12-31
  train rows :  52484  eval rows :  17544
  training Agent for 200,000 steps...
    step   20000/200000  eps=0.620  buf=20000
    step   40000/200000  eps=0.240  buf=40000
    step   60000/200000  eps=0.050  buf=60000
    ...
  evaluating on eval split...
  sortino=  X.XXXX  cum_ret=  X.XXXX  max_dd= -X.XXXX
```

### Qué es un buen resultado vs uno malo

La **métrica primaria es el Sortino** (`compute_metrics()` en `src/metrics.py, línea 16`). El Sortino es el retorno anualizado dividido por la volatilidad *negativa* (solo penaliza las caídas, no la volatilidad al alza).

| Resultado | Diagnóstico |
|-----------|-------------|
| Sortino agente > Sortino EqualWeight | El agente aprendió timing de mercado útil. Buen resultado. |
| Sortino agente ≈ Sortino EqualWeight | El agente no aporta valor. Revisar reward y exploración. |
| Sortino agente < Sortino HoldCash | El agente destruye capital. Problema grave de reward hacking o lookahead. |
| `cum_ret` muy positivo en train, negativo en eval | Sobreajuste al período de entrenamiento. |
| `turnover` promedio > 1.5 por paso | El agente hace churn. Aumentar `LAMBDA` en la recompensa (`agent.py, línea 44`). |

---

## 9. Riesgos conocidos y qué vigilar

Basado en el Apéndice de `GUIA_EQUIPO.md`.

---

### Riesgo 1 — Lookahead en el scaler

**Qué es:** el `StandardScaler` se entrena con datos que incluyen el período de evaluación, "filtrando" información del futuro hacia el pasado.

**Cómo se ve cuando ocurre:** las métricas en evaluación son artificialmente buenas (10–40% mejores de lo real). Al agregar nuevos datos futuros, el desempeño colapsa.

**Cómo detectarlo:** buscar cualquier llamada a `build_features(data_completo, fit=True)` donde `data_completo` incluya fechas del período de eval.

**Fix exacto:** en `agent.py, train_fold(), línea ~288`, la única llamada con `fit=True` debe ser sobre `train_data`. La evaluación usa el `scaler` ya entrenado sin re-fit.

---

### Riesgo 2 — NaN en features rolling

**Qué es:** las primeras filas del DataFrame de features son NaN porque las ventanas rodantes (21 períodos) no tienen suficiente historia. Si el entorno intenta leer esas filas, la red neuronal recibe NaN y sus gradientes explotan.

**Cómo se ve:** `AssertionError: NaN in obs at t=0`, o pérdidas de entrenamiento que de repente se vuelven `nan`.

**Fix exacto:** `_lookback = 21` en `agent.py, TradingEnv.__init__(), línea ~57`. El entorno empieza en `t=21`, la primera fila válida. Verificar con `np.isnan(obs).any()` en el primer `reset()`.

---

### Riesgo 3 — Turnover hacking (churn)

**Qué es:** el agente aprende que puede maximizar la recompensa cambiando de posición en cada paso, porque el horizonte corto no le permite ver que los costos acumulados destruyen el retorno.

**Cómo se ve:** `info["turnover"]` > 1.0 en promedio. El portafolio fluctúa violentamente pero no crece en el largo plazo.

**Fix:** aumentar `LAMBDA` (penalización de turnover) en `agent.py, línea 44`. Valor actual: 0.1. Probar 0.2–0.5 si el churn persiste. Cuidado: demasiado alto y el agente se paraliza (Riesgo 4).

---

### Riesgo 4 — Cash parking

**Qué es:** el agente aprende que quedarse 100% en cash tiene turnover=0 y drawdown=0, lo que da recompensa estable cercana a 0 en lugar de explorar estrategias más riesgosas pero potencialmente mejores.

**Cómo se ve:** el agente elige acción 0 (All Cash) en > 80% de los pasos. `cum_ret ≈ 0`. `sortino` del agente es menor que el de `EqualWeight`.

**Fix:** reducir `LAMBDA` y/o `MU`. Verificar que `EPSILON_START = 1.0` y que la exploración dure suficiente (`EPSILON_DECAY_STEPS = 50000`). Si el problema persiste al final del entrenamiento, el reward está mal calibrado.

---

### Riesgo 5 — Tunear hiperparámetros mirando el test

**Qué es:** ajustar `LAMBDA`, `MU`, la arquitectura de la red, o cualquier otro parámetro mirando las métricas del período de evaluación. Esto es trampa: el agente "aprende" el período de evaluación indirectamente.

**Cómo se ve:** el agente funciona muy bien en un fold específico pero mal en los demás. Los resultados no son reproducibles.

**Fix estricto:** toda búsqueda de hiperparámetros debe hacerse comparando desempeño en los folds de walk-forward (folds 1–5), nunca mirando el held-out final (`held_out_end: 2026-12-31` en `configs/default.yaml, línea 12`). El held-out se evalúa exactamente una vez, al final.

---

### Riesgo 6 — Posiciones cortas sin costo de financiamiento

**Qué es:** el modelo permite pesos negativos (posiciones cortas) pero no cobra el *funding rate* diario que los contratos de futuros o préstamos de acciones tienen en la realidad. Esto hace que las posiciones cortas sean artificialmente baratas.

**Cómo se ve:** el agente tiende a sobreutilizar acciones 6, 7 u 8 (Short A0/A1/A2). El drawdown en evaluación es mayor de lo esperado porque el costo real era mayor.

**Fix:** documentar explícitamente esta simplificación en el reporte. Opcionalmente, reportar métricas con y sin acciones de short para cuantificar el impacto.

---

### Riesgo 7 — Indexación temporal incorrecta en `_obs()`

**Qué es:** usar el índice entero `self._t` para acceder al DataFrame de features en lugar del timestamp. Si los índices del DataFrame de precios y el DataFrame de features no están perfectamente alineados (por ejemplo, al trabajar con el split de evaluación), se lee la fila equivocada.

**Cómo se ve:** resultados extraños sin NaN explícitos. El agente recibe features de un período diferente al precio que está viendo.

**Fix exacto:** en `agent.py, TradingEnv._obs(), línea ~82`, el acceso es:
```python
timestamp = self.data.index[t]
feat = self.features.loc[timestamp]   # acceso por fecha, no por posición
```
Esto garantiza que siempre se lee la feature del mismo instante de tiempo que el precio, independientemente de cómo estén alineados los DataFrames.

---

*Última actualización: 2026-06-06. Código base: `agent.py` + infraestructura en `src/`. Tests: `tests/test_submission.py`.*

---

## 10. Troubleshooting y Guía de Operación

---

### 10.1 Cómo correr el entrenamiento

#### Comando exacto para smoke test (un fold)

Desde el directorio raíz del proyecto:

```bash
uv run python agent.py
```

Esto ejecuta el bloque `if __name__ == "__main__"` de `agent.py` (línea 323). El bloque hace primero verificaciones rápidas de entorno y agente (tardan ~10 segundos), y después lanza automáticamente el fold 5 vía `train_fold()` en la línea 367. Los datos de ese fold son: **train hasta 2023-12-31, eval 2024–2025**.

Para correr **un fold específico** sin las verificaciones previas:

```bash
uv run python -c "from agent import train_fold; m = train_fold({'train_end': '2023-12-31', 'eval_end': '2025-12-31'}); print(m)"
```

Reemplazar las fechas según la tabla de folds en `configs/default.yaml, líneas 7–11`.

#### Comando exacto para entrenamiento completo (los 5 folds)

**No existe en el código actual.** No hay ninguna función ni script que itere los 5 folds automáticamente. La función `train_fold()` (`agent.py, línea 276`) corre exactamente un fold por llamada. Para correr todos:

```python
from agent import train_fold

folds = [
    {"train_end": "2019-12-31", "eval_end": "2020-12-31"},
    {"train_end": "2020-12-31", "eval_end": "2021-12-31"},
    {"train_end": "2021-12-31", "eval_end": "2022-12-31"},
    {"train_end": "2022-12-31", "eval_end": "2023-12-31"},
    {"train_end": "2023-12-31", "eval_end": "2025-12-31"},
]
results = {f["eval_end"]: train_fold(f) for f in folds}
```

Cada fold tarda entre 3 y 8 minutos en CPU. Los 5 folds en secuencia pueden tardar entre 20 y 45 minutos.

#### Cómo reanudar desde checkpoint si el proceso se cayó

El sistema de checkpoints **ya está implementado** y funciona automáticamente cuando se llama a `train_fold()`.

Durante el entrenamiento, cada 10 000 pasos `Agent.train()` guarda dos archivos (`agent.py, líneas 292–300`):

- `models/checkpoints/fold_{fold_id}_step_{N}.pt` — snapshot permanente en el paso N (quedan 20 archivos al terminar un fold de 200 000 pasos).
- `models/checkpoints/fold_{fold_id}_latest.pt` — siempre sobreescrito con el checkpoint más reciente; es el que usa el resume.

Cada checkpoint contiene: pesos de `online_net`, pesos de `target_net`, estado del optimizador Adam, y el paso exacto `self._step`.

Al inicio de `Agent.train()` (`agent.py, líneas 252–259`), si `fold_{fold_id}_latest.pt` existe, se carga automáticamente y el loop de entrenamiento retoma desde `self._step + 1` en lugar de 1.

**Para reanudar**, simplemente volver a correr el mismo comando:

```bash
uv run python -c "from agent import train_fold; train_fold({'train_end': '2023-12-31', 'eval_end': '2025-12-31'})"
```

Si el checkpoint existe, la consola mostrará:
```
    Resumed from checkpoint: step=50000
    step   60000/200000  eps=0.050  buf=60000
    ...
```

El `fold_id` se deriva automáticamente de `train_end` si no se especifica: `"2023-12-31"` → `"20231231"`. Para que el resume funcione, el `fold_id` implícito de la segunda llamada debe coincidir con el de la primera (lo cual es automático si se usa la misma fecha `train_end`).

---

### 10.2 Qué esperar en los primeros 5 minutos

#### Qué debe aparecer en consola si todo va bien

El output correcto tiene esta secuencia. Los primeros tres bloques aparecen en menos de 30 segundos; el cuarto bloque tarda minutos.

**Bloque 1 — Sanidad del entorno** (`agent.py, líneas 338–346`):
```
Sanity test passed: reset() + step(0) x5, no NaN.
  obs_shape     : (22,)  OK (22,)
  action_space  : Discrete(10)
  _lookback     : 21
  portfolio_val : 10000.0000
  All 10 actions have weights summing to 1. OK.
```

**Bloque 2 — Sanidad del agente** (`agent.py, líneas 349–352`):
```
Agent sanity: act(zeros, epsilon=1.0) = <número entre 0 y 9>
  device      : cpu
  buffer size : 0 / 100000
  epsilon     : 1.0000  (step=0, decays to 0.05)
```

**Bloque 3 — 10 pasos de recompensa** (`agent.py, líneas 354–363`):
La primera fila siempre tiene `turnover=2.0000` (el portafolio empieza en all-cash y la acción 4 cambia todos los pesos, produciendo el máximo turnover posible). Las filas 2–10 tienen `turnover=0.0000` porque la acción no cambia.

**Bloque 4 — Inicio del fold 5** (`agent.py, líneas 285, 292, 306`):
```
--- Fold 5 smoke test (train: start-2023, eval: 2024-2025) ---
[train_fold]  train_end=2023-12-31  eval_end=2025-12-31
  train rows :  52484  eval rows :  17544
  training Agent for 200,000 steps...
```

Los números `52484` y `17544` son deterministas: si aparece cualquier otro valor, el split de datos está mal.

#### Números concretos de epsilon — la única señal de progreso durante el entrenamiento

El loop de entrenamiento (`agent.py, líneas 302–304`) imprime una línea cada 20 000 pasos. La **loss no se imprime en consola**, pero se escribe en `models/metrics/fold_{fold_id}_loss.csv` cada 500 pasos. Para ver la evolución de la loss durante o después del entrenamiento sin esperar al final:

```python
import pandas as pd
df = pd.read_csv("models/metrics/fold_20231231_loss.csv")
print(df.tail(10))
# columnas: step, loss, epsilon, buffer_size
```

La señal visible en consola sigue siendo el paso, el epsilon y el tamaño del buffer.

Los valores de epsilon son **completamente deterministas** porque dependen solo de la fórmula lineal en `agent.py, propiedad epsilon, línea 175–178`:

| Paso | Línea esperada | Epsilon exacto |
|------|---------------|----------------|
| 20 000 | `step   20000/200000  eps=0.620  buf=20000` | `1.0 + (20000/50000) × (0.05−1.0) = 0.620` |
| 40 000 | `step   40000/200000  eps=0.240  buf=40000` | `1.0 + (40000/50000) × (0.05−1.0) = 0.240` |
| 60 000 | `step   60000/200000  eps=0.050  buf=60000` | Epsilon ya llegó al mínimo (`EPSILON_END = 0.05`) |
| 80 000+ | `step   80000/200000  eps=0.050  buf=80000` | Se mantiene en 0.050 para siempre |
| 100 001+ | `step  100000/200000  eps=0.050  buf=100000` | Buffer lleno. `buf` se queda en 100 000. |

Si los valores de epsilon difieren de los de la tabla, hay un bug en el contador `self._step` (`agent.py, línea 245`).

---

### 10.3 Señales de que algo está mal

**Qué se puede observar ahora:** la loss de la red se graba en `models/metrics/fold_{fold_id}_loss.csv` cada 500 pasos (`agent.py, líneas 287–289`). La recompensa del episodio de evaluación y el Sortino se graban en `models/metrics/fold_{fold_id}_metrics.csv` al terminar cada fold (`agent.py, líneas 370–388`). **Lo que NO aparece en consola:** ni la loss ni la recompensa se imprimen durante el loop; para verlos hay que leer los CSV.

---

#### Loss explota (NaN o > 1000)

**Síntoma observable:** el proceso termina con `AssertionError: NaN in obs at t=...` (`agent.py, TradingEnv._obs(), línea 89`). Esto ocurre porque una loss NaN corrompe los pesos de la red, que luego produce valores NaN en la predicción, que se propagan a la observación del siguiente paso. Alternativamente, el proceso puede continuar sin error pero el comportamiento del agente se vuelve completamente aleatorio desde ese punto.

**Causa probable:** el scaler no fue aplicado correctamente y las features tienen valores de magnitud extrema (miles o millones en lugar de ~0 a ~3). Esto satura las capas lineales de `QNetwork` y produce gradientes infinitos.

**Fix exacto:** verificar en `agent.py, train_fold(), línea 296` que `build_features(train_data, fit=True)` se llama sobre `train_data` solamente, y que la línea 299 usa el mismo `scaler` objeto sin re-ajustar. Confirmar que las features tienen valores en rango razonable imprimiendo `train_features.describe()` antes de crear `TradingEnv`.

---

#### Reward no mejora después de 50 000 steps

**Síntoma observable:** la loss **sí se graba** en `models/metrics/fold_{fold_id}_loss.csv`. Si la loss se estanca en un valor constante desde el paso 500, el agente no está aprendiendo. Al terminar el fold, `sortino` y `cum_ret` en `models/metrics/fold_{fold_id}_metrics.csv` (y en consola, `agent.py, líneas 366–368`) son iguales o peores que el baseline.

**Señal indirecta disponible:** después de 50 000 pasos, epsilon se fija en 0.050 (`EPSILON_END`, `agent.py, línea 149`). Si el agente hubiera aprendido algo útil, debería estar eligiendo acciones coherentes en el 95% de los pasos. Si las métricas finales muestran `cum_ret ≈ 0`, el agente no aprendió.

**Causa probable:** `GAMMA = 0.99` (`agent.py, línea 147`) es adecuado para el problema, pero si el reward es siempre cercano a cero (por drawdown permanente + penalización de turnover), la señal de aprendizaje es demasiado débil.

**Fix exacto:** cambiar temporalmente a R2 (comentar R3 y descomentar R2 en `agent.py, línea 115`) para eliminar la penalización de drawdown y verificar si el agente aprende con esa señal más simple. Si aprende con R2, el problema está en la calibración de `MU = 0.5` (`agent.py, línea 46`); reducirlo a 0.1 y probar.

---

#### El agente siempre elige la misma acción

**Síntoma observable:** en la línea de evaluación final, `cum_ret` es exactamente `0.0000` (acción 0, all-cash) o un valor fijo que no varía entre episodios. No es visible durante el entrenamiento.

**Causa probable A — cash parking:** la combinación de penalización de turnover y drawdown hace que all-cash (turnover=0, drawdown=0) sea la estrategia dominante. El agente aprende rápido que no hacer nada es "seguro".

**Fix A:** reducir `LAMBDA` de 0.1 a 0.02 (`agent.py, línea 45`). Reducir `MU` de 0.5 a 0.1 (`agent.py, línea 46`). El agente necesita una penalización de inactividad implícita: un retorno positivo del mercado que no captura debería pesar más que la comodidad del cash.

**Causa probable B — exploración insuficiente:** `EPSILON_DECAY_STEPS = 50_000` (`agent.py, línea 150`) puede ser demasiado corto para 200 000 pasos de entrenamiento. El agente explora solo en los primeros 25% del entrenamiento.

**Fix B:** aumentar `EPSILON_DECAY_STEPS` a 100 000. Cambiar el valor en `agent.py, línea 150`.

---

#### El agente nunca elige cash (acción 0)

**Síntoma observable:** `max_dd` en la evaluación final es peor que `-0.50` (caída de más del 50% desde el pico) sin recuperación. En el período bear de 2022, un agente sin refugio en cash puede perder > 80%.

**Causa probable:** la penalización de drawdown en R3 (`agent.py, líneas 109, 118`) no está siendo lo suficientemente fuerte para que el agente aprenda a retirarse al cash cuando el mercado cae. El coeficiente `MU = 0.5` puede ser insuficiente si el log-retorno de los activos en tendencia alcista supera consistentemente 0.5 × drawdown.

**Fix exacto:** aumentar `MU` de 0.5 a 1.0 o 2.0 en `agent.py, línea 46`. Esto hace que cada 1% de drawdown cueste 2× más en recompensa, forzando al agente a valorar la protección de capital.

---

#### El agente hace flip de acción en cada step (churn)

**Síntoma observable:** `cum_ret` en evaluación es negativo o cercano a cero a pesar de que el mercado subió en ese período (consultable en `notebooks/eda.ipynb` o en el precio de cierre de los datos). El agente está destruyendo capital con costos de transacción.

**Nota:** `info["turnover"]` es retornado por `env.step()` (`src/env.py, línea 67`) pero el loop de `Agent.train()` (`agent.py, líneas 238–258`) lo descarta con `_`. No está registrado en ningún lado durante el entrenamiento.

**Causa probable:** `LAMBDA = 0.1` es insuficiente para disuadir el churn. Con acciones de short que tienen turnover=2.0 al entrar y 2.0 al salir, el costo de un flip completo es `0.1 × 4.0 = 0.4` en penalización de reward, que puede ser menor al ruido de retorno del mercado en una hora.

**Fix exacto:** aumentar `LAMBDA` de 0.1 a 0.3 en `agent.py, línea 45`. Si el churn persiste, aumentar a 0.5. Cuidado con el riesgo opuesto (cash parking) si se sube demasiado.

---

#### El proceso se cae con OOM (out of memory)

**Síntoma observable:** `MemoryError` o `RuntimeError: CUDA out of memory` en consola.

**Probabilidad real: extremadamente baja.** Basado en el código actual:

- El archivo parquet completo tiene 70 005 filas × 16 columnas de float32 ≈ **4.5 MB** (`src/data.py, load_prices(), línea 27`).
- El replay buffer lleno tiene 100 000 transiciones × (22+1+1+22+1) floats × 4 bytes ≈ **18.8 MB** (`agent.py, BUFFER_SIZE = 100_000, línea 152`).
- La `QNetwork` tiene 22×256 + 256×256 + 256×10 = 73 728 parámetros × 4 bytes ≈ **289 KB** × 2 redes ≈ **578 KB** (`agent.py, HIDDEN_DIMS = (256, 256), línea 145`).
- Total estimado: **< 50 MB** en RAM, prácticamente imposible de agotar en cualquier máquina moderna.

**Causa probable si ocurre de todas formas:** una versión de PyTorch o numpy que genera tensores intermedios inesperadamente grandes, o un bug que crea un loop infinito llenando memoria.

**Fix:** verificar con `import psutil; print(psutil.virtual_memory())` que hay RAM disponible. Si el error es de CUDA: `agent.py, línea 163` detecta CUDA automáticamente con `torch.cuda.is_available()`. Para forzar CPU: agregar `os.environ["CUDA_VISIBLE_DEVICES"] = ""` antes de crear el `Agent`.

---

### 10.4 Dónde están los resultados guardados

`train_fold()` crea la carpeta `models/` en la raíz del proyecto y escribe tres tipos de archivos. Las carpetas se crean con `os.makedirs(..., exist_ok=True)` (`agent.py, líneas 247–248`), así que no hace falta crearlas manualmente.

#### Layout de archivos después de correr un fold completo

```
models/
  checkpoints/
    fold_20231231_step_10000.pt     ← snapshot en paso 10 000
    fold_20231231_step_20000.pt     ← snapshot en paso 20 000
    ...
    fold_20231231_step_200000.pt    ← snapshot final
    fold_20231231_latest.pt         ← siempre igual al último snapshot (para resume)
  metrics/
    fold_20231231_loss.csv          ← 400 filas (200 000 / 500 pasos)
    fold_20231231_metrics.csv       ← 1 fila por llamada a train_fold()
```

#### Contenido de cada archivo

| Archivo | Columnas | Cómo leerlo | Cuándo se escribe |
|---------|----------|-------------|------------------|
| `fold_{id}_loss.csv` | `step, loss, epsilon, buffer_size` | `pd.read_csv(...)` | Cada 500 pasos de entrenamiento (`agent.py, línea 289`) |
| `fold_{id}_metrics.csv` | `step, episode_reward, sortino, cum_ret, max_dd, epsilon, timestamp` | `pd.read_csv(...)` | Al terminar la evaluación de cada fold (`agent.py, línea 375`) |
| `fold_{id}_step_N.pt` | Pesos de `online_net`, `target_net`, `optimizer`, `step` | `torch.load(...)` | Cada 10 000 pasos de entrenamiento (`agent.py, línea 299`) |
| `fold_{id}_latest.pt` | Igual que el anterior | `torch.load(...)` | Sobreescrito en cada checkpoint (`agent.py, línea 300`) |

#### Comandos rápidos para ver las métricas

```python
import pandas as pd

# Ver la curva de loss del fold 5
df_loss = pd.read_csv("models/metrics/fold_20231231_loss.csv")
print(df_loss[["step", "loss"]].tail(10))

# Ver métricas de evaluación de todos los folds ya corridos
import glob
dfs = [pd.read_csv(f) for f in sorted(glob.glob("models/metrics/fold_*_metrics.csv"))]
print(pd.concat(dfs)[["step", "sortino", "cum_ret", "max_dd", "timestamp"]])
```

#### Cómo cargar manualmente un checkpoint guardado

```python
import torch
from agent import Agent

ckpt = torch.load("models/checkpoints/fold_20231231_latest.pt")
agent = Agent(obs_dim=22, n_actions=10)
agent.online_net.load_state_dict(ckpt["online_net"])
agent.target_net.load_state_dict(ckpt["target_net"])
agent._step = ckpt["step"]
print(f"Checkpoint en paso {agent._step}, epsilon={agent.epsilon:.4f}")
```

---

### 10.5 Cómo saber si el entrenamiento fue exitoso

#### Criterio concreto: sortino del agente vs. sortino del baseline EqualWeight

La **métrica primaria es el Sortino** (`src/metrics.py, línea 16`): retorno anualizado dividido por la desviación estándar de los retornos negativos. Un Sortino más alto significa mejor retorno por unidad de riesgo de caída.

El criterio mínimo de éxito es:

```
sortino(Agente) > sortino(EqualWeight)
```

**Advertencia:** `train_fold()` escribe las métricas del agente en `models/metrics/fold_{fold_id}_metrics.csv` (`agent.py, líneas 370–388`), pero **no corre los baselines automáticamente**. Para obtener el sortino de EqualWeight y comparar, hay que correrlo manualmente. EqualWeight siempre elige acción 4 (`src/baselines.py, EqualWeight.act(), línea 18`), por lo que se puede simular así desde Python:

```python
from agent import TradingEnv
from src.data import load_prices, split, build_features
from src.metrics import compute_metrics
import numpy as np

data = load_prices("1h")
train_data, eval_data = split(data, "2023-12-31", "2025-12-31")
train_features, scaler = build_features(train_data, fit=True)
eval_features, _ = build_features(eval_data, scaler=scaler)
eval_env = TradingEnv(eval_data, eval_features, scaler)

# Simular EqualWeight (acción 4 siempre)
values = [eval_env.initial_cash]
obs, _ = eval_env.reset()
done = False
while not done:
    obs, _, terminated, truncated, info = eval_env.step(4)
    values.append(info["portfolio_value"])
    done = terminated or truncated

baseline_metrics = compute_metrics(np.array(values))
print("EqualWeight sortino:", baseline_metrics["sortino"])

# Comparar con el agente ya entrenado
import pandas as pd
agent_metrics = pd.read_csv("models/metrics/fold_20231231_metrics.csv").iloc[-1]
print("Agente sortino:    ", agent_metrics["sortino"])
print("Diferencia:        ", agent_metrics["sortino"] - baseline_metrics["sortino"])
```

#### Qué fold revisar primero y por qué

**Revisar el fold 3 primero** (train hasta 2021-12-31, eval 2022 completo).

El año 2022 fue el peor año del período completo del dataset: los activos de riesgo cayeron más del 70% durante la subida de tasas de la FED. Un agente que sobrevive el fold 3 con un Sortino razonablemente cercano al de EqualWeight demostró que sabe proteger capital en un bear market extremo — la habilidad más difícil de aprender y la más importante para el grader.

Si el agente falla en el fold 3 (Sortino << EqualWeight o max_dd peor que -0.80), el problema suele ser cash parking o ausencia de protección por drawdown. Revisar `LAMBDA` y `MU` en `agent.py, líneas 45–46` antes de mirar otros folds.

El fold 5 (el que corre el smoke test automático) es el más relevante para la nota final porque evalúa en 2024–2025, el período más reciente. Pero el fold 3 es el más diagnóstico porque el señal de mercado es la más clara: si un agente no puede aprender "no pierdas dinero en un crash del 70%", la arquitectura tiene un problema fundamental.
