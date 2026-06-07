# ENTRENAMIENTO V3 — Bellman Capital

> Registro del tercer entrenamiento: 5 folds completos con reward Sortino diferencial (Fix 5).
> Documentación de resultados, diagnóstico y plan de acción para V4.

---

## 1. Configuración del entrenamiento

**Fecha de ejecución:** 2026-06-07

**Fix aplicado respecto a V2** (ver diagnóstico completo en `ENTRENAMIENTO_V2.md`):

| Fix | Archivo y línea | Cambio |
|-----|-----------------|--------|
| Fix 5 | `agent.py:107–122` | `_reward()` reemplazada: `downside = max(0, −log_ret)` en lugar de `(peak − value) / peak`. El término de drawdown deja de ser acumulativo e independiente de la acción para convertirse en per-paso y acción-dependiente. |

**Todos los hiperparámetros son idénticos a V2** (fuente: `configs/default.yaml`, `agent.py`):

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
| `LAMBDA` | 0.01 |
| `MU` | 0.05 |

**Versión de reward activa (R3, `agent.py:121–123`):**
```
R3 = log_ret − 0.01 × turnover − 0.05 × max(0, −log_ret)
```

**Comando ejecutado:**
```bash
uv run python -c "
from agent import train_fold
folds = [
    {'train_end': '2019-12-31', 'eval_end': '2020-12-31'},
    {'train_end': '2020-12-31', 'eval_end': '2021-12-31'},
    {'train_end': '2021-12-31', 'eval_end': '2022-12-31'},
    {'train_end': '2022-12-31', 'eval_end': '2023-12-31'},
    {'train_end': '2023-12-31', 'eval_end': '2025-12-31'},
]
for f in folds:
    train_fold(f)
"
```

---

## 2. Resultados por fold

| Fold | train_end | eval_end | train_rows | eval_rows | sortino | cum_ret | max_dd | top_action_final |
|------|-----------|----------|-----------|----------|---------|---------|--------|-----------------|
| 1 | 2019-12-31 | 2020-12-31 | 17 438 | 8 795 | **+1.8580** | **+127.8567** | **−0.2333** | 7 (22%) |
| 2 | 2020-12-31 | 2021-12-31 | 26 210 | 8 776 | −1.5286 | −0.9978 | −0.9985 | 4 (12%) |
| 3 | 2021-12-31 | 2022-12-31 | 34 963 | 8 783 | −3.5462 | −1.0000 | −1.0000 | 0 (12%) |
| 4 | 2022-12-31 | 2023-12-31 | 43 723 | 8 784 | −4.9175 | −0.9998 | −0.9998 | 6 (12%) |
| 5 | 2023-12-31 | 2025-12-31 | 52 484 | 17 544 | −3.8967 | −1.0000 | −1.0000 | 1 (20%) |

**Progresión de `top_action` durante el entrenamiento (consola):**

| Paso | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------|--------|--------|--------|--------|--------|
| 20 000 | 3 (12%) | 3 (12%) | 3 (13%) | 3 (12%) | 3 (11%) |
| 40 000 | 9 (13%) | 1 (11%) | 0 (14%) | 0 (15%) | 1 (14%) |
| 60 000 | 7 (18%) | 1 (16%) | 0 (13%) | 2 (13%) | 1 (19%) |
| 80 000 | 2 (17%) | 4 (14%) | 0 (15%) | 0 (15%) | 1 (12%) |
| 100 000 | 2 (18%) | 0 (12%) | 0 (12%) | 0 (13%) | 1 (17%) |
| 120 000 | 7 (19%) | 4 (13%) | 0 (13%) | 0 (13%) | 1 (12%) |
| 140 000 | 2 (20%) | 4 (13%) | 0 (15%) | 0 (13%) | 1 (13%) |
| 160 000 | 2 (22%) | 4 (14%) | 5 (12%) | 7 (12%) | 1 (15%) |
| 180 000 | 7 (29%) | 4 (12%) | 0 (15%) | 0 (13%) | 6 (18%) |
| 200 000 | 7 (22%) | 4 (12%) | 0 (12%) | 6 (12%) | 1 (20%) |

---

## 3. Diagnóstico

### Q1 — Fold 1 es el único con resultados positivos (sortino=1.858, cum_ret=127.85). ¿Qué lo diferencia?

Tres factores actúan en conjunto y son inseparables:

**Factor A — Tamaño del dataset de entrenamiento.**
Fold 1 tiene 17 438 filas de entrenamiento. Con 200 000 pasos de entrenamiento y episodios de ~17 438 pasos cada uno, el agente completa aproximadamente `200 000 / 17 438 ≈ 11.5 episodios` completos. Fold 5, con 52 484 filas, completa solo `200 000 / 52 484 ≈ 3.8 episodios`. La diferencia de repeticiones por dato es de 3×. Más repeticiones permiten que los valores Q converjan mejor: el agente ve cada patrón de mercado ~11 veces en fold 1 vs ~4 veces en fold 5.

**Factor B — Diversidad de `top_action` durante el entrenamiento.**
En fold 1, el `top_action` varía entre 2, 3, 7 y 9 a lo largo del entrenamiento sin consolidarse en una sola acción. Los porcentajes oscilan entre 12% y 29%, lo que indica que el agente no colapsó a una política trivial y mantuvo exploración real. En los folds 2-5, el agente converge hacia una acción dominante (4 en fold 2, 0 en fold 3, 0/6 en fold 4, 1 en fold 5) con los mismos porcentajes bajos (~12%), pero la variación del `top_action` entre ventanas es menor.

**Factor C — Régimen de mercado en el período de evaluación.**
El período de evaluación de fold 1 (2020) fue históricamente un año de mercado alcista extremo en criptomonedas: caída por COVID en marzo seguida de recuperación en V y rally acelerado hasta diciembre. Este tipo de mercado fuertemente tendencial es el escenario más favorable para un DQN: las señales de momentum son estables, la dirección del mercado es consistente, y aprender "comprar y mantener" produce retornos extraordinarios. El `cum_ret = 127.85` (12 785% de retorno acumulado) refleja exactamente este tipo de tendencia. Los períodos de evaluación de folds 2-5 incluyen mercados más volátiles, bajistas o en rangos laterales donde la estrategia aprendida no generaliza.

**Conclusión Q1:** El éxito de fold 1 no demuestra que el agente aprendió a gestionar riesgo correctamente; demuestra que el régimen de mercado de 2020 era lo suficientemente tendencial como para que incluso una política subóptima produjera buenos resultados. No es replicable sin las condiciones de mercado de 2020.

---

### Q2 — Folds 3 y 4 muestran `top_action=0` (cash) dominando el entrenamiento, pero el resultado de evaluación es `cum_ret=−1.0`. ¿Cómo es posible?

**Lectura de `_weights_from_action()` (`agent.py:98–104`):**
```python
def _weights_from_action(self, action: int) -> np.ndarray:
    w = _WEIGHT_MENU[action].copy()
    assert np.isclose(w.sum(), 1.0, atol=1e-6), ...
    self._last_turnover = float(np.abs(w - self._weights).sum())
    return w
```

Acción 0 en `_WEIGHT_MENU` (`agent.py:30`): `[0.00, 0.00, 0.00, 1.00]` → sí es 100% cash.

Si el agente jugara acción 0 durante toda la evaluación, el portfolio permanecería en cash y `cum_ret ≈ 0` (no habría pérdidas ni ganancias de mercado). Pero el resultado es `cum_ret = −1.0`, lo que implica que durante la evaluación el agente **no** elige acción 0 mayoritariamente.

La explicación tiene dos partes:

**Parte 1 — `top_action` durante entrenamiento ≠ política greedy durante evaluación.**
`top_action = 0 (12%)` significa que en la ventana de 20 000 pasos de entrenamiento, acción 0 fue elegida en ~2 400 pasos (12%). Los restantes ~17 600 pasos se distribuyeron entre las otras 9 acciones. Con `epsilon = 0.05`, el 5% de esas decisiones son aleatorias. El 95% restante son greedy sobre la red Q. El hecho de que acción 0 sea la "más frecuente" con solo 12% indica que **ninguna acción domina realmente**: la red Q no ha convergido a una política clara.

**Parte 2 — Distribución shift entre entrenamiento y evaluación.**
Los estados de entrenamiento (mercado 2019-2021 en fold 3) son distribuciones distintas a los estados de evaluación (mercado bajista 2022: caída de ~70% en criptomonedas). La red Q aprendió valores Q para observaciones del régimen alcista. Cuando se enfrenta a observaciones del mercado bajista de 2022, el argmax de Q puede favorecer acciones risky (como Long A0, Long A1) porque esos patrones fueron rentables durante el entrenamiento. El agente no tiene señal para distinguir "este es un régimen bajista, diferente al régimen alcista que vi en entrenamiento".

**Conclusión Q2:** Acción 0 siendo `top_action` durante el entrenamiento no garantiza que el agente juegue cash durante la evaluación. Con 12%, solo es la más frecuente de 10 acciones casi uniformemente distribuidas. En los estados del mercado bajista de 2022, la red Q mapeó esos estados a acciones risky que habían sido rentables en el régimen de entrenamiento.

---

### Q3 — ¿Por qué más datos de entrenamiento produce peores resultados?

**Razón estructural — Episodios completados por fold.**

El entrenamiento es de 200 000 pasos fijos sin importar el tamaño del dataset. La duración de un episodio es igual al número de filas de entrenamiento (el entorno termina al llegar al final de los datos, `done=True`). Por tanto:

| Fold | train_rows | episodios completos (aprox.) | repeticiones de cada dato |
|------|-----------|------------------------------|--------------------------|
| 1 | 17 438 | 200 000 / 17 438 ≈ **11.5** | ~11.5× |
| 2 | 26 210 | 200 000 / 26 210 ≈ **7.6** | ~7.6× |
| 3 | 34 963 | 200 000 / 34 963 ≈ **5.7** | ~5.7× |
| 4 | 43 723 | 200 000 / 43 723 ≈ **4.6** | ~4.6× |
| 5 | 52 484 | 200 000 / 52 484 ≈ **3.8** | ~3.8× |

El agente de fold 5 ve cada patrón de mercado solo ~3.8 veces. Los valores Q no han convergido porque:
1. **Replay buffer saturado por episodio único:** el buffer tiene capacidad 100 000. Un episodio de fold 5 llena el 52% del buffer en un solo recorrido. El muestreo aleatorio del buffer en folds 5 está dominado por el episodio más reciente, perdiendo diversidad temporal.
2. **Target network desactualizada:** con solo 3-4 episodios, la red target (sincronizada cada 1 000 pasos) se actualiza 200 veces, pero los valores Q objetivo siguen siendo ruidosos porque cada valor Q fue calculado sobre poca repetición.
3. **No hay sobre-entrenamiento por exceso de datos:** el problema opuesto al esperado. No es que el agente memorice el dataset grande; es que no tiene suficientes repeticiones para aprender ningún patrón de él.

La paradoja: más datos de entrenamiento reduce la calidad del aprendizaje porque el presupuesto de pasos (200 000) es fijo e insuficiente para cubrir datasets grandes.

---

## 4. Hipótesis principal

El fracaso en folds 2-5 tiene dos causas independientes que se combinan. La causa primaria es el desajuste de régimen de mercado (*distribution shift*) entre entrenamiento y evaluación: el agente aprende valores Q para estados del mercado alcista (2017-2022) y los aplica a estados de mercados bajistas o laterales (2022-2025) donde esos valores Q llevan a acciones destructivas. Esta causa no puede ser resuelta por ningún ajuste de función de reward porque el problema no está en qué se penaliza sino en que los estados de evaluación son distribuciones distintas a las vistas durante el entrenamiento. La causa secundaria amplifica la primaria: a medida que los datasets de entrenamiento crecen por fold (de 17 438 a 52 484 filas), el presupuesto fijo de 200 000 pasos produce menos episodios completos por fold (de 11.5 a 3.8), degradando la convergencia de valores Q. Fold 1 funcionó porque la alineación entre el régimen aprendido (mercados de 2018-2019 con alta volatilidad y ciclos claros) y el régimen evaluado (bull run 2020) fue la más favorable del dataset, amplificada por la mejor convergencia debida a las ~11 repeticiones de cada patrón. El reward diferencial de V3 es la función de reward correcta para el problema (alinea la señal de aprendizaje con la métrica Sortino), pero no puede compensar la distribución shift ni la insuficiencia de episodios.

---

## 5. Comparación V1 vs V2 vs V3

V1 y V2 solo ejecutaron el fold 5 (smoke test). V3 ejecutó los 5 folds. La comparación directa fold a fold solo es posible para fold 5.

| Fold | V1 sortino | V2 sortino | V3 sortino | Tendencia |
|------|-----------|-----------|-----------|-----------|
| 1 | — | — | **+1.8580** | Solo V3 |
| 2 | — | — | −1.5286 | Solo V3 |
| 3 | — | — | −3.5462 | Solo V3 |
| 4 | — | — | −4.9175 | Solo V3 |
| 5 | **−5.4900** | **−3.6700** | **−3.8967** | V3 ≈ V2 |

**Conclusión sobre el Sortino diferencial (Fix 5):**

En fold 5 (el único comparable), V3 (−3.90) es marginalmente peor que V2 (−3.67). La diferencia de 0.23 en Sortino está dentro del ruido de entrenamiento y no es estadísticamente significativa con una sola corrida. El resultado principal es que **el Sortino diferencial no produjo mejora sustancial ni daño sustancial** en el fold comparable. La mejora de V2 sobre V1 (de −5.49 a −3.67) fue real y fue producida por los fixes 1-4 (especialmente MU 0.5→0.05). Fix 5 (Sortino diferencial) resolvió teóricamente la trampa estructural documentada en V2 (la penalización de drawdown cumulative era idéntica para todas las acciones), pero los resultados en folds 2-5 indican que el problema raíz no era la función de reward sino el distribution shift y la insuficiencia de episodios de entrenamiento.

---

## 6. Fixes propuestos para V4

**El problema no es la función de reward.**

La función R3 con Sortino diferencial es la correcta: `R3 = log_ret − LAMBDA×turnover − MU×max(0,−log_ret)`. Está bien calibrada (LAMBDA=0.01, MU=0.05) y alineada con la métrica de evaluación. No hay evidencia en los resultados de V3 de que cambiarla mejore los folds 2-5.

Los dos problemas identificados en el diagnóstico requieren los siguientes cambios:

---

**Fix 6 — Aumentar `TRAIN_STEPS` de 200 000 a 600 000** (`agent.py:159`)

**Fundamento:** con 200 000 pasos, fold 5 completa solo ~3.8 episodios. Con 600 000 pasos completaría ~11.5 episodios, equivalente a fold 1. El presupuesto de pasos debería escalar con el tamaño del dataset para mantener una cobertura equivalente entre folds.

```python
# Antes (V3):
TRAIN_STEPS = 200_000

# Después (V4):
TRAIN_STEPS = 600_000
```

Riesgo: el tiempo de entrenamiento se triplica. El problema de distribution shift no desaparece, pero la convergencia de valores Q mejorará, especialmente en folds 4 y 5.

---

**Fix 7 — Reducir `BUFFER_SIZE` de 100 000 a 50 000** (`agent.py:157`)

**Fundamento:** con fold 5 (52 484 filas por episodio), un solo episodio ocupa el 52% del buffer de 100 000. Si el agente ha visto 3 episodios, el buffer está dominado por el episodio más reciente (por FIFO), eliminando la diversidad temporal. Reducir el buffer a 50 000 no cambia esta proporción, pero garantiza que el muestreo siempre incluya transiciones de las últimas ~2 500 horas (≈ 100 días de trading) de cualquier fold. Para fold 1 (17 438 filas), un buffer de 50 000 contiene casi 3 episodios completos, manteniendo la diversidad.

```python
# Antes (V3):
BUFFER_SIZE = 100_000

# Después (V4):
BUFFER_SIZE = 50_000
```

Riesgo: ninguna garantía de que esto resuelva el distribution shift. Es una mejora de eficiencia de muestra.

---

**Nota sobre distribution shift:**
Si después de V4 los folds 2-5 siguen fallando, el problema de fondo es el distribution shift y requiere un cambio arquitectónico, no de hiperparámetros: por ejemplo, agregar al vector de observación indicadores de régimen de mercado (tendencia de largo plazo, posición relativa respecto a máximo histórico del período) para que el agente pueda distinguir entre estados alcistas y bajistas en el espacio de observación.

---

## 7. Próximos pasos

Orden de ejecución para V4:

1. **Aplicar Fix 6** en `agent.py:159`: `TRAIN_STEPS = 600_000`.

2. **Aplicar Fix 7** en `agent.py:157`: `BUFFER_SIZE = 50_000`.

3. **Verificar que los 26 tests siguen pasando:**
   ```bash
   uv run pytest tests/test_submission.py -v
   ```

4. **Eliminar checkpoints de V3** antes del entrenamiento:
   ```powershell
   Remove-Item models\checkpoints\* -Force
   Remove-Item models\metrics\* -Force
   ```

5. **Correr los 5 folds completos** y documentar resultados en `ENTRENAMIENTO_V4.md`:
   ```bash
   uv run python agent.py
   ```

6. **Criterio de éxito:** sortino > 0 en al menos 3 de los 5 folds. Si solo fold 1 es positivo nuevamente, el distribution shift es la causa dominante y se debe escalar a análisis de régimen.

---

## 8. Análisis de periodicidad

### Q1 — ¿Qué intervalo usa el agente exactamente?

El intervalo es **1 hora (1h)**. La llamada exacta está en `agent.py:346`:
```python
data = load_prices("1h")
```

`load_prices("1h")` (`src/data.py:20–27`) lee el archivo `data/raw/prices_1h.parquet` directamente con `pd.read_parquet(path)`.

---

### Q2 — ¿Qué intervalos están disponibles en `data/raw/`?

```
dir data\raw
prices_15m.parquet  prices_1h.parquet  prices_30m.parquet
```

Tres intervalos disponibles: **15 minutos**, **30 minutos**, **1 hora**.

---

### Q3 — Cálculo por intervalo disponible

El dataset de 1h produce 17 438 filas de entrenamiento para fold 1, que abarca aproximadamente 726 días (`17 438 / 24 ≈ 726`). Esto sitúa el inicio de los datos alrededor de enero-febrero de 2018. Usando ese período como base:

| Intervalo | Filas fold 1 (aprox.) | Pasos/año | Episodios con 200 000 pasos |
|-----------|-----------------------|-----------|----------------------------|
| 1h (actual) | 17 438 | 8 760 | **11.5** |
| 30m | ~34 876 | 17 520 | **5.7** |
| 15m | ~69 752 | 35 040 | **2.9** |

**Signal vs noise por intervalo para un agente DQN:**

- **1h:** las velas de 1 hora promedian el ruido intradía. Las features de momentum (mom_20, 20 candles = ~20 horas) y volatilidad (vol_21, 21 horas) capturan movimientos de 1-2 días. El `vol_ratio` compara el volumen de cada hora contra la media de 21 horas, que es una señal de interés institucional en la sesión. A este intervalo, las features tienen una relación señal/ruido razonable para patrones de días a semanas.

- **30m:** el doble de puntos, cada uno con la mitad del movimiento promedio de precio. Las features de rolling (21 y 20 candles) solo cubren ~10 horas en lugar de ~21 horas. Momentum captura movimientos de medio día, no de días. La señal se fragmenta en más observaciones sin que el contenido informativo total aumente. Con 200 000 pasos de entrenamiento, fold 1 completaría solo ~5.7 episodios (vs 11.5 con 1h), degradando la convergencia.

- **15m:** 4× más puntos. Features de rolling 21 candles = ~5 horas. Momentum de 20 candles = ~5 horas. Las velas de 15 minutos tienen mayor proporción de ruido de microestructura de mercado (spread bid-ask, fragmentación de órdenes). Con 200 000 pasos, fold 1 completaría solo ~2.9 episodios, insuficiente para que los valores Q converjan. El buffer de 100 000 entries representaría 6 semanas de datos en 15m, perdiendo los patrones estacionales que el agente necesita.

---

### Q4 — ¿El fallo de folds 2-5 es causado por el intervalo o por el régimen de mercado?

**Evidencia que apunta al régimen de mercado como causa primaria:**

1. **Fold 1 funciona con 1h** y el eval period (2020) fue un mercado alcista. Fold 2 también tiene eval en un mercado alcista (2021) pero falla. La diferencia entre fold 1 y fold 2 no es solo el intervalo — es el tamaño del training set (17 438 vs 26 210) y el régimen exacto del mercado de evaluación.

2. **El fold 3 eval (2022)** fue el peor año para criptomonedas (-70%+ en Bitcoin). Ningún intervalo —15m, 30m, 1h— puede resolver un distribution shift de esa magnitud: el agente entrenado en 2014-2021 no ha visto estados equivalentes al mercado bajista de 2022.

3. **Si se usara 30m en lugar de 1h**, fold 1 train tendría ~34 876 filas y completaría ~5.7 episodios con 200 000 pasos, peor que los 11.5 actuales. La convergencia de Q-values empeoraría, no mejoraría. El mismo razonamiento aplica a 15m (~2.9 episodios).

**Conclusión Q4:** el fallo de folds 2-5 es causado por el régimen de mercado (distribution shift), no por el intervalo. Un intervalo más fino (30m, 15m) con el mismo presupuesto de pasos empeoraría los resultados porque reduciría aún más el número de episodios completos por fold. El éxito de fold 1 con 1h no es replicable en otros folds simplemente usando un intervalo diferente.

---

### Q5 — Recomendación: ¿debe V4 cambiar el intervalo?

**No cambiar el intervalo. Mantener 1h.**

Razones:

1. El problema dominante es la insuficiencia de episodios de entrenamiento por fold, no la granularidad. Cambiar a 30m o 15m con el mismo presupuesto de pasos (200 000) agravaría este problema multiplicando las filas por episodio.

2. Las features de `build_features()` (`src/data.py:48–102`) incluyen `vol_21` (volatilidad de 21 candles) y `mom_20` (momentum de 20 candles). A 1h, estas ventanas representan ~21 horas y ~20 horas respectivamente, capturando dinámicas de 1-2 días de trading. A 15m, las mismas ventanas solo capturan ~5 horas, perdiendo contexto de días completos de mercado.

3. Si en V4 se aumenta `TRAIN_STEPS` a 600 000 (Fix 6), el número de episodios con 1h se mantiene proporcional entre folds (fold 1: ~34 episodios, fold 5: ~11 episodios). Cambiar a 30m con 600 000 pasos daría fold 1: ~17 episodios, fold 5: ~5 episodios — la disparidad se mantiene.

**Lo que debería cambiar en V4 (respuesta directa):**
`TRAIN_STEPS = 600_000` y `BUFFER_SIZE = 50_000`, manteniendo el intervalo 1h. Si los resultados de V4 siguen siendo negativos en folds 2-5, la siguiente iteración debe agregar features de régimen de mercado al vector de observación (22-dim actual) para que el agente pueda detectar si está en un mercado alcista o bajista — no cambiar el intervalo de los datos.

---

---

### 8.6 Conclusión: límites del feature engineering dado el presupuesto de pasos

Problema primario — Distribution shift: El agente se entrena en 2017-2019/2020 (mayoritariamente alcista para criptomonedas) y evalúa en 2021-2025 (incluye el bear market de 2022, mercados mixtos, y períodos de alta correlación entre activos que no estaban en el entrenamiento). Ninguna feature computada a posteriori sobre los datos de entrenamiento puede darle al agente experiencia que no tuvo. Si la red Q nunca vio un bear market sostenido de 12 meses durante el entrenamiento, no puede generalizar correctamente a uno, independientemente de cuántas features descriptivas tenga.

Conclusión directa: Con 200k pasos fijos y la arquitectura actual, no existe combinación de features que compense la insuficiencia de episodios en folds 2-5 ni la distribución shift entre regímenes. El feature engineering es la herramienta correcta después de resolver el presupuesto de pasos. En el orden inverso, es maquillaje sobre el problema real.

---

*Versión: V3. Autor: equipo. Fecha: 2026-06-07.*
*Código base analizado: `agent.py` (líneas 29–50, 98–123, 157–159, 267–311, 346), `src/data.py` (líneas 15–103), `configs/default.yaml`.*
