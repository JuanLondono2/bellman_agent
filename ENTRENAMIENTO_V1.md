# ENTRENAMIENTO V1 — Bellman Capital

> Registro de la primera ejecución de entrenamiento completo. Documentación de resultados,
> diagnóstico de fallos y plan de acción para V2.

---

## 1. Configuración del entrenamiento

**Fecha de ejecución:** 2026-06-07

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

**Versión de reward usada:** R3 (activa en `agent.py, _reward(), línea 122`):
```
recompensa = log_ret − 0.1 × turnover − 0.5 × drawdown
```
Con `LAMBDA = 0.1` (`agent.py, línea 49`) y `MU = 0.5` (`agent.py, línea 50`).

**Comando exacto ejecutado:**
```bash
uv run python agent.py
```
Esto ejecuta el bloque `__main__` de `agent.py` (línea 395), que corre el smoke test del fold 5
con `train_end=2023-12-31, eval_end=2025-12-31`.

---

## 2. Resultados por fold

Solo se corrió el fold 5 (smoke test). Los demás folds no se ejecutaron en V1.

| Fold | train_end | eval_end | sortino | cum_ret | max_dd |
|------|-----------|----------|---------|---------|--------|
| 5 | 2023-12-31 | 2025-12-31 | **−5.49** | **−0.9999** | **−0.9999** |
| 1–4 | — | — | no corrido | no corrido | no corrido |

**Datos del proceso de entrenamiento observados en consola:**

```
train rows :  52484  eval rows :  17544
training Agent for 200,000 steps...
  step   20000/200000  eps=0.620  buf=20000
  step   40000/200000  eps=0.240  buf=40000
  step   60000/200000  eps=0.050  buf=60000
  step   80000/200000  eps=0.050  buf=80000
  step  100000/200000  eps=0.050  buf=100000
  ...
  step  200000/200000  eps=0.050  buf=100000
sortino=  -5.4900  cum_ret= -0.9999  max_dd= -0.9999
```

`cum_ret = −0.9999` significa que el portafolio perdió el 99.99% de su valor inicial
(de $10,000 quedó aproximadamente $1) durante el período de evaluación de 17,544 pasos horarios.

---

## 3. Diagnóstico de fallos

### Fallo A — Valor del portafolio puede ir a cero o negativo con posiciones cortas

**Síntoma observado:** `cum_ret = −0.9999`, `max_dd = −0.9999`. El portafolio llegó a
prácticamente cero al final del período de evaluación.

**Causa raíz:** `src/env.py, BaseTradingEnv.step(), línea 60`:
```python
self._value = self._value * float(np.dot(w, ret)) - self._value * turnover * self.tc
```
Con una posición corta (por ejemplo, acción 6: `w = [-0.5, 0.5, 0.5, 0.5]`) y un activo que
sube 0.1% por hora (`ret[0] = 1.001`):
```
dot(w, ret) = -0.5×1.001 + 0.5×1.0 + 0.5×1.0 + 0.5×1.0 = 0.9995
```
Pérdida de 0.05% por hora. Compuesto sobre 17,544 horas:
```
0.9995^17544 = e^(−8.78) ≈ 0.00015  →  cum_ret ≈ −0.9999
```
Si el bull market de 2024-2025 tiene un retorno promedio de ~0.1%/hora en los activos
de riesgo, una posición corta sostenida produce exactamente el patrón observado.

**Impacto:** el valor del portafolio decae exponencialmente al cero. `cum_ret` y `max_dd`
convergen a −1.0.

---

### Fallo B — `log_ret` produce NaN cuando `curr_value` es negativo

**Síntoma observado:** no visible directamente en consola (no hay logs de NaN durante el
entrenamiento). Se manifiesta como resultados completamente incorrectos al final del fold.

**Causa raíz:** `agent.py, TradingEnv._reward(), línea 111`:
```python
log_ret = float(np.log(curr_value / (prev_value + 1e-8) + 1e-8))
```
Si `curr_value < -1e-8` (posible con posiciones cortas en un rally fuerte), el argumento
del logaritmo es negativo, y `np.log(número negativo) = NaN`. Este NaN se guarda en el
replay buffer vía `agent.py, store(), línea 195`. Cuando ese batch se samplea en
`update()` (línea 197), los gradientes son NaN, los pesos de la red se corrompen, y el
agente produce Q-values NaN de ahí en adelante.

**Impacto:** entrenamiento corrupto silencioso. La red aprende una política aleatoria o
degenerada sin ninguna señal de error en consola.

---

### Fallo C — Espiral de drawdown: todas las acciones producen el mismo reward negativo

**Síntoma observado:** una vez que el portafolio entra en drawdown, no logra recuperarse.
`max_dd = −0.9999` confirma que el portafolio cayó desde su pico y nunca volvió.

**Causa raíz:** `agent.py, TradingEnv._reward(), línea 113`:
```python
drawdown = max(0.0, (self._peak_value - curr_value) / (self._peak_value + 1e-8))
```
Cuando el portafolio cae a, por ejemplo, 50% de su pico (`curr_value / peak = 0.5`):
```
drawdown = 0.5
MU * drawdown = 0.5 × 0.5 = 0.25
```
La penalización de drawdown de 0.25 supera el log_ret horario típico (~0.0001 a 0.001
para mercados normales) en dos órdenes de magnitud. Todas las acciones producen reward
≈ `−0.25` sin importar cuál se elija. El gradiente es casi cero para todas, y el agente
no puede distinguir entre acciones buenas y malas. La señal de aprendizaje desaparece.

**Impacto:** el agente no aprende a recuperarse del drawdown. Queda atrapado eligiendo
acciones arbitrariamente durante el período de drawdown, lo que profundiza el colapso.

---

### Fallo D — Sin visibilidad de qué acción elige el agente durante el entrenamiento

**Síntoma observado:** el output de consola cada 20,000 pasos solo muestra `step`, `eps`
y `buf`. No hay ningún registro de cuál acción es elegida con más frecuencia.

**Causa raíz:** `agent.py, Agent.train(), líneas 302–304`:
```python
if global_step % 20_000 == 0 or global_step == n_steps:
    print(f"    step {global_step:>7}/{n_steps}  eps={eps:.3f}  buf={len(self.buffer)}")
```
No existe ningún contador de frecuencia de acciones en el código. Después del paso 60,000
(cuando `epsilon = 0.05`), el 95% de las acciones son greedy pero no se registran.

**Impacto:** no se puede saber, sin modificar el código, si el agente tiene cash parking
(siempre elige acción 0) o short bias (siempre elige acciones 6-8). Esta información es
crítica para diagnosticar el fallo y está completamente ausente.

---

### Fallo E — LAMBDA penaliza demasiado la primera acción de cada episodio

**Síntoma observado:** no directamente visible, pero es consistente con el agente que
no abandona la posición inicial (all-cash).

**Causa raíz:** `agent.py, líneas 49 y 122`. El estado inicial de cada episodio es
all-cash `[0, 0, 0, 1]`. Cualquier cambio a una acción de riesgo tiene turnover = 2.0:
```
penalización = LAMBDA × turnover = 0.1 × 2.0 = 0.2
```
El log_ret horario típico es ~0.001. La penalización de 0.2 supera al retorno esperado
por un factor de 200. El agente aprende en los primeros episodios que moverse desde cash
es inmediatamente costoso, independientemente del resultado del mercado.

**Impacto:** empuja al agente hacia cash parking. Sin embargo, si el agente parkea en
cash, `cum_ret ≈ 0`, no `−0.9999`. Por tanto, este fallo solo explica parcialmente el
resultado observado — el agente debe estar eligiendo algo peor que cash en evaluación.

---

## 4. Hipótesis principal

La causa más probable del `cum_ret = −0.9999` es que el agente aprendió, durante el
entrenamiento en datos 2018-2023, a favorecer una posición corta en alguno de los
tres activos (acciones 6, 7 u 8 del menú). El período 2022 fue un fuerte bear market
(caídas superiores al 70% en crypto), en el cual las posiciones cortas fueron
consistentemente rentables. El agente, con SEED=42 y una red inicializada aleatoriamente,
pudo haber convergido a esta política dado que las posiciones cortas produjeron los
mejores rewards en ese segmento de los datos de entrenamiento. Sin embargo, el período
de evaluación 2024-2025 fue un fuerte bull market, haciendo que esa misma política corta
produjera pérdidas de ~0.05% por hora compuesto, hasta llevar el portafolio a $0. Este
resultado es agravado por el Fallo C (espiral de drawdown): una vez que el portafolio
cae por el short bias, la señal de reward queda dominada por la penalización de drawdown
y el agente no puede corregir su política. La ausencia de logging de acciones (Fallo D)
impide confirmar este diagnóstico sin modificar el código.

---

## 5. Fixes propuestos

### Fix 1 — Proteger `log_ret` contra valores negativos o nulos del portafolio

**Qué cambiar:** `agent.py, TradingEnv._reward(), línea 111`.

**Cambio:**
```python
# Antes:
log_ret = float(np.log(curr_value / (prev_value + 1e-8) + 1e-8))

# Después:
safe_ratio = max(curr_value, 1e-8) / (prev_value + 1e-8)
log_ret = float(np.log(safe_ratio + 1e-8))
```
Esto clampea `curr_value` a un mínimo de `1e-8` antes de calcular el log, evitando NaN.
**Comportamiento esperado:** el agente nunca produce NaN en el buffer, y el training
permanece estable incluso si el portafolio llega a valores muy pequeños.

---

### Fix 2 — Reducir `MU` para evitar la espiral de drawdown

**Qué cambiar:** `agent.py, TradingEnv, línea 50`.

**Cambio:**
```python
# Antes:
MU = 0.5

# Después:
MU = 0.05
```
Con `MU = 0.05` y drawdown = 0.5, la penalización es `0.025`, del mismo orden de magnitud
que el log_ret horario esperado. El agente puede distinguir entre acciones buenas y malas
incluso durante un drawdown moderado.
**Comportamiento esperado:** el agente mantiene la capacidad de aprender a salir del
drawdown en lugar de quedar ciego a las diferencias entre acciones.

---

### Fix 3 — Agregar logging de frecuencia de acciones cada 20 000 pasos

**Qué cambiar:** `agent.py, Agent.train(), líneas 302–304`.

**Cambio:** agregar un contador de acciones dentro del loop y loguearlo junto con el paso:
```python
# Agregar al inicio del loop (antes del for):
action_counts = [0] * self.n_actions

# Dentro del loop, después de elegir action:
action_counts[action] += 1

# En el bloque de print cada 20 000 pasos, agregar:
top_action = max(range(self.n_actions), key=lambda a: action_counts[a])
top_pct = 100 * action_counts[top_action] / sum(action_counts)
print(f"    step {global_step:>7}/{n_steps}  eps={eps:.3f}  buf={len(self.buffer)}"
      f"  top_action={top_action} ({top_pct:.0f}%)")
action_counts = [0] * self.n_actions  # reset para el próximo intervalo
```
**Comportamiento esperado:** se puede ver, en cada bloque de 20 000 pasos, qué acción
domina. Si `top_action=0 (100%)` → cash parking confirmado. Si `top_action=6 (98%)` →
short bias confirmado.

---

### Fix 4 — Reducir `LAMBDA` para permitir exploración de acciones de riesgo

**Qué cambiar:** `agent.py, TradingEnv, línea 49`.

**Cambio:**
```python
# Antes:
LAMBDA = 0.1

# Después:
LAMBDA = 0.01
```
Con `LAMBDA = 0.01` y turnover = 2.0, la penalización del primer paso es `0.02`, que es
del mismo orden que el log_ret esperado. El agente puede aprender que moverse desde cash
puede ser rentable si el retorno de mercado lo justifica.
**Comportamiento esperado:** el agente deja de estar sesgado hacia cash parking. El
churn excesivo se puede detectar vía el logging del Fix 3.

---

## 6. Próximos pasos

Orden de ejecución para el entrenamiento V2:

1. **Implementar Fix 1** (`log_ret` seguro): elimina el riesgo de NaN que corrompe
   el buffer. Es el cambio más urgente — sin él, cualquier otra mejora puede fallar
   silenciosamente.

2. **Implementar Fix 3** (logging de acciones): necesario para saber si el agente está
   mejorando o simplemente eligiendo una acción dominante. Sin visibilidad, el debugging
   es ciego.

3. **Implementar Fix 2 + Fix 4** (`MU = 0.05`, `LAMBDA = 0.01`): recalibrar la reward
   para que el aprendizaje sea posible durante drawdowns y el primer paso de cada episodio.

4. **Correr el smoke test (solo fold 5) con los fixes aplicados** para verificar que:
   - No aparece `nan` en ningún valor
   - El `top_action` en el primer bloque de 20 000 pasos no es 100% la misma acción
   - `cum_ret` mejora respecto a −0.9999

5. **Verificar que los 26 tests siguen pasando** después de los cambios:
   ```bash
   uv run pytest tests/test_submission.py -v
   ```

6. **Si el smoke test pasa**, correr los 5 folds completos y documentar resultados
   en `ENTRENAMIENTO_V2.md`.

7. **Comparar Sortino del agente vs EqualWeight** en cada fold usando el script de
   evaluación de baselines documentado en `IMPLEMENTACION.md, Sección 10.5`.

---

*Versión: V1. Autor: equipo. Fecha: 2026-06-07.*
*Código base analizado: `agent.py` (445 líneas), `src/env.py` (84 líneas), `configs/default.yaml`.*
