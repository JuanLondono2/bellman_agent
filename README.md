# Bellman Capital — Agente DQN de Asignación de Portafolio

Proyecto académico de Deep Reinforcement Learning. Se construyó un agente DQN que asigna capital entre tres activos de riesgo y cash usando datos OHLCV horarios de criptomonedas, evaluado con validación walk-forward en 5 folds temporales.

## Equipo

| Nombre | Rol |
|--------|-----|
| Alejandro Rubiano | Investigador |
| Juan Camilo San Miguel | Investigador |
| Juan Sebastian Londoño | Investigador |

**Fecha de inicio:** 2026-05-30

---

## Tesis del agente

Los mercados de criptomonedas a escala horaria exhiben suficiente estructura de momentum y clustering de volatilidad para que una política aprendida con DQN pueda asignar capital de forma rentable con ajuste por riesgo. La métrica objetivo es el **ratio de Sortino**, que penaliza solo la volatilidad negativa.

---

## Estructura del repositorio

```
bellmancapital/
├── agent.py               # Agente DQN completo (única entrega)
├── INFORME_FINAL.md       # Informe técnico de entrega
├── docs/
│   ├── ENTRENAMIENTO_V1.md
│   ├── ENTRENAMIENTO_V2.md
│   ├── ENTRENAMIENTO_V3.md
│   └── IMPLEMENTACION.md
├── src/                   # Infraestructura de solo lectura
│   ├── base.py
│   ├── baselines.py
│   ├── data.py
│   ├── env.py
│   └── metrics.py
├── data/raw/              # Precios OHLCV (15m, 30m, 1h)
├── models/                # Checkpoints y métricas de entrenamiento
├── configs/default.yaml   # Splits temporales y parámetros del entorno
└── tests/                 # 26 tests de validación de la entrega
```

---

## Diseño del agente

### Estado (22 dimensiones)

18 features de mercado (6 por activo: `log_ret`, `vol_21`, `mom_20`, `atr_14`, `vol_ratio`, `taker_buy_ratio`) más 4 pesos actuales del portafolio. Normalizadas con `StandardScaler` ajustado **solo sobre datos de entrenamiento** para evitar lookahead.

### Acciones (10 portafolios discretos)

| ID | Nombre | Pesos `[a0, a1, a2, cash]` |
|----|--------|---------------------------|
| 0 | All Cash | `[0.00, 0.00, 0.00, 1.00]` |
| 1 | Long A0 | `[1.00, 0.00, 0.00, 0.00]` |
| 2 | Long A1 | `[0.00, 1.00, 0.00, 0.00]` |
| 3 | Long A2 | `[0.00, 0.00, 1.00, 0.00]` |
| 4 | Equal Weight | `[0.33, 0.33, 0.33, 0.00]` |
| 5 | Conservative | `[0.25, 0.25, 0.25, 0.25]` |
| 6 | Short A0 | `[−0.50, 0.50, 0.50, 0.50]` |
| 7 | Short A1 | `[0.50, −0.50, 0.50, 0.50]` |
| 8 | Short A2 | `[0.50, 0.50, −0.50, 0.50]` |
| 9 | Long A1+A2 | `[0.00, 0.50, 0.50, 0.00]` |

### Función de recompensa — Sortino Diferencial (R3)

```
R = log_ret − 0.01 × turnover − 0.05 × max(0, −log_ret)
```

- `log_ret`: retorno logarítmico del portafolio en el paso actual
- `turnover`: suma de cambios absolutos en pesos (penaliza rebalanceo excesivo)
- `max(0, −log_ret)`: penaliza solo pérdidas (downside), no la volatilidad al alza

Las versiones R1 (log-retorno puro) y R2 (con drawdown acumulado) fueron descartadas durante el diagnóstico — ver `docs/ENTRENAMIENTO_V1.md` y `docs/ENTRENAMIENTO_V2.md`.

### Algoritmo — Double DQN

- Red online para selección de acción; red target para evaluación del valor Q
- Target sincronizada cada 1 000 pasos
- Replay buffer: 50 000 transiciones
- Entrenamiento: 600 000 pasos por fold
- Checkpoints cada 10 000 pasos con reanudación automática

---

## Protocolo de evaluación — Walk-Forward

| Fold | train_end | eval_end | train_rows | eval_rows |
|------|-----------|----------|-----------|----------|
| 1 | 2019-12-31 | 2020-12-31 | 17 438 | 8 795 |
| 2 | 2020-12-31 | 2021-12-31 | 26 210 | 8 776 |
| 3 | 2021-12-31 | 2022-12-31 | 34 963 | 8 783 |
| 4 | 2022-12-31 | 2023-12-31 | 43 723 | 8 784 |
| 5 | 2023-12-31 | 2025-12-31 | 52 484 | 17 544 |

Splits definidos en `configs/default.yaml` e inmutables.

---

## Resultados — Versión 3 (entrega final)

| Fold | sortino | cum_ret | max_dd |
|------|---------|---------|--------|
| 1 | **+1.8580** | **+127.86%** | **−23.33%** |
| 2 | −1.5286 | −99.78% | −99.85% |
| 3 | −3.5462 | −100.00% | −100.00% |
| 4 | −4.9175 | −99.98% | −99.98% |
| 5 | −3.8967 | −100.00% | −100.00% |

**Análisis de resultados:** Fold 1 (mercado alcista 2020) fue exitoso. Folds 2-5 fallaron por **desajuste de régimen** entre el período de entrenamiento y el período de evaluación: el agente aprendió a operar en un mercado que no es el que encontró en producción. Un factor agravante es que los folds 2-5 tienen conjuntos de entrenamiento 1.5–3× más grandes, por lo que con el mismo presupuesto de pasos completan muchos menos episodios completos, limitando la convergencia de los Q-valores.

---

## Reproducibilidad

```bash
# Instalar dependencias
uv sync

# Verificar que la entrega pasa todos los tests
uv run pytest tests/test_submission.py -v
```

Los 26 tests deben pasar. Datos de entrenamiento incluidos en `data/raw/`.

---

## Datos

Precios OHLCV + `taker_buy_ratio` para tres activos de criptomonedas (identidades anonimizadas). Intervalo utilizado: **1 hora**. Disponibles también: 15m, 30m.
