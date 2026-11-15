"""Cálculo causal de variables que replica el catálogo de la corrida preliminar del Yanuncay.

Respaldo del adaptador cuando ``src/data_prep/variables_yanuncay.py`` no está disponible.
Reglas (iguales a ``construir_features`` del preprocesamiento preliminar):
- Lluvia: acumulados móviles de 3, 6, 24 y 72 h (``rolling(h, min_periods=h).sum()``).
- Nivel: rezagos de 1, 3, 6, 12 y 24 h y deltas de 1 y 3 h.
- Tiempo: seno y coseno de la hora y del mes (índice en UTC, como en el entrenamiento).
- Indicadores ``<col>__imp`` (los entrega el pronosticador junto con la ventana imputada).

Todas las operaciones miran solo hacia atrás: el valor en t usa datos ≤ t.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

VENTANAS_LLUVIA_H: tuple[int, ...] = (3, 6, 24, 72)
LAGS_NIVEL_H: tuple[int, ...] = (1, 3, 6, 12, 24)
DELTAS_NIVEL_H: tuple[int, ...] = (1, 3)


def _dominio(col: str) -> str:
    variable = col.split("__")[1] if "__" in col else col
    if variable in ("Level_Avg", "Caudal"):
        return "nivel"
    if variable == "Lluvia_Tot":
        return "lluvia"
    return "meteo"


def codificacion_temporal(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Hora y mes cíclicos (en la zona horaria del índice; el entrenamiento usó UTC)."""
    h, m = indice.hour.values, indice.month.values
    return pd.DataFrame({
        "hora_sin": np.sin(2 * np.pi * h / 24), "hora_cos": np.cos(2 * np.pi * h / 24),
        "mes_sin": np.sin(2 * np.pi * (m - 1) / 12), "mes_cos": np.cos(2 * np.pi * (m - 1) / 12),
    }, index=indice)


def calcular_variables_preliminar(ventana: pd.DataFrame, columnas_base: Iterable[str] | None = None) -> pd.DataFrame:
    """Devuelve base + derivadas + tiempo + indicadores ``__imp`` en unidades físicas (sin escalar).

    ``ventana``: malla horaria continua con las columnas de sensores ya imputadas y, opcionalmente,
    las columnas ``<col>__imp``. Las primeras 72 h quedan con NaN en los acumulados largos.
    """
    if columnas_base is None:
        columnas_base = [c for c in ventana.columns if not c.endswith("__imp")]
    columnas_base = list(columnas_base)
    derivadas: dict[str, pd.Series] = {}
    for col in columnas_base:
        serie = ventana[col].astype(float)
        dom = _dominio(col)
        if dom == "lluvia":
            for h in VENTANAS_LLUVIA_H:
                derivadas[f"{col}_acum_{h}h"] = serie.rolling(window=h, min_periods=h).sum()
        elif dom == "nivel":
            for h in LAGS_NIVEL_H:
                derivadas[f"{col}_lag_{h}h"] = serie.shift(h)
            for h in DELTAS_NIVEL_H:
                derivadas[f"{col}_delta_{h}h"] = serie.diff(h)
    imp_cols = [f"{c}__imp" for c in columnas_base]
    indicadores = pd.DataFrame(
        {c: ventana[c].fillna(1).astype(np.int8) if c in ventana.columns
         else pd.Series(0, index=ventana.index, dtype=np.int8) for c in imp_cols},
        index=ventana.index,
    )
    return pd.concat(
        [ventana[columnas_base].astype(float), pd.DataFrame(derivadas, index=ventana.index),
         codificacion_temporal(ventana.index), indicadores],
        axis=1,
    )
