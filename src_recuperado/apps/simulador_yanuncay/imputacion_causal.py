"""Control de calidad e imputación CAUSAL para la ventana móvil del pronosticador.

Solo se usan datos del instante t o anteriores (nada de PCHIP ni interpolación con anclas futuras):
- Nivel: límites físicos, detección de valor congelado y relleno hacia adelante limitado.
  Si el hueco supera el límite, la hora queda sin dato y no se emite pronóstico.
- Lluvia: promedio ponderado de los pluviómetros vecinos en la misma hora (pesos = correlación en
  train, al estilo CCW); si ninguno reporta, 0 mm (respaldo usado en el preprocesamiento).
- Meteorología: relleno hacia adelante corto, luego regresión lineal con la misma variable en la
  estación vecina (ajustada en train) y, por último, climatología hora × mes de train.
Los parámetros se ajustan una sola vez con la partición de entrenamiento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from configuracion import estacion_de, variable_de


@dataclass
class ParametrosImputacion:
    pesos_lluvia: dict[str, dict[str, float]] = field(default_factory=dict)
    regresion_vecina: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)
    climatologia: dict[str, pd.Series] = field(default_factory=dict)


def ajustar_parametros(train: pd.DataFrame, config: dict[str, Any]) -> ParametrosImputacion:
    """Ajusta pesos de vecinos, regresiones y climatología con datos observados de train."""
    p = ParametrosImputacion()
    lluvias = list(config["estaciones"]["lluvia"])
    for col in lluvias:
        p.pesos_lluvia[col] = {}
        for otra in lluvias:
            if otra != col:
                r = train[col].corr(train[otra])
                p.pesos_lluvia[col][otra] = float(max(r, 0.0)) if np.isfinite(r) else 0.0
    meteo = list(config["estaciones"]["meteo"])
    for col in meteo:
        vecinas = [c for c in meteo if variable_de(c) == variable_de(col) and estacion_de(c) != estacion_de(col)]
        ajustes = []
        for v in vecinas:
            par = train[[col, v]].dropna()
            if len(par) > 100:
                b, a = np.polyfit(par[v].to_numpy(), par[col].to_numpy(), 1)
                ajustes.append((v, float(a), float(b)))
        p.regresion_vecina[col] = ajustes
        clima = train[col].groupby([train.index.month, train.index.hour]).mean()
        p.climatologia[col] = clima
    return p


def _aplicar_limites(ventana: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    salida = ventana.copy()
    for col in salida.columns:
        lim = config["limites_fisicos"].get(variable_de(col))
        if lim is not None:
            salida[col] = salida[col].where(salida[col].between(lim[0], lim[1]))
    return salida


def _mascara_congelada(s: pd.Series, min_h: int) -> pd.Series:
    """True en las horas de una racha de valores idénticos de longitud ≥ ``min_h`` (hasta t)."""
    grupo = (s != s.shift()).cumsum()
    largo = s.groupby(grupo).transform("size")
    return s.notna() & (largo >= min_h)


def control_calidad(ventana: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Límites físicos y valores congelados -> NaN."""
    salida = _aplicar_limites(ventana, config)
    imp = config["imputacion"]
    for col in salida.columns:
        var = variable_de(col)
        if var == "Level_Avg":
            salida[col] = salida[col].mask(_mascara_congelada(salida[col], int(imp["congelado_nivel_min_h"])))
        elif var == "TempAire_Avg":
            salida[col] = salida[col].mask(_mascara_congelada(salida[col], int(imp["congelado_meteo_min_h"])))
    return salida


def imputar(ventana: pd.DataFrame, params: ParametrosImputacion,
            config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (valores imputados, indicadores ``__imp``). Los NaN restantes no son imputables."""
    limpio = control_calidad(ventana, config)
    imp = config["imputacion"]
    salida = limpio.copy()
    indicadores = pd.DataFrame({f"{c}__imp": limpio[c].isna().astype(np.int8) for c in limpio.columns},
                               index=limpio.index)
    for col in limpio.columns:
        var = variable_de(col)
        s = limpio[col]
        if not s.isna().any():
            continue
        if var == "Level_Avg":
            salida[col] = s.ffill(limit=int(imp["ffill_nivel_max_h"]))
        elif var == "Lluvia_Tot":
            pesos = params.pesos_lluvia.get(col, {})
            num = pd.Series(0.0, index=s.index)
            den = pd.Series(0.0, index=s.index)
            for otra, w in pesos.items():
                if otra in limpio.columns and w > 0:
                    ok = limpio[otra].notna()
                    num += np.where(ok, limpio[otra].fillna(0.0) * w, 0.0)
                    den += np.where(ok, w, 0.0)
            vecinos = (num / den.replace(0.0, np.nan))
            salida[col] = s.fillna(vecinos).fillna(0.0).clip(lower=0.0)
        else:
            r = s.ffill(limit=int(imp["ffill_meteo_max_h"]))
            for vecina, a, b in params.regresion_vecina.get(col, []):
                if vecina in limpio.columns:
                    r = r.fillna(a + b * limpio[vecina])
            clima = params.climatologia.get(col)
            if clima is not None and r.isna().any():
                claves = list(zip(r.index.month, r.index.hour))
                r = r.fillna(pd.Series([clima.get(k, np.nan) for k in claves], index=r.index))
            lim = config["limites_fisicos"].get(var)
            salida[col] = r.clip(lim[0], lim[1]) if lim is not None else r
    return salida, indicadores
