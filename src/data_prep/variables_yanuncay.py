# -*- coding: utf-8 -*-
"""Compatibilidad: variables del Yanuncay = ``variables_rio`` con el objetivo del Yanuncay por defecto.

La implementación vive en ``src/data_prep/variables_rio.py`` (genérica para los cuatro ríos). Este módulo se conserva
porque la aplicación ``src/apps/simulador_yanuncay`` lo carga por ruta (``importlib``) y las pruebas lo importan por
nombre. Reexporta todo ``variables_rio`` y mantiene los valores por defecto anteriores:

- ``ConfigVariables(entradas=...)`` sin ``target`` usa ``LIM_Yanuncay-AJ-Tarqui__Level_Avg``;
- ``desescalar`` y ``desescalar_cambio`` usan por defecto la fila de esa columna en ``escaladores_yanuncay.csv``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# La aplicación carga este archivo por ruta, sin src/data_prep en sys.path: se añade para importar variables_rio.
_CARPETA = str(Path(__file__).resolve().parent)
if _CARPETA not in sys.path:
    sys.path.insert(0, _CARPETA)

import variables_rio as _vr  # noqa: E402
from variables_rio import *  # noqa: E402,F401,F403

TARGET_POR_DEFECTO = "LIM_Yanuncay-AJ-Tarqui__Level_Avg"


@dataclass(frozen=True)
class ConfigVariables(_vr.ConfigVariables):
    """``variables_rio.ConfigVariables`` con el objetivo del Yanuncay como valor por defecto."""

    target: str | None = TARGET_POR_DEFECTO


def desescalar(valores: np.ndarray | pd.Series, escaladores: pd.DataFrame,
               columna: str = TARGET_POR_DEFECTO) -> np.ndarray:
    """Vuelve a unidades físicas (cm para el nivel)."""
    return _vr.desescalar(valores, escaladores, columna)


def desescalar_cambio(delta: np.ndarray | pd.Series, escaladores: pd.DataFrame,
                      columna: str = TARGET_POR_DEFECTO) -> np.ndarray:
    """Convierte un cambio Δ escalado a cm (solo se multiplica por el rango; el mínimo se cancela)."""
    return _vr.desescalar_cambio(delta, escaladores, columna)


__all__ = [*_vr.__all__, "TARGET_POR_DEFECTO"]
