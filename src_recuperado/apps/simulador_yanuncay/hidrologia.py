"""Curva de descarga y estados de alerta por umbrales de caudal (sin clasificador independiente).

Q = a · max(H − H0, 0)^p, con H en cm y Q en m³/s. Los estados se asignan comparando el caudal
derivado del nivel (observado o pronosticado) con los umbrales oficiales de la estación.
"""
from __future__ import annotations

from typing import Any

import numpy as np

ESTADOS: tuple[str, ...] = ("normal", "prealerta", "alerta")
SIN_DATO = "sin dato"
COLORES_ESTADO: dict[str, str] = {
    "normal": "#2e9e44", "prealerta": "#f2b705", "alerta": "#d62828", SIN_DATO: "#8a8a8a",
}


def caudal_desde_nivel(h_cm: float | np.ndarray, curva: dict[str, float]) -> float | np.ndarray:
    """Caudal (m³/s) a partir del nivel (cm)."""
    h = np.asarray(h_cm, dtype=float)
    q = curva["a"] * np.power(np.clip(h - curva["H0"], 0.0, None), curva["p"])
    return float(q) if q.ndim == 0 else q


def nivel_desde_caudal(q_m3s: float | np.ndarray, curva: dict[str, float]) -> float | np.ndarray:
    """Nivel (cm) equivalente a un caudal (inversa de la curva)."""
    q = np.asarray(q_m3s, dtype=float)
    h = curva["H0"] + np.power(np.clip(q, 0.0, None) / curva["a"], 1.0 / curva["p"])
    return float(h) if h.ndim == 0 else h


def estado_desde_caudal(q_m3s: float | None, umbrales: dict[str, float]) -> str:
    """normal: Q < prealerta; prealerta: prealerta ≤ Q < alerta; alerta: Q ≥ alerta."""
    if q_m3s is None or not np.isfinite(q_m3s):
        return SIN_DATO
    if q_m3s >= umbrales["alerta"]:
        return "alerta"
    if q_m3s >= umbrales["prealerta"]:
        return "prealerta"
    return "normal"


def rango_estado(estado: str) -> int:
    """Orden de severidad (−1 para sin dato)."""
    return ESTADOS.index(estado) if estado in ESTADOS else -1


def umbrales_cm(config: dict[str, Any]) -> dict[str, float]:
    """Umbrales de caudal convertidos a nivel con la curva de descarga."""
    curva = config["curva_descarga"]
    return {k: float(nivel_desde_caudal(v, curva)) for k, v in config["umbrales_m3s"].items()}
