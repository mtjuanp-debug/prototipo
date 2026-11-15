# -*- coding: utf-8 -*-
"""Espacios de búsqueda aleatoria de los modelos exploratorios (anexo).

Mismo esquema que el notebook oficial: un candidato inicial razonable por familia y el resto muestreado al azar
con una semilla por familia (42, índice). Las mismas configuraciones se prueban en ambos horizontes.
Presupuesto oficial: 30 configuraciones en árboles y 12 en redes (por familia y horizonte).

| Familia | Espacio |
|---|---|
| LightGBM | learning_rate 0,01–0,1 (log), hasta 3000 rondas con early stopping 50, num_leaves 15–255, max_depth {-1, 4–12}, min_child_samples 10–100, subsample 0,6–1 (bagging cada ronda), colsample_bytree 0,5–1, reg_lambda 0–10, reg_alpha 0–1 |
| CatBoost | learning_rate 0,01–0,1 (log), hasta 3000 iteraciones con early stopping 50, depth 4–10, l2_leaf_reg 1–10, random_strength 0–2, bagging_temperature 0–1 |
| Comunes a las redes | lr 3e-4–1e-3 (log), lote {64, 128, 256, 512}, hasta 100 épocas, paciencia 8–10, Adam o AdamW (wd 1e-4), dropout 0–0,3 |
| GRU | oculto {32, 64, 128}, capas 1–3 |
| TCN | canales {32, 64, 128}, niveles 3–5 (dilatación 2^i), kernel {2, 3}; campo receptivo ≥ 24 h |
| N-HiTS | pooling por pila {[8,4,1], [4,2,1], [2,1,1]}, bloques por pila 1–2, ancho del MLP {128, 256, 512}, capas del MLP 2 |
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SEMILLA = 42
FAMILIAS = ("LGBM", "CAT", "GRU", "TCN", "NHITS")
FAMILIAS_ARBOL = ("LGBM", "CAT")
FAMILIAS_RED = ("GRU", "TCN", "NHITS")
NOMBRE_LARGO = {"LGBM": "LightGBM", "CAT": "CatBoost", "GRU": "GRU", "TCN": "TCN", "NHITS": "N-HiTS"}
LOOKBACK_H = 24

ARBOL_COMUN = {
    "LGBM": dict(n_estimators=3000, early_stopping_rounds=50, objective="regression", subsample_freq=1,
                 random_state=SEMILLA, n_jobs=-1, verbose=-1),
    "CAT": dict(iterations=3000, early_stopping_rounds=50, loss_function="RMSE", random_seed=SEMILLA,
                thread_count=-1),
}


def _log_uniforme(rng: np.random.Generator, a: float, b: float) -> float:
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def muestrear_lgbm(rng: np.random.Generator) -> dict:
    return dict(learning_rate=round(_log_uniforme(rng, 0.01, 0.1), 4),
                num_leaves=int(rng.integers(15, 256)),
                max_depth=[-1, 4, 6, 8, 10, 12][int(rng.integers(6))],
                min_child_samples=int(rng.integers(10, 101)),
                subsample=round(float(rng.uniform(0.6, 1.0)), 2),
                colsample_bytree=round(float(rng.uniform(0.5, 1.0)), 2),
                reg_lambda=round(float(rng.uniform(0.0, 10.0)), 2),
                reg_alpha=round(float(rng.uniform(0.0, 1.0)), 2))


def muestrear_cat(rng: np.random.Generator) -> dict:
    return dict(learning_rate=round(_log_uniforme(rng, 0.01, 0.1), 4),
                depth=int(rng.integers(4, 11)),
                l2_leaf_reg=round(float(rng.uniform(1.0, 10.0)), 2),
                random_strength=round(float(rng.uniform(0.0, 2.0)), 2),
                bagging_temperature=round(float(rng.uniform(0.0, 1.0)), 2))


def _comunes_red(rng: np.random.Generator) -> dict:
    optimizador = ["adam", "adamw"][int(rng.integers(2))]
    return dict(dropout=round(float(rng.uniform(0.0, 0.3)), 2),
                batch_size=[64, 128, 256, 512][int(rng.integers(4))], epocas=100,
                patience=int(rng.integers(8, 11)), lr=round(_log_uniforme(rng, 3e-4, 1e-3), 6),
                optimizador=optimizador, weight_decay=1e-4 if optimizador == "adamw" else 0.0)


def muestrear_gru(rng: np.random.Generator) -> dict:
    return dict(hidden_size=[32, 64, 128][int(rng.integers(3))], num_layers=int(rng.integers(1, 4)),
                **_comunes_red(rng))


def campo_receptivo_tcn(n_niveles: int, kernel: int) -> int:
    return 1 + 2 * (kernel - 1) * (2 ** n_niveles - 1)


def muestrear_tcn(rng: np.random.Generator) -> dict:
    while True:
        kernel, n_niveles = [2, 3][int(rng.integers(2))], int(rng.integers(3, 6))
        if campo_receptivo_tcn(n_niveles, kernel) >= LOOKBACK_H:
            break
    return dict(canales=[32, 64, 128][int(rng.integers(3))], n_niveles=n_niveles, kernel=kernel,
                **_comunes_red(rng))


def muestrear_nhits(rng: np.random.Generator) -> dict:
    return dict(pooling=[[8, 4, 1], [4, 2, 1], [2, 1, 1]][int(rng.integers(3))],
                n_bloques=int(rng.integers(1, 3)), ancho=[128, 256, 512][int(rng.integers(3))], n_capas=2,
                **_comunes_red(rng))


_E1_RED = dict(dropout=0.2, batch_size=512, epocas=100, patience=10, lr=1e-3, optimizador="adam",
               weight_decay=0.0)
INICIALES: dict[str, dict[str, dict]] = {
    "LGBM": {"LGBM_E1": dict(learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=20,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0)},
    "CAT": {"CAT_E1": dict(learning_rate=0.05, depth=6, l2_leaf_reg=3.0, random_strength=1.0,
                           bagging_temperature=1.0)},
    "GRU": {"GRU_E1": dict(hidden_size=64, num_layers=2, **_E1_RED)},
    "TCN": {"TCN_E1": dict(canales=64, n_niveles=4, kernel=2, **_E1_RED)},
    "NHITS": {"NHITS_E1": dict(pooling=[4, 2, 1], n_bloques=1, ancho=256, n_capas=2, **_E1_RED)},
}
MUESTREADORES: dict[str, Callable[[np.random.Generator], dict]] = {
    "LGBM": muestrear_lgbm, "CAT": muestrear_cat, "GRU": muestrear_gru, "TCN": muestrear_tcn,
    "NHITS": muestrear_nhits}


def reducir_humo(familia: str, cfg: dict, epocas: int = 3) -> dict:
    """Versión reducida para la prueba de humo (pocas épocas); los árboles se acortan en ARBOL_COMUN."""
    cfg = dict(cfg)
    if familia in FAMILIAS_RED:
        cfg.update(epocas=epocas, patience=1)
    return cfg


def generar_candidatos(familia: str, n_total: int, humo: bool = False) -> list[tuple[str, dict]]:
    if familia not in FAMILIAS:
        raise ValueError(f"Familia desconocida: {familia}")
    rng = np.random.default_rng([SEMILLA, 100 + FAMILIAS.index(familia)])   # una semilla por familia
    candidatos = list(INICIALES[familia].items())[:n_total]
    i = 1
    while len(candidatos) < n_total:
        candidatos.append((f"{familia}_c{i:02d}", MUESTREADORES[familia](rng)))
        i += 1
    if humo:
        candidatos = [(cid, reducir_humo(familia, cfg)) for cid, cfg in candidatos]
    return candidatos


def comun_arbol(familia: str, humo: bool = False) -> dict:
    comun = dict(ARBOL_COMUN[familia])
    if humo:
        clave = "n_estimators" if familia == "LGBM" else "iterations"
        comun.update({clave: 60, "early_stopping_rounds": 10})
    return comun
