# -*- coding: utf-8 -*-
"""Datos y walk-forward de los modelos exploratorios del Yanuncay (anexo de la tesis).

La lógica oficial (malla horaria, etiqueta desplazada en el tiempo, ventana de 24 h válidas, bloques de val con
margen y early stopping en el último 10 %) solo existe dentro de ``src/models/modelos_yanuncay.ipynb``
(secciones 1 y 2). Aquí se reimplementa de forma fiel, sin cambios de criterio, y se añaden dos opciones que la
formulación oficial puede adoptar:

* ``objetivo="delta"``: el modelo aprende Δ = y_obs(t+τ) − nivel(t) y el nivel se reconstruye como
  nivel(t) + Δ̂. ``nivel(t)`` es la misma columna que usa la persistencia, de modo que Δ̂ = 0 es la persistencia.
* pesos de crecida: w = 1 + α · 1[y_obs(t+τ) > umbral] (umbral P95 de train, prealerta o alerta), opcionalmente
  multiplicados por una columna de pesos del parquet. Se normalizan a media 1 en las filas de ajuste.

La lista de variables se lee del catálogo ``catalogo_features_yanuncay.csv`` de la carpeta de datos.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TARGET = "LIM_Yanuncay-AJ-Tarqui__Level_Avg"
LOOKBACK_H = 24          # ventana de las redes (igual que el notebook oficial)
SEED = 42
N_BLOQUES = 5            # bloques del walk-forward dentro de val
FRACCION_ES = 0.10       # último 10 % cronológico para early stopping
EMPATE_SS = 0.01
COLUMNAS_CONTROL = ("y_obs", "valido")
TIPOS_NO_ENTRADA = {"objetivo_observado", "fila_valida_regla_72h"}
TIPOS_SOLO_TABULARES = {"lag", "tasa_de_cambio"}   # la ventana de 24 h ya contiene esa historia
OBJETIVOS = ("nivel", "delta")
UMBRALES_PESO = ("p95", "prealerta", "alerta")


# ---------------------------------------------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------------------------------------------
def leer_particion(data_dir: Path, nombre: str) -> pd.DataFrame:
    d = pd.read_parquet(Path(data_dir) / f"{nombre}_yanuncay.parquet")
    if "timestamp" in d.columns:
        d = d.set_index("timestamp")
    d.index = pd.DatetimeIndex(d.index, name="timestamp")
    d = d.sort_index()
    if "valido" not in d.columns:
        warnings.warn(f"{nombre} no tiene la columna 'valido'; se asume valido = 1.")
        d["valido"] = 1
    return d


def leer_catalogo(data_dir: Path, columnas: list[str],
                  excluir: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Devuelve (variables tabulares, variables por paso de tiempo) según el catálogo de la carpeta de datos."""
    cat = pd.read_csv(Path(data_dir) / "catalogo_features_yanuncay.csv")
    cat = cat[~cat["tipo"].isin(TIPOS_NO_ENTRADA) & ~cat["feature"].isin(COLUMNAS_CONTROL + tuple(excluir))]
    cat = cat.drop_duplicates("feature")
    faltan = [f for f in cat["feature"] if f not in columnas]
    if faltan:
        raise ValueError(f"El catálogo lista variables que no están en los parquet: {faltan[:5]}")
    sobran = [c for c in columnas if c not in set(cat["feature"]) and c not in COLUMNAS_CONTROL + tuple(excluir)]
    if sobran:
        warnings.warn(f"Columnas del parquet fuera del catálogo (no se usan): {sobran[:5]}")
    feat_tab = cat["feature"].tolist()
    solo_tab = cat["tipo"].isin(TIPOS_SOLO_TABULARES) | cat["feature"].str.contains("_lag_|_delta_", regex=True)
    feat_seq = cat.loc[~solo_tab, "feature"].tolist()
    if TARGET not in feat_tab or TARGET not in feat_seq:
        raise ValueError("El nivel en t debe ser variable de entrada (persistencia, objetivo Δ y N-HiTS)")
    return feat_tab, feat_seq


# ---------------------------------------------------------------------------------------------------------------
# Malla horaria global
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Malla:
    ts: pd.DatetimeIndex
    part: np.ndarray
    y_obs: np.ndarray          # nivel observado (escala min-max), NaN si no se observó
    pers: np.ndarray           # nivel en t (columna TARGET): base de la persistencia y del objetivo Δ
    x_tab: np.ndarray
    x_seq: np.ndarray          # variables por paso de tiempo, NaN -> 0 (nunca entran en una ventana válida)
    entrada_ok: np.ndarray
    ventana_ok: np.ndarray
    bordes: list
    margen_h: int
    horizontes: list
    tgt_min: float
    tgt_max: float
    p95_train_cm: float
    feat_tab: list
    feat_seq: list
    cfg_horizontes: dict
    peso_columna: Optional[np.ndarray] = None

    @property
    def rango_cm(self) -> float:
        return float(self.tgt_max - self.tgt_min)

    @property
    def n(self) -> int:
        return len(self.ts)

    def a_cm(self, y) -> np.ndarray:
        return np.asarray(y, dtype=float) * self.rango_cm + self.tgt_min

    def rmse_cm(self, y_esc, pred_esc) -> float:
        return float(np.sqrt(np.mean((np.asarray(pred_esc, float) - np.asarray(y_esc, float)) ** 2)) * self.rango_cm)

    def umbral_cm(self, nombre: str) -> float:
        if nombre == "p95":
            return self.p95_train_cm
        return float(self.cfg_horizontes[f"umbral_{nombre}_cm"])


def ventana_valida(entrada_ok: np.ndarray, lookback: int = LOOKBACK_H) -> np.ndarray:
    """True en t si las horas t-lookback+1 .. t (solo pasado y presente) son entradas válidas."""
    return pd.Series(entrada_ok.astype(np.int32)).rolling(lookback).sum().eq(lookback).to_numpy()


def construir_malla(particiones: dict, escaladores: pd.DataFrame, cfg_horizontes: dict,
                    feat_tab: list, feat_seq: list, columna_pesos: Optional[str] = None) -> Malla:
    for nombre, d in particiones.items():
        faltan = set(feat_tab) - set(d.columns)
        assert not faltan, f"{nombre} no tiene las columnas {sorted(faltan)[:5]}"
    tr, va, te = particiones["train"], particiones["val"], particiones["test"]
    assert tr.index.max() < va.index.min() < va.index.max() < te.index.min(), "Particiones no cronológicas"
    extra = [columna_pesos] if columna_pesos else []
    unidas = pd.concat([d[feat_tab + list(COLUMNAS_CONTROL) + extra].assign(particion=p)
                        for p, d in particiones.items()])
    malla = pd.date_range(unidas.index.min(), unidas.index.max(), freq="h", name="timestamp")
    G = unidas.reindex(malla)
    G["valido"] = G["valido"].fillna(0).astype(np.int8)
    G["particion"] = G["particion"].fillna("fuera")
    part = G["particion"].to_numpy()
    x_tab = G[feat_tab].to_numpy(np.float32)
    pers = G[TARGET].to_numpy(np.float32)
    y_obs = G["y_obs"].to_numpy(np.float32)
    entrada_ok = (G["valido"].to_numpy() == 1) & ~np.isnan(x_tab).any(axis=1) & ~np.isnan(pers)
    tgt_min, tgt_max = escaladores.loc[TARGET, ["min", "max"]].astype(float)
    val_ini = va.index.min()
    val_fin = va.index.max() + pd.Timedelta(hours=1)
    bordes = [val_ini + (val_fin - val_ini) * k / N_BLOQUES for k in range(N_BLOQUES + 1)]
    bordes = [b.floor("h") for b in bordes[:-1]] + [val_fin]
    y_cm_train = y_obs[(part == "train") & entrada_ok] * (tgt_max - tgt_min) + tgt_min
    peso_col = None
    if columna_pesos:
        peso_col = np.nan_to_num(G[columna_pesos].to_numpy(np.float32), nan=1.0)
    return Malla(ts=G.index, part=part, y_obs=y_obs, pers=pers, x_tab=x_tab,
                 x_seq=np.nan_to_num(G[feat_seq].to_numpy(np.float32)), entrada_ok=entrada_ok,
                 ventana_ok=ventana_valida(entrada_ok), bordes=bordes, margen_h=int(cfg_horizontes["margen_h"]),
                 horizontes=sorted({int(cfg_horizontes["tau1"]), int(cfg_horizontes["tau2"])}),
                 tgt_min=float(tgt_min), tgt_max=float(tgt_max),
                 p95_train_cm=float(np.nanpercentile(y_cm_train, 95)), feat_tab=feat_tab, feat_seq=feat_seq,
                 cfg_horizontes=cfg_horizontes, peso_columna=peso_col)


def variables_formulacion(formulacion: Path, feat_tab: list, feat_seq: list) -> tuple[list, list]:
    """Toma las listas (y su orden) de ``features_tabulares``/``features_secuencia`` del JSON de la formulación
    oficial. Deben ser exactamente las mismas variables que derivan del catálogo; si no, se detiene."""
    with open(formulacion, encoding="utf-8") as f:
        j = json.load(f)
    tab, seq = list(j["features_tabulares"]), list(j["features_secuencia"])
    for nombre, oficial, catalogo in (("tabulares", tab, feat_tab), ("secuencia", seq, feat_seq)):
        if set(oficial) != set(catalogo) or len(oficial) != len(set(oficial)):
            raise ValueError(f"Las variables {nombre} del JSON no coinciden con el catálogo: "
                             f"solo JSON {sorted(set(oficial) - set(catalogo))[:5]}, "
                             f"solo catálogo {sorted(set(catalogo) - set(oficial))[:5]}")
    return tab, seq


def cargar_malla(data_dir: Path, horizontes_json: Optional[Path] = None,
                 columna_pesos: Optional[str] = None, formulacion: Optional[Path] = None) -> Malla:
    data_dir = Path(data_dir)
    with open(horizontes_json or data_dir / "horizontes_yanuncay.json", encoding="utf-8") as f:
        cfg_h = json.load(f)
    particiones = {p: leer_particion(data_dir, p) for p in ("train", "val", "test")}
    excluir = (columna_pesos,) if columna_pesos else ()
    feat_tab, feat_seq = leer_catalogo(data_dir, list(particiones["train"].columns), excluir)
    if formulacion is not None:
        feat_tab, feat_seq = variables_formulacion(formulacion, feat_tab, feat_seq)
    escaladores = pd.read_csv(data_dir / "escaladores_yanuncay.csv").set_index("columna")
    return construir_malla(particiones, escaladores, cfg_h, feat_tab, feat_seq, columna_pesos)


# ---------------------------------------------------------------------------------------------------------------
# Horizonte: etiqueta, objetivo y pesos
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Horizonte:
    tau: int
    objetivo: str
    y: np.ndarray              # objetivo de entrenamiento (nivel o Δ, escala min-max)
    y_nivel: np.ndarray        # y_obs(t+τ) en escala min-max (para métricas)
    base: np.ndarray           # 0 (nivel) o nivel(t) (delta)
    pos: np.ndarray            # posiciones utilizables de la malla
    pesos: Optional[np.ndarray]

    def a_nivel(self, pred, pos) -> np.ndarray:
        """Reconstruye el nivel (escala min-max): ŷ_nivel = base(t) + ŷ."""
        return np.asarray(pred, dtype=float) + self.base[pos].astype(float)


def etiqueta_desplazada(y_obs: np.ndarray, part: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """y(t) = y_obs(t+τ) en la malla y máscara de 'misma partición en t y t+τ'."""
    y = np.full(len(y_obs), np.nan, dtype=np.float32)
    y[:-tau] = y_obs[tau:]
    misma = np.zeros(len(y_obs), dtype=bool)
    misma[:-tau] = part[tau:] == part[:-tau]
    return y, misma


def calcular_pesos(y_cm: np.ndarray, alpha: float, umbral_cm: float,
                   extra: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """w = 1 + α·1[y(t+τ) > umbral] (× columna de pesos opcional). None si no hay ponderación."""
    if alpha < 0:
        raise ValueError("pesos_alpha debe ser >= 0")
    if alpha == 0 and extra is None:
        return None
    w = 1.0 + alpha * np.nan_to_num(np.asarray(y_cm, float) > umbral_cm).astype(np.float32)
    if extra is not None:
        w = w * extra
    return w.astype(np.float32)


def preparar_horizonte(m: Malla, tau: int, objetivo: str = "nivel", pesos_alpha: float = 0.0,
                       pesos_umbral: str = "p95") -> Horizonte:
    if objetivo not in OBJETIVOS:
        raise ValueError(f"objetivo debe ser uno de {OBJETIVOS}")
    y_niv, misma = etiqueta_desplazada(m.y_obs, m.part, tau)
    pos = np.flatnonzero(m.ventana_ok & misma & ~np.isnan(y_niv))
    if objetivo == "delta":
        base = m.pers.astype(np.float32)
        y = y_niv - base
    else:
        base = np.zeros(m.n, dtype=np.float32)
        y = y_niv.copy()
    pesos = calcular_pesos(m.a_cm(y_niv), pesos_alpha, m.umbral_cm(pesos_umbral), m.peso_columna)
    return Horizonte(tau=tau, objetivo=objetivo, y=y, y_nivel=y_niv, base=base, pos=pos, pesos=pesos)


# ---------------------------------------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------------------------------------
def particion_fold(m: Malla, pos: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fold k (1..5): entrena con train + val anterior al bloque k menos el margen; evalúa en el bloque k."""
    ini, fin = m.bordes[k - 1], m.bordes[k]
    t, part = m.ts[pos], m.part[pos]
    evaluacion = (part == "val") & (t >= ini) & (t < fin)
    entrenamiento = (part == "train") | ((part == "val") & (t < ini - pd.Timedelta(hours=m.margen_h)))
    return pos[entrenamiento], pos[evaluacion]


def division_es(pos: np.ndarray, fraccion: float = FRACCION_ES) -> tuple[np.ndarray, np.ndarray]:
    """Último 10 % cronológico para early stopping (las posiciones ya están en orden temporal)."""
    n_es = max(1, int(round(fraccion * len(pos))))
    return pos[:-n_es], pos[-n_es:]


def indices_ventana(pos: np.ndarray, lookback: int = LOOKBACK_H) -> np.ndarray:
    """Matriz (n, lookback) de posiciones de la malla t-lookback+1 .. t para cada t (solo pasado)."""
    return np.asarray(pos)[:, None] + np.arange(-lookback + 1, 1)[None, :]


def submuestra(pos: np.ndarray, n: int, semilla: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return np.sort(rng.choice(pos, min(n, len(pos)), replace=False))
