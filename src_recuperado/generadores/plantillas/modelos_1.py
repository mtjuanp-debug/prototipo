# -*- coding: utf-8 -*-
"""Plantilla del notebook de modelos (src/models/modelos_<slug>.ipynb): parte 1 de 3.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("md", r"""
# Entrenamiento de modelos — Río @@nombre@@

Entrena los modelos de `src/models/README.MD` para predecir el **nivel** de `@@estacion@@`
con los horizontes **τ₁ y τ₂** fijados en `Docs/resources/horizontes_@@slug@@.json` (a partir del tiempo de
respuesta Tr estimado en train). Estrategia **directa**: un modelo por horizonte.

| Familia | Candidatos iniciales | Búsqueda aleatoria | Curva de sobreajuste guardada |
|---|---|---|---|
| Persistencia (referencia) | — | — | — |
| Random Forest | RF_E1 | `N_ITER_ARBOLES` configuraciones | Error vs n.º de árboles + curva de `max_depth` |
| XGBoost | XGB_E1 | `N_ITER_ARBOLES` | `evals_result()` por ronda (early stopping 50) |
| AdaBoost.R2 | ADA_E1 | `N_ITER_ARBOLES` | `staged_predict()` por estimador |
| RNN base | RNN_base (E1 con celda RNN) | `N_ITER_REDES` | Pérdida por época |
| LSTM | LSTM_E1, E2, E3 | `N_ITER_REDES` | Pérdida por época |

**Datos:** `Docs/resources/{train,val,test}_@@slug@@.parquet`, salida de `preprocesamiento_@@slug@@.ipynb`.
- **Objetivo:** `y = y_obs(t+τ)`, desplazado **en el tiempo** sobre una malla horaria completa (las horas del
  margen entre particiones no existen en los archivos). Solo entran filas con `valido = 1`, `y_obs(t+τ)`
  observado y, para todas las familias, una ventana de 24 h consecutivas válidas (así árboles y redes se
  evalúan sobre las mismas filas).
- **Formulación (fase A):** con `@@PREFIJO@@OBJETIVO=delta` los modelos aprenden el **cambio**
  Δ = y_obs(t+τ) − nivel(t), donde nivel(t) es la entrada del nivel en t (observada o imputada). El pronóstico de
  nivel es nivel(t) + Δ̂, y la persistencia equivale a Δ̂ = 0. Las predicciones guardadas, las métricas y la
  selección siguen en **nivel (cm)**. Con `@@PREFIJO@@PESO_ALFA = α > 0` cada fila de entrenamiento pesa
  w = 1 + α·𝟙[y_obs(t+τ) ≥ P95 de train  o  lluvia areal(t) ≥ 0,1 mm] (`sample_weight` en RF, XGBoost y AdaBoost;
  MSE ponderada en RNN/LSTM). El early stopping y la selección usan el error sin ponderar.
@@bloque_ablacion@@
- **Búsqueda de hiperparámetros (tesis):** búsqueda aleatoria con *walk-forward* expansivo. Val se divide en
  5 bloques consecutivos de igual duración. El fold k entrena con train + los bloques de val anteriores a k,
  deja un **margen** de `margen_h` horas antes del bloque k y evalúa en el bloque k. Test no participa.
- **Early stopping** (XGBoost y redes): último 10 % cronológico de los datos de entrenamiento de cada fold.
- **Selección:** por familia y horizonte, la configuración con mayor **SS medio** entre folds
  (SS = 1 − RMSE_modelo / RMSE_persistencia, sobre las mismas filas). El mejor modelo global es el de mayor SS
  promedio entre folds y entre horizontes; si la diferencia es menor de 0,01 gana el de menor RMSE en niveles
  mayores al P95 de `y_obs` de train.
- **Modelo final:** la mejor configuración se reentrena con train + val (XGBoost y redes con early stopping en
  el último 10 % cronológico) y **test se predice una sola vez**. Las métricas de test están en
  `src/metrics/@@slug@@_models_metric.ipynb`.

**Variables de entorno**

| Variable | Uso | Valor por defecto |
|---|---|---|
| `@@PREFIJO@@RAPIDO=1` | Versión reducida para verificar el código (2 folds, pocos árboles y épocas) | apagado |
| `@@PREFIJO@@DATA_DIR` | Carpeta con los parquet, `escaladores_@@slug@@.csv` y el json de horizontes | `Docs/resources` |
| `@@PREFIJO@@HORIZONTES_JSON` | Ruta del json de horizontes | `<DATA_DIR>/horizontes_@@slug@@.json` |
| `@@PREFIJO@@OUT_DIR` | Carpeta de salidas | `Docs/resources/modelos_@@slug@@[_rapido]` |
| `@@PREFIJO@@N_ITER_ARBOLES` / `@@PREFIJO@@N_ITER_REDES` | Configuraciones por familia y horizonte (incluye los candidatos iniciales) | 30 / 12 (rápido: 2 / 1) |
| `@@PREFIJO@@FOLDS` | Folds a evaluar, p. ej. `1,2,3,4,5` | todos (rápido: `4,5`) |
| `@@PREFIJO@@FAMILIAS` | Subconjunto de familias, p. ej. `RandomForest,LSTM` | todas |
| `@@PREFIJO@@XGB_GPU=0` | Fuerza XGBoost en CPU | GPU si está disponible |
| `@@PREFIJO@@GUARDAR_RF=0` | No guarda los Random Forest finales (archivos grandes, joblib comprimido) | se guardan |
| `@@PREFIJO@@OBJETIVO` | `delta` (Δ = y(t+τ) − nivel(t)) o `nivel` (corrida preliminar) | `delta` |
| `@@PREFIJO@@VARIABLES` | `todas` o `base` (sin las variables de lluvia extendidas de la fase A) | `todas` |
| `@@PREFIJO@@PESO_ALFA` | α de los pesos de crecida (0 = sin pesos) | 0 |
| `@@PREFIJO@@SOLO_BUSQUEDA=1` | Solo búsqueda walk-forward (ablación): sin modelo final ni test | apagado |
| `@@PREFIJO@@RF_MAX_ARBOLES` | Tope de `n_estimators` de RF (solo para la ablación reducida) | sin tope |
| `@@PREFIJO@@CONFIGS` | Solo las configuraciones con estos ids, p. ej. `LSTM_c08` (ablación reducida) | todas |
"""),
    ("md", r"""
## 0. Configuración
"""),
    ("code", r'''
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import gc
import json
import platform
import time
import warnings
from pathlib import Path

# torch se importa antes que numpy/sklearn: en Windows evita el error de DLL (WinError 1114) de c10.dll
import torch
import torch.nn as nn

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor

try:
    import shap
except Exception as error:   # SHAP es opcional: si no está instalado se omite shap_importancia.csv
    shap = None
    print(f"Aviso: no se pudo importar shap ({error}). Se omite la importancia SHAP.")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

RAPIDO = os.environ.get("@@PREFIJO@@RAPIDO") == "1"


def _raiz_repo():
    actual = Path.cwd().resolve()
    for candidato in [actual, *actual.parents]:
        if (candidato / ".git").exists() and (candidato / "Docs" / "resources").exists():
            return candidato
    return None


def _ruta_env(nombre, por_defecto):
    valor = os.environ.get(nombre)
    if valor:
        ruta = Path(valor)
        # Las rutas relativas se interpretan desde la raíz del repositorio, no desde la carpeta del notebook
        return ruta if ruta.is_absolute() or ROOT is None else ROOT / ruta
    if por_defecto is None:
        raise FileNotFoundError(f"No se encontró Docs/resources del repositorio; defina la variable {nombre}.")
    return Path(por_defecto)


def _entero_env(nombre, por_defecto):
    return int(os.environ.get(nombre) or por_defecto)


ROOT = _raiz_repo()
RES_DIR = None if ROOT is None else ROOT / "Docs" / "resources"
DATA_DIR = _ruta_env("@@PREFIJO@@DATA_DIR", RES_DIR)
HORIZONTES_JSON = _ruta_env("@@PREFIJO@@HORIZONTES_JSON", DATA_DIR / "horizontes_@@slug@@.json")
OUT_DIR = _ruta_env("@@PREFIJO@@OUT_DIR", None if RES_DIR is None else
                    RES_DIR / ("modelos_@@slug@@_rapido" if RAPIDO else "modelos_@@slug@@"))
MODELOS_DIR = OUT_DIR / "modelos"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "@@target@@"
LOOKBACK_H = 24          # ventana de las redes recurrentes
SEED = 42
N_BLOQUES = 5            # bloques del walk-forward dentro de val
FRACCION_ES = 0.10       # último 10 % cronológico para early stopping
EMPATE_SS = 0.01         # diferencia de SS por debajo de la cual se desempata con niveles altos

with open(HORIZONTES_JSON, encoding="utf-8") as f:
    CFG_HORIZONTES = json.load(f)
HORIZONTES = sorted({int(CFG_HORIZONTES["tau1"]), int(CFG_HORIZONTES["tau2"])})
MARGEN_H = int(CFG_HORIZONTES["margen_h"])

N_ITER_ARBOLES = _entero_env("@@PREFIJO@@N_ITER_ARBOLES", 2 if RAPIDO else 30)
N_ITER_REDES = _entero_env("@@PREFIJO@@N_ITER_REDES", 1 if RAPIDO else 12)
_folds_env = os.environ.get("@@PREFIJO@@FOLDS")
FOLDS = ([int(k) for k in _folds_env.split(",")] if _folds_env else
         ([N_BLOQUES - 1, N_BLOQUES] if RAPIDO else list(range(1, N_BLOQUES + 1))))
assert all(1 <= k <= N_BLOQUES for k in FOLDS)

TODAS_FAMILIAS = ["RandomForest", "XGBoost", "AdaBoost", "RNN", "LSTM"]
FAMILIAS_ARBOL = ["RandomForest", "XGBoost", "AdaBoost"]
FAMILIAS_RED = ["RNN", "LSTM"]
_familias_env = os.environ.get("@@PREFIJO@@FAMILIAS")
FAMILIAS = [f.strip() for f in _familias_env.split(",")] if _familias_env else TODAS_FAMILIAS
assert set(FAMILIAS) <= set(TODAS_FAMILIAS), FAMILIAS
CORTO = {"Persistencia": "Persistencia", "RandomForest": "RF", "XGBoost": "XGB", "AdaBoost": "ADA",
         "RNN": "RNN", "LSTM": "LSTM"}
GUARDAR_RF = os.environ.get("@@PREFIJO@@GUARDAR_RF", "1") == "1"
# Formulación (fase A). Los valores por defecto son los ganadores de la ablación walk-forward con RF y XGBoost
# (Docs/resources/modelos_yanuncay/tablas/ablacion_walkforward.csv): objetivo Δ + variables de lluvia
# extendidas, sin pesos (α = 2 y α = 5 bajaron el SS medio de 0,143 a 0,126 y 0,113).
OBJETIVO = os.environ.get("@@PREFIJO@@OBJETIVO", "delta")
VARIABLES = os.environ.get("@@PREFIJO@@VARIABLES", "todas")
PESO_ALFA = float(os.environ.get("@@PREFIJO@@PESO_ALFA", "0"))
SOLO_BUSQUEDA = os.environ.get("@@PREFIJO@@SOLO_BUSQUEDA") == "1"
RF_MAX_ARBOLES = _entero_env("@@PREFIJO@@RF_MAX_ARBOLES", 0)
# Subconjunto de configuraciones por id (p. ej. "LSTM_c08" para la ablación reducida de la LSTM)
CONFIGS_FILTRO = {c.strip() for c in os.environ.get("@@PREFIJO@@CONFIGS", "").split(",") if c.strip()}
UMBRAL_LLUVIA_PESO_MM = 0.1
assert OBJETIVO in ("delta", "nivel") and VARIABLES in ("todas", "base") and PESO_ALFA >= 0

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))


def _dispositivo_xgb():
    if os.environ.get("@@PREFIJO@@XGB_GPU", "1") != "1" or not torch.cuda.is_available():
        return "cpu"
    try:
        xgb.XGBRegressor(n_estimators=2, tree_method="hist", device="cuda").fit(
            np.random.rand(64, 3), np.random.rand(64))
        return "cuda"
    except Exception:
        return "cpu"


XGB_DEVICE = _dispositivo_xgb()
print(f"Modo rápido: {RAPIDO} | Redes en: {DEVICE} | XGBoost en: {XGB_DEVICE}")
print(f"Horizontes τ: {HORIZONTES} h | margen: {MARGEN_H} h | folds: {FOLDS} | familias: {FAMILIAS}")
print(f"Presupuesto: {N_ITER_ARBOLES} configuraciones por árbol y {N_ITER_REDES} por red (por horizonte)")
print(f"Formulación: objetivo = {OBJETIVO} | variables = {VARIABLES} | α pesos = {PESO_ALFA} "
      f"| solo búsqueda = {SOLO_BUSQUEDA} | tope árboles RF = {RF_MAX_ARBOLES or '-'}")
print("Datos:", DATA_DIR)
print("Salidas:", OUT_DIR)
'''),
    ("md", r"""
### Espacios de búsqueda (solo regresión)

Los antiguos escenarios del README se incluyen como **candidatos iniciales** y el resto se muestrea al azar
(semilla 42). Las mismas configuraciones se prueban en ambos horizontes.

| Familia | Espacio |
|---|---|
| RF | `n_estimators` 200–500, `max_depth` {10, 12, 15, 20, None}, `min_samples_leaf` 1–10, `max_features` 0,3–0,8 |
| XGBoost | `learning_rate` 0,01–0,1 (log), hasta 3000 rondas con early stopping 50, `max_depth` 3–8, `subsample` 0,6–1, `colsample_bytree` 0,5–1, `min_child_weight` 1–10, `reg_lambda` 1–10, `tree_method="hist"` |
| AdaBoost.R2 | árbol base con `max_depth` 3–6, `n_estimators` 25–150, `learning_rate` 0,01–0,3, pérdida lineal |
| RNN / LSTM | oculto {32, 64, 128}, capas 1–3, dropout 0–0,3, lr 3e-4–1e-3 (log), lote {64, 128, 256, 512}, hasta 100 épocas, paciencia 8–10, Adam o AdamW (wd 1e-4) |

Comunes a las redes: pérdida MSE, recorte del gradiente a norma 1,0, `ReduceLROnPlateau` (factor 0,5,
paciencia 3) y cabeza densa de 32 neuronas con ReLU cuyo sesgo se inicializa con la media del objetivo.
"""),
    ("code", r'''
def _log_uniforme(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def muestrear_rf(rng):
    return dict(n_estimators=int(rng.integers(200, 501)),
                max_depth=[10, 12, 15, 20, None][int(rng.integers(5))],
                min_samples_leaf=int(rng.integers(1, 11)),
                max_features=round(float(rng.uniform(0.3, 0.8)), 2))


def muestrear_xgb(rng):
    return dict(learning_rate=round(_log_uniforme(rng, 0.01, 0.1), 4),
                max_depth=int(rng.integers(3, 9)),
                subsample=round(float(rng.uniform(0.6, 1.0)), 2),
                colsample_bytree=round(float(rng.uniform(0.5, 1.0)), 2),
                min_child_weight=int(rng.integers(1, 11)),
                reg_lambda=round(float(rng.uniform(1.0, 10.0)), 2))


def muestrear_ada(rng):
    return dict(max_depth=int(rng.integers(3, 7)), n_estimators=int(rng.integers(25, 151)),
                learning_rate=round(float(rng.uniform(0.01, 0.3)), 3), loss="linear")


def muestrear_red(rng, celda):
    optimizador = ["adam", "adamw"][int(rng.integers(2))]
    return dict(celda=celda, hidden_size=[32, 64, 128][int(rng.integers(3))],
                num_layers=int(rng.integers(1, 4)), dropout=round(float(rng.uniform(0.0, 0.3)), 2),
                batch_size=[64, 128, 256, 512][int(rng.integers(4))], epocas=100,
                patience=int(rng.integers(8, 11)), lr=round(_log_uniforme(rng, 3e-4, 1e-3), 6),
                optimizador=optimizador, weight_decay=1e-4 if optimizador == "adamw" else 0.0)


_E1_RED = dict(hidden_size=64, num_layers=2, dropout=0.2, batch_size=512, epocas=12, patience=3, lr=1e-3,
               optimizador="adam", weight_decay=0.0)
INICIALES = {
    "RandomForest": {"RF_E1": dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features=0.5)},
    "XGBoost": {"XGB_E1": dict(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=1, reg_lambda=1)},
    "AdaBoost": {"ADA_E1": dict(max_depth=3, n_estimators=200, learning_rate=0.1, loss="linear")},
    "RNN": {"RNN_base": dict(celda="rnn", **_E1_RED)},
    "LSTM": {
        "LSTM_E1": dict(celda="lstm", **_E1_RED),
        "LSTM_E2": dict(celda="lstm", hidden_size=32, num_layers=3, dropout=0.2, batch_size=32, epocas=100,
                        patience=10, lr=1e-3, optimizador="adam", weight_decay=0.0),
        "LSTM_E3": dict(celda="lstm", hidden_size=128, num_layers=3, dropout=0.2, batch_size=256, epocas=100,
                        patience=10, lr=1e-4, optimizador="adamw", weight_decay=1e-4),
    },
}
MUESTREADORES = {"RandomForest": muestrear_rf, "XGBoost": muestrear_xgb, "AdaBoost": muestrear_ada,
                 "RNN": lambda rng: muestrear_red(rng, "rnn"), "LSTM": lambda rng: muestrear_red(rng, "lstm")}
XGB_COMUN = dict(n_estimators=3000, early_stopping_rounds=50, objective="reg:squarederror", eval_metric="rmse",
                 tree_method="hist", n_jobs=-1, random_state=SEED, device=XGB_DEVICE)
PACIENCIA_LR = 3
RF_PROFUNDIDADES = [5, 10, 15, 20, 30, None]   # curva de validación de max_depth (README)
RF_ARBOLES_CURVA = 100


def reducir_rapido(familia, cfg):
    cfg = dict(cfg)
    if familia == "RandomForest":
        cfg["n_estimators"] = min(cfg["n_estimators"], 20)
    elif familia == "AdaBoost":
        cfg["n_estimators"] = min(cfg["n_estimators"], 10)
    elif familia in FAMILIAS_RED:
        cfg.update(epocas=2, patience=1)
    return cfg


def generar_candidatos(familia):
    n_total = N_ITER_REDES if familia in FAMILIAS_RED else N_ITER_ARBOLES
    rng = np.random.default_rng([SEED, TODAS_FAMILIAS.index(familia)])   # una semilla por familia
    candidatos = list(INICIALES[familia].items())[:n_total]
    i = 1
    while len(candidatos) < n_total:
        candidatos.append((f"{CORTO[familia]}_c{i:02d}", MUESTREADORES[familia](rng)))
        i += 1
    if RAPIDO:
        candidatos = [(cid, reducir_rapido(familia, cfg)) for cid, cfg in candidatos]
    if CONFIGS_FILTRO:
        candidatos = [(cid, cfg) for cid, cfg in candidatos if cid in CONFIGS_FILTRO] or candidatos
    if RF_MAX_ARBOLES and familia == "RandomForest":
        candidatos = [(cid, {**cfg, "n_estimators": min(cfg["n_estimators"], RF_MAX_ARBOLES)}) for cid, cfg in candidatos]
    return candidatos


if RAPIDO:
    XGB_COMUN.update(n_estimators=60, early_stopping_rounds=10)
    RF_PROFUNDIDADES, RF_ARBOLES_CURVA = [5, None], 10

CANDIDATOS = {f: generar_candidatos(f) for f in FAMILIAS}
for familia, lista in CANDIDATOS.items():
    print(f"{familia:12s}: {len(lista)} configuraciones -> {[cid for cid, _ in lista][:6]}{' ...' if len(lista) > 6 else ''}")

N_PERM_FILAS, N_PERM_REPETICIONES = (1000, 1) if RAPIDO else (4000, 3)
N_SHAP_FILAS = 300 if RAPIDO else 2000
N_CURVA_TRAIN = 3000 if RAPIDO else 10000   # submuestra de train para curvas (costo de cómputo)
N_PROCESOS_ADA = max(1, min(N_BLOQUES, (os.cpu_count() or 2) // 2))   # folds de AdaBoost en paralelo
'''),
    ("md", r"""
## 1. Datos

Las tres particiones se colocan sobre **una malla horaria única** (de la primera hora de train a la última de
test). Las horas que no están en los archivos (margen entre particiones o filas descartadas) quedan con
`valido = 0`. Así, el desplazamiento τ de la etiqueta y la ventana de 24 h de las redes se hacen por tiempo y
no por posición de fila. La etiqueta solo se toma dentro de la misma partición.
"""),
    ("code", r'''
def leer_particion(nombre):
    d = pd.read_parquet(DATA_DIR / f"{nombre}_@@slug@@.parquet")
    if "timestamp" in d.columns:
        d = d.set_index("timestamp")
    d.index = pd.DatetimeIndex(d.index, name="timestamp")
    d = d.sort_index()
    if "valido" not in d.columns:
        print(f"Aviso: {nombre} no tiene la columna 'valido'; se asume valido = 1 en todas las filas.")
        d["valido"] = 1
    return d


particiones = {p: leer_particion(p) for p in ("train", "val", "test")}
escaladores = pd.read_csv(DATA_DIR / "escaladores_@@slug@@.csv").set_index("columna")
TGT_MIN, TGT_MAX = escaladores.loc[TARGET, ["min", "max"]].astype(float)
RANGO_CM = TGT_MAX - TGT_MIN


def a_cm(y):
    return np.asarray(y, dtype=float) * RANGO_CM + TGT_MIN


FEAT_TAB = [c for c in particiones["train"].columns if c not in ("y_obs", "valido")]
_ruta_cat = DATA_DIR / "catalogo_features_@@slug@@.csv"
CATALOGO = pd.read_csv(_ruta_cat) if _ruta_cat.exists() else pd.DataFrame(columns=["feature"])
if VARIABLES == "base" and "conjunto" in CATALOGO:
    _extendidas = set(CATALOGO.loc[CATALOGO["conjunto"] == "lluvia_extendida", "feature"])
    FEAT_TAB = [c for c in FEAT_TAB if c not in _extendidas]
# Pluviómetros de entrada (columnas base): definen la lluvia areal de la regla de pesos en ambos conjuntos
COLS_LLUVIA = [c for c in FEAT_TAB if c.endswith("__Lluvia_Tot") and not c.startswith("AREAL_")]
# Las secuencias ya contienen la historia de 24 h: se quitan lags y deltas (redundantes);
# se mantienen los acumulados de lluvia porque llegan más allá de la ventana.
FEAT_SEQ = [c for c in FEAT_TAB if "_lag_" not in c and "_delta_" not in c]
assert TARGET in FEAT_TAB, "La persistencia necesita el nivel en t como columna de entrada"
for nombre, d in particiones.items():
    faltan = set(FEAT_TAB) - set(d.columns)
    assert not faltan, f"{nombre} no tiene las columnas {sorted(faltan)[:5]}"
assert particiones["train"].index.max() < particiones["val"].index.min() < particiones["val"].index.max() \
    < particiones["test"].index.min(), "Las particiones deben ser cronológicas y sin solape"

# Malla horaria global
_unidas = pd.concat([d[FEAT_TAB + ["y_obs", "valido"]].assign(particion=p) for p, d in particiones.items()])
MALLA = pd.date_range(_unidas.index.min(), _unidas.index.max(), freq="h", name="timestamp")
G = _unidas.reindex(MALLA)
G["valido"] = G["valido"].fillna(0).astype(np.int8)
G["particion"] = G["particion"].fillna("fuera")
N_MALLA = len(G)
TS = G.index
PART = G["particion"].to_numpy()
Y_OBS = G["y_obs"].to_numpy(np.float32)
PERS = G[TARGET].to_numpy(np.float32)
X_TAB = G[FEAT_TAB].to_numpy(np.float32)
# Base del objetivo: con Δ el modelo aprende y(t+τ) − nivel(t) y el nivel se reconstruye sumando nivel(t)
BASE_OBJ = PERS.copy() if OBJETIVO == "delta" else np.zeros(N_MALLA, dtype=np.float32)
# Lluvia areal en mm (media de los pluviómetros de entrada, desescalados) para la regla de pesos
LLUVIA_AREAL_MM = np.mean([G[c].to_numpy(float) * (escaladores.at[c, "max"] - escaladores.at[c, "min"])
                           + escaladores.at[c, "min"] for c in COLS_LLUVIA], axis=0) if COLS_LLUVIA else np.zeros(N_MALLA)

# Entrada válida en t y ventana de 24 h consecutivas válidas (común a árboles y redes)
ENTRADA_OK = (G["valido"].to_numpy() == 1) & ~np.isnan(X_TAB).any(axis=1) & ~np.isnan(PERS)
VENTANA_OK = pd.Series(ENTRADA_OK.astype(np.int32)).rolling(LOOKBACK_H).sum().eq(LOOKBACK_H).to_numpy()
del _unidas
gc.collect()

# Bloques del walk-forward: 5 tramos consecutivos de igual duración sobre el rango temporal de val
VAL_INI = particiones["val"].index.min()
VAL_FIN = particiones["val"].index.max() + pd.Timedelta(hours=1)
BORDES = [VAL_INI + (VAL_FIN - VAL_INI) * k / N_BLOQUES for k in range(N_BLOQUES + 1)]
BORDES = [b.floor("h") for b in BORDES[:-1]] + [VAL_FIN]
P95_TRAIN_CM = float(np.nanpercentile(a_cm(Y_OBS[(PART == "train") & ENTRADA_OK]), 95))

for nombre, d in particiones.items():
    print(f"{nombre:5s}: {len(d):,} h | {d.index.min()} -> {d.index.max()} | valido=1: {d['valido'].mean():.1%} "
          f"| y_obs observado: {d['y_obs'].notna().mean():.1%}")
print(f"Malla horaria: {N_MALLA:,} h | horas fuera de los archivos: {(PART == 'fuera').sum():,} "
      f"| filas con ventana de 24 h válida: {VENTANA_OK.sum():,}")
print(f"Variables tabulares (árboles): {len(FEAT_TAB)} | por paso de tiempo (RNN/LSTM): {len(FEAT_SEQ)}")
print(f"Escala del objetivo: {TGT_MIN:.1f} – {TGT_MAX:.1f} cm | P95 de y_obs en train: {P95_TRAIN_CM:.1f} cm")
P95_TRAIN_ESC = (P95_TRAIN_CM - TGT_MIN) / RANGO_CM
_llueve = (LLUVIA_AREAL_MM >= UMBRAL_LLUVIA_PESO_MM) & ENTRADA_OK & (PART == "train")
print(f"Pluviómetros de la lluvia areal (pesos): {COLS_LLUVIA} | horas de train con lluvia areal >= "
      f"{UMBRAL_LLUVIA_PESO_MM} mm: {_llueve.sum() / max(1, (ENTRADA_OK & (PART == 'train')).sum()):.1%}")
print("\nBloques de val:")
for k in range(N_BLOQUES):
    print(f"  bloque {k + 1}: {BORDES[k]} -> {BORDES[k + 1]}")
'''),
]
