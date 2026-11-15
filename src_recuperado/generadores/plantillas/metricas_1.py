# -*- coding: utf-8 -*-
"""Plantilla del notebook de métricas (src/metrics/<slug>_models_metric.ipynb): parte 1 de 3.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("md", r"""
# Métricas y gráficas de evaluación — Río @@nombre@@

Evaluación de los modelos entrenados en `src/models/modelos_@@slug@@.ipynb` con las métricas que
fija la tesis (`Docs/TesisCardenasPlazaSanchezRivadeneira.docx`, §1.3 y §2.2.8–2.2.9). Los horizontes
τ₁ y τ₂ y los umbrales de prealerta y alerta se leen de `Docs/resources/horizontes_@@slug@@.json`.

| Bloque | Contenido |
|---|---|
| Criterio principal | Índice de habilidad SS = 1 − RMSE_modelo / RMSE_persistencia y prueba de Diebold–Mariano (varianza HAC de Newey–West, corrección de Harvey–Leybourne–Newbold y de Holm) |
| Complementarias | NSE, KGE (r, α, β), RMSE, MAE, sesgo medio, PBIAS y R² |
| Crecidas | Por evento de prealerta: error en el pico, % de picos subestimados y desfase del pico; niveles altos (> P95 de train): MAE, RMSE y sesgo |
| Alertas | Estados normal / prealerta / alerta: matriz 3 × 3, POD, precisión, F1 y F1 macro, FAR, POFD, errores ordinales, eventos omitidos y anticipación efectiva (simulación secuencial) |
| Diagnóstico | Brecha de generalización relativa a la persistencia, SS por fold del walk-forward, importancia por permutación y SHAP |
| Figuras | G01–G16 |

**Reglas:**
- Todas las métricas se calculan en cm y solo sobre horas con nivel observado.
- Todos los modelos se comparan sobre **las mismas horas**: la intersección de sus predicciones en
  cada horizonte y partición (y en cada fold del walk-forward).
- La selección del modelo se hace con el walk-forward de validación, nunca con test. Test se usa una
  sola vez, para el contraste final frente a la persistencia.
- Los criterios de Moriasi et al. (2007) no se aplican (P173 de la tesis): el NSE es complementario.
"""),
    ("md", r"""
## 0. Configuración y carga
"""),
    ("code", r'''
import os
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, PredictionErrorDisplay, confusion_matrix, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

RAPIDO = os.environ.get("@@PREFIJO@@RAPIDO") == "1"


def _raiz_repo():
    actual = Path.cwd().resolve()
    for candidato in [actual, *actual.parents]:
        if (candidato / "Docs" / "resources").exists():
            return candidato
    return actual


ROOT = _raiz_repo()
RES_DIR = ROOT / "Docs" / "resources"
# Rutas configurables: @@PREFIJO@@DATA_DIR (particiones, escaladores, horizontes) y @@PREFIJO@@OUT_DIR (salidas de modelos)
DATA_DIR = Path(os.environ.get("@@PREFIJO@@DATA_DIR", RES_DIR))
MOD_DIR = Path(os.environ.get("@@PREFIJO@@OUT_DIR", RES_DIR / ("modelos_@@slug@@_rapido" if RAPIDO else "modelos_@@slug@@")))
FIG_DIR = MOD_DIR / "figuras"
TAB_DIR = MOD_DIR / "tablas"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

PARTICIONES = ["train", "val", "test"]
FAMILIAS_ORDEN = ["RandomForest", "XGBoost", "AdaBoost", "RNN", "LSTM"]
SEED = 42
SEPARACION_EVENTOS_H = 6     # cruces separados por menos de 6 h forman un mismo evento (P53)
VENTANA_PICO_H = 12          # ventana alrededor del evento para buscar el pico pronosticado
TOL_SS = 0.01                # diferencia de SS por debajo de la cual se desempata (P56)
ALFA_DM = 0.05

# Horizontes y umbrales oficiales (fijados antes de entrenar)
with open(DATA_DIR / "horizontes_@@slug@@.json", encoding="utf-8") as f:
    HZ = json.load(f)
TAU1, TAU2 = int(HZ["tau1"]), int(HZ["tau2"])
# Los estados se determinan en CAUDAL (Tesis §1.3 P103, §2.2.9 P247): el nivel pronosticado y el
# observado se convierten con la curva de descarga de la estación, Q = A(H − H₀)^p (Tabla 2), y se
# comparan con los caudales de alerta de ETAPA EP (prealerta desde @@q_pre_g@@ m³/s, alerta desde @@q_ale_g@@ m³/s).
CURVA_A, CURVA_H0, CURVA_P = @@curva_a@@, @@curva_h0@@, @@curva_p@@      # @@estacion_tesis@@ (Tabla 2)
Q_PRE = float(HZ.get("umbral_prealerta_m3s", @@q_pre@@))
Q_ALE = float(HZ.get("umbral_alerta_m3s", @@q_ale@@))
UMBRAL_ESTADO = {1: Q_PRE, 2: Q_ALE}                         # en m³/s
# Niveles equivalentes: solo como referencia en gráficas y textos
U_PRE, U_ALE = float(HZ["umbral_prealerta_cm"]), float(HZ["umbral_alerta_cm"])


def a_caudal(nivel):
    h = np.asarray(nivel, dtype=float)
    return CURVA_A * np.clip(h - CURVA_H0, 0, None) ** CURVA_P
ESTADOS = ["normal", "prealerta", "alerta"]


def _leer_opcional(nombre, lector):
    ruta = MOD_DIR / nombre
    if ruta.exists():
        return lector(ruta)
    print(f"[aviso] No existe {nombre}; se omite lo que depende de este archivo.")
    return None


pred = pd.read_parquet(MOD_DIR / "predicciones_@@slug@@.parquet")
pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
if "fold" not in pred.columns:
    pred["fold"] = pd.NA
pred["fold"] = pred["fold"].astype("Int64")
HIST = _leer_opcional("historiales.json", lambda r: json.loads(r.read_text(encoding="utf-8"))) or {}
busqueda = _leer_opcional("busqueda_resultados.csv", pd.read_csv)
importancias = _leer_opcional("importancia_permutacion.csv", pd.read_csv)
shap_imp = _leer_opcional("shap_importancia.csv", pd.read_csv)
resumen_entrenamiento = _leer_opcional("resumen_entrenamiento.csv", pd.read_csv)
CONFIG = _leer_opcional("configuracion.json", lambda r: json.loads(r.read_text(encoding="utf-8"))) or {}
rf_prof = _leer_opcional("rf_curva_profundidad.csv", pd.read_csv) if (MOD_DIR / "rf_curva_profundidad.csv").exists() else None

HORIZONTES = sorted(int(h) for h in pred["horizonte"].unique())
if set(HORIZONTES) != {TAU1, TAU2}:
    print(f"[aviso] Horizontes en predicciones {HORIZONTES} distintos de τ₁={TAU1}, τ₂={TAU2} del JSON.")


def etiqueta_h(h):
    if h == "combinado":
        return "combinado (τ₁, τ₂)"
    return {TAU1: f"τ₁ = {h} h", TAU2: f"τ₂ = {h} h"}.get(h, f"{h} h")


print(f"Datos: {DATA_DIR}\nSalidas de modelos: {MOD_DIR}")
print(f"Tr = {HZ.get('Tr_h')} h | τ₁ = {TAU1} h | τ₂ = {TAU2} h | prealerta = {Q_PRE:.0f} m³/s (≈{U_PRE:.1f} cm) | alerta = {Q_ALE:.0f} m³/s (≈{U_ALE:.1f} cm)")
'''),
    ("code", r'''
# Nivel observado (cm) por partición y P95 de train (niveles altos)
escaladores = pd.read_csv(DATA_DIR / "escaladores_@@slug@@.csv").set_index("columna")
tmin, tmax = escaladores.loc["y_obs", ["min", "max"]].astype(float)
SERIE_OBS = {}
for p in PARTICIONES:
    s = pd.read_parquet(DATA_DIR / f"{p}_@@slug@@.parquet", columns=["y_obs"])["y_obs"] * (tmax - tmin) + tmin
    s.index = pd.to_datetime(s.index, utc=True)
    SERIE_OBS[p] = s.sort_index()
OBS_TOTAL = pd.concat(SERIE_OBS.values()).sort_index()
OBS_TOTAL = OBS_TOTAL[~OBS_TOTAL.index.duplicated()]
# Mismo umbral que la selección del notebook de entrenamiento (P95 de y_obs en las filas válidas de train);
# si no hay configuracion.json se usa todo el nivel observado de train.
P95 = float(CONFIG["p95_train_cm"]) if CONFIG.get("p95_train_cm") else SERIE_OBS["train"].dropna().quantile(0.95)
DEF_P95 = ("P95 de y_obs en las filas válidas de train (configuracion.json)" if CONFIG.get("p95_train_cm")
           else "P95 de todo y_obs observado en train")
print(f"P95 del nivel observado en train (niveles altos): {P95:.1f} cm — {DEF_P95}")

# Convención temporal: ¿timestamp es el instante objetivo (t + τ) o el de emisión (t)?
pp = pred[(pred["modelo"] == "Persistencia") & pred["fold"].isna()]
pp = pp[pp["horizonte"] == HORIZONTES[0]].head(20000)
dif_obj = (pp["y_real_cm"].to_numpy() - OBS_TOTAL.reindex(pp["timestamp"]).to_numpy())
dif_emi = (pp["y_real_cm"].to_numpy() - OBS_TOTAL.reindex(pp["timestamp"] + pd.Timedelta(hours=HORIZONTES[0])).to_numpy())
TS_ES_OBJETIVO = not (np.nanmedian(np.abs(dif_emi)) < np.nanmedian(np.abs(dif_obj)))
desfase = pd.to_timedelta(pred["horizonte"], unit="h")
pred["t_obj"] = pred["timestamp"] if TS_ES_OBJETIVO else pred["timestamp"] + desfase
pred["t_emi"] = pred["t_obj"] - desfase
print("Convención temporal detectada:", "timestamp = instante objetivo" if TS_ES_OBJETIVO else "timestamp = instante de emisión")


def interseccion(df, grupo):
    # Conserva las horas presentes en TODOS los modelos del horizonte dentro de cada grupo
    n_mod = df.groupby("horizonte")["modelo"].nunique()
    conteo = df.groupby(grupo + ["timestamp"])["modelo"].transform("nunique")
    return df[conteo == df["horizonte"].map(n_mod)]


final = pred[pred["fold"].isna() & pred["particion"].isin(PARTICIONES)]
wf = pred[pred["particion"] == "val_fold"]
antes = len(final) + len(wf)
pred_final = interseccion(final, ["horizonte", "particion"]).sort_values(["modelo", "horizonte", "t_obj"])
pred_wf = interseccion(wf, ["horizonte", "fold"]).sort_values(["modelo", "horizonte", "t_obj"]) if len(wf) else wf
print(f"Predicciones finales: {len(pred_final):,} filas | walk-forward: {len(pred_wf):,} filas "
      f"(se descartaron {antes - len(pred_final) - len(pred_wf):,} sin contraparte en todos los modelos)")

FAMILIA_DE = pred.drop_duplicates("modelo").set_index("modelo")["familia"].to_dict()
FAMILIAS = [f for f in FAMILIAS_ORDEN if f in FAMILIA_DE.values()] + \
           sorted(set(FAMILIA_DE.values()) - set(FAMILIAS_ORDEN) - {"Persistencia"})
orden = list(dict.fromkeys(resumen_entrenamiento["modelo"])) if resumen_entrenamiento is not None else []
MODELOS = [m for m in orden if m in FAMILIA_DE] + sorted(set(pred_final["modelo"]) - set(orden))
MODELOS = ["Persistencia"] + [m for m in MODELOS if m != "Persistencia"]
DATOS = {k: g.set_index("t_obj").sort_index() for k, g in pred_final.groupby(["modelo", "particion", "horizonte"])}
print(f"Modelos ({len(MODELOS)}): {MODELOS}")
print(f"Familias: {FAMILIAS} | Horizontes: {HORIZONTES}")


def guardar_figura(fig, nombre):
    fig.savefig(FIG_DIR / nombre, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
'''),
    ("md", r"""
## 1. Funciones de métricas

Notación de la tesis (P166): oᵢ observado, sᵢ pronosticado, n número de horas.

| Métrica | Fórmula | Nota |
|---|---|---|
| RMSE (7) | √(Σ(sᵢ − oᵢ)²/n) | cm |
| MAE (8) | Σ\|sᵢ − oᵢ\|/n | cm |
| Sesgo medio (9) | Σ(sᵢ − oᵢ)/n | positivo = sobreestimación |
| NSE (10) | 1 − Σ(sᵢ − oᵢ)²/Σ(oᵢ − ō)² | complementaria |
| R² (11) | r² de Pearson | complementaria |
| PBIAS (12) | 100·Σ(oᵢ − sᵢ)/Σoᵢ | convención de Moriasi: positivo = subestimación |
| KGE (13) | 1 − √((r − 1)² + (α − 1)² + (β − 1)²), α = σₛ/σₒ, β = s̄/ō | |
| SS (14) | 1 − RMSE_modelo/RMSE_persistencia | mismas horas, horizonte y conjunto |

**Crecidas (P238–P243).** Evento = episodio continuo en que el nivel observado transformado a caudal
(Q = A(H − H₀)^p, Tabla 2) alcanza el umbral de prealerta de ETAPA EP (@@q_pre_g@@ m³/s); los
tramos separados por menos de 6 h por debajo del umbral se agrupan (P53). En cada evento se busca el
pico pronosticado en la ventana [inicio − 12 h, fin + 12 h]: error en el pico = pico pronosticado −
pico observado (cm y %), subestimado si es negativo, y desfase = hora del pico pronosticado − hora del
pico observado (positivo = el pronóstico llega tarde). Niveles altos: horas con nivel observado > P95
de train.

**Alertas (P196–P211).** Estado = normal si nivel < prealerta, prealerta si prealerta ≤ nivel < alerta,
y alerta si nivel ≥ alerta. Para cada estado (uno contra el resto): POD = VP/(VP + FN),
precisión = VP/(VP + FP), F1 = 2·precisión·POD/(precisión + POD); F1 macro = media de los F1 de las
clases **observadas** en el conjunto evaluado (`f1_score(..., average="macro", zero_division=0)`): una clase
observada que el modelo nunca pronostica cuenta con F1 = 0 (no se omite del promedio). En la tabla 2 × 2: FAR = FP/(VP + FP) (razón de falsas alarmas) y POFD = FP/(FP + VN)
(probabilidad de falsa detección).
"""),
    ("code", r'''
def _div(a, b):
    return a / b if b > 0 else np.nan


def metricas_regresion(o, s):
    e = s - o
    rmse = np.sqrt(np.mean(e ** 2))
    r = np.corrcoef(o, s)[0, 1] if len(o) > 1 else np.nan
    alfa, beta = _div(s.std(), o.std()), _div(s.mean(), o.mean())
    return {"RMSE": rmse, "MAE": np.mean(np.abs(e)), "sesgo": np.mean(e),
            "NSE": 1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2), "R2": r ** 2,
            "PBIAS": 100 * np.sum(o - s) / np.sum(o),
            "KGE": 1 - np.sqrt((r - 1) ** 2 + (alfa - 1) ** 2 + (beta - 1) ** 2),
            "KGE_r": r, "KGE_alfa": alfa, "KGE_beta": beta}


def metricas_niveles_altos(o, s, umbral=None):
    umbral = P95 if umbral is None else umbral
    alto = o > umbral
    if not alto.any():
        return {"n_alto": 0, "MAE_alto": np.nan, "RMSE_alto": np.nan, "sesgo_alto": np.nan}
    e = s[alto] - o[alto]
    return {"n_alto": int(alto.sum()), "MAE_alto": np.mean(np.abs(e)), "RMSE_alto": np.sqrt(np.mean(e ** 2)),
            "sesgo_alto": np.mean(e)}


def a_horas(t):
    t = pd.DatetimeIndex(t)
    return np.asarray((t - t[0]) / pd.Timedelta(hours=1), dtype=float)


def episodios(horas, valores, umbral, sep_h=SEPARACION_EVENTOS_H):
    # Episodios con valores >= umbral; se unen si las horas por debajo entre ambos son < sep_h
    sobre = np.flatnonzero(np.nan_to_num(valores, nan=-np.inf) >= umbral)
    if len(sobre) == 0:
        return []
    cortes = np.flatnonzero(np.diff(horas[sobre]) - 1 >= sep_h) + 1
    return np.split(sobre, cortes)


def eventos_pico(t, o, s, umbral=None, ventana_h=VENTANA_PICO_H):
    # Evento = episodio con el nivel observado transformado a caudal >= umbral de prealerta (m³/s)
    umbral = Q_PRE if umbral is None else umbral
    t = pd.DatetimeIndex(t)
    horas = a_horas(t)
    filas = []
    for ev in episodios(horas, a_caudal(o), umbral):
        i_obs = ev[np.argmax(o[ev])]
        idx = np.flatnonzero((horas >= horas[ev[0]] - ventana_h) & (horas <= horas[ev[-1]] + ventana_h) & ~np.isnan(s))
        i_pred = idx[np.argmax(s[idx])]
        filas.append({"inicio": t[ev[0]], "fin": t[ev[-1]], "duracion_h": horas[ev[-1]] - horas[ev[0]] + 1,
                      "t_pico_obs": t[i_obs], "pico_obs_cm": o[i_obs], "t_pico_pred": t[i_pred],
                      "pico_pred_cm": s[i_pred], "pico_obs_m3s": float(a_caudal(o[i_obs])),
                      "pico_pred_m3s": float(a_caudal(s[i_pred])), "error_pico_cm": s[i_pred] - o[i_obs],
                      "error_pico_rel_pct": 100 * (s[i_pred] - o[i_obs]) / o[i_obs],
                      "subestimado": bool(s[i_pred] < o[i_obs]), "desfase_pico_h": horas[i_pred] - horas[i_obs]})
    return pd.DataFrame(filas)


def resumen_picos(ev):
    if ev is None or len(ev) == 0:
        return {"n_eventos": 0, "error_pico_medio_cm": np.nan, "error_pico_abs_medio_cm": np.nan,
                "error_pico_rel_medio_pct": np.nan, "pct_picos_subestimados": np.nan,
                "desfase_medio_h": np.nan, "desfase_abs_medio_h": np.nan}
    return {"n_eventos": len(ev), "error_pico_medio_cm": ev["error_pico_cm"].mean(),
            "error_pico_abs_medio_cm": ev["error_pico_cm"].abs().mean(),
            "error_pico_rel_medio_pct": ev["error_pico_rel_pct"].mean(),
            "pct_picos_subestimados": 100 * ev["subestimado"].mean(),
            "desfase_medio_h": ev["desfase_pico_h"].mean(), "desfase_abs_medio_h": ev["desfase_pico_h"].abs().mean()}
'''),
    ("code", r'''
def a_estado(nivel):
    # Nivel (cm) -> caudal (m³/s) -> estado: 0 normal (Q < @@q_pre_g@@), 1 prealerta (@@q_pre_g@@ <= Q < @@q_ale_g@@), 2 alerta (Q >= @@q_ale_g@@)
    nivel = np.asarray(nivel, dtype=float)
    q = a_caudal(nivel)
    est = np.select([q >= Q_ALE, q >= Q_PRE], [2.0, 1.0], 0.0)
    return np.where(np.isnan(nivel), np.nan, est)


def tabla_2x2(obs, prd, k):
    ob, pr = obs >= k, prd >= k
    vp, fp = int((ob & pr).sum()), int((~ob & pr).sum())
    fn, vn = int((ob & ~pr).sum()), int((~ob & ~pr).sum())
    pod, prec = _div(vp, vp + fn), _div(vp, vp + fp)
    return {"VP": vp, "FP": fp, "FN": fn, "VN": vn, "POD": pod, "precision": prec,
            "F1": _div(2 * prec * pod, prec + pod) if not (np.isnan(pod) or np.isnan(prec)) else np.nan,
            "FAR": _div(fp, vp + fp), "POFD": _div(fp, fp + vn)}


def metricas_estados(obs, prd):
    # obs, prd: estados 0/1/2 sin NaN. Devuelve matriz 3x3 y métricas por clase, 2x2 y ordinales.
    cm = confusion_matrix(obs, prd, labels=[0, 1, 2])
    total = cm.sum()
    res = {"n": int(total)}
    f1s = []
    for c, nombre in enumerate(ESTADOS):
        vp = cm[c, c]
        fp, fn = cm[:, c].sum() - vp, cm[c, :].sum() - vp
        pod, prec = _div(vp, vp + fn), _div(vp, vp + fp)
        f1 = _div(2 * prec * pod, prec + pod) if not (np.isnan(pod) or np.isnan(prec)) else np.nan
        if cm[c, :].sum() > 0 and np.isnan(f1):
            f1 = 0.0          # clase observada sin aciertos (nunca pronosticada o sin VP): F1 = 0
        res.update({f"n_obs_{nombre}": int(cm[c, :].sum()), f"POD_{nombre}": pod,
                    f"precision_{nombre}": prec, f"F1_{nombre}": f1})
        f1s.append(f1)
    observadas = [c for c in range(len(ESTADOS)) if cm[c, :].sum() > 0]
    res["F1_macro"] = (float(f1_score(np.asarray(obs, int), np.asarray(prd, int), labels=observadas, average="macro", zero_division=0))
                       if observadas else np.nan)
    for k, etiqueta in [(1, "aviso"), (2, "alerta")]:   # aviso = prealerta o alerta frente a normal
        res.update({f"{m}_{etiqueta}": v for m, v in tabla_2x2(obs, prd, k).items()})
    d = prd - obs
    for nombre, cond in [("exacto", d == 0), ("sub_1", d == -1), ("sub_2", d == -2), ("sobre_1", d == 1), ("sobre_2", d == 2)]:
        res[f"pct_{nombre}"] = 100 * cond.mean() if total else np.nan
    return res, cm


def estados_horizonte(modelo, part, h):
    g = DATOS[(modelo, part, h)]
    return pd.DataFrame({"est_obs": a_estado(g["y_real_cm"]), "est_pred": a_estado(g["y_pred_cm"])},
                        index=pd.DatetimeIndex(g["t_emi"], name="t_emi")).sort_index()


def estados_combinados(modelo, part):
    # Estado emitido en t = el más severo de los pronosticados a los horizontes del río (P53).
    # Referencia observada = el más severo de los observados en los mismos instantes objetivo.
    partes = [estados_horizonte(modelo, part, h).add_suffix(f"_{h}") for h in HORIZONTES]
    j = pd.concat(partes, axis=1).sort_index()
    cols_p = [f"est_pred_{h}" for h in HORIZONTES]
    cols_o = [f"est_obs_{h}" for h in HORIZONTES]
    return pd.DataFrame({"est_pred": j[cols_p].max(axis=1, skipna=True), "est_obs": j[cols_o].max(axis=1, skipna=True),
                         "completo": j[cols_p + cols_o].notna().all(axis=1)}, index=j.index)


def simular_eventos(estado_emitido, obs, k, anticipacion_max):
    # Simulación secuencial: un evento observado (cruce del umbral del estado k, agrupando cruces a < 6 h)
    # se detecta si en t = cruce − 1 h ya se emitía un estado >= k y se mantuvo sin interrupción desde la
    # emisión; la anticipación efectiva es la duración de esa racha, acotada por el horizonte.
    o = obs.asfreq("h")
    e = estado_emitido.reindex(o.index)
    horas = a_horas(o.index)
    filas = []
    for ev in episodios(horas, a_caudal(o.to_numpy()), UMBRAL_ESTADO[k]):
        if ev[0] == 0:
            continue   # empieza con la partición: no se observa el cruce
        c = o.index[ev[0]]
        previos = e.reindex([c - pd.Timedelta(hours=j) for j in range(1, anticipacion_max + 1)]).to_numpy()
        racha = 0
        for v in previos:
            if not np.isnan(v) and v >= k:
                racha += 1
            else:
                break
        filas.append({"cruce": c, "pico_obs_cm": o.iloc[ev].max(), "evaluable": bool(np.any(~np.isnan(previos))),
                      "detectado": racha >= 1, "anticipacion_h": racha if racha >= 1 else np.nan})
    eventos = pd.DataFrame(filas, columns=["cruce", "pico_obs_cm", "evaluable", "detectado", "anticipacion_h"])
    # Episodios de alerta emitida sin evento observado en [inicio, fin + horizonte]: falsas alarmas por episodio
    em = estado_emitido.asfreq("h")
    falsas = 0
    for ep in episodios(a_horas(em.index), em.to_numpy(), k):
        ini, fin = em.index[ep[0]], em.index[ep[-1]] + pd.Timedelta(hours=anticipacion_max)
        if not (a_caudal(obs.loc[ini:fin].to_numpy()) >= UMBRAL_ESTADO[k]).any():
            falsas += 1
    return eventos, falsas
'''),
    ("code", r'''
def diebold_mariano(e_mod, e_ref, t, h, rezago=None):
    # Prueba DM con pérdida cuadrática: d_t = e_mod² − e_ref². H1 unilateral: el modelo tiene menor pérdida.
    # Varianza de largo plazo HAC de Newey–West (núcleo de Bartlett) sobre la grilla horaria (los huecos no
    # se tratan como contiguos) y corrección de Harvey, Leybourne y Newbold (1997) con t de Student (n − 1).
    d = pd.Series(np.asarray(e_mod) ** 2 - np.asarray(e_ref) ** 2, index=pd.DatetimeIndex(t)).sort_index()
    d = d[~d.index.duplicated()].asfreq("h")
    n = int(d.notna().sum())
    dbar = d.mean()
    L = rezago if rezago is not None else max(h - 1, math.ceil(n ** (1 / 3)))
    dc = (d - dbar).to_numpy()
    gamma0 = np.nansum(dc * dc) / n
    var = gamma0
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * np.nansum(dc[k:] * dc[:-k]) / n
    if var <= 0:
        var = gamma0
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"n": n, "rezago_NW": L, "d_medio_cm2": dbar, "DM": dm, "DM_HLN": dm * hln,
            "p_unilateral": stats.t.cdf(dm * hln, df=n - 1)}


def holm(pvalores):
    p = np.asarray(pvalores, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acumulado = 0.0
    for i, j in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - i) * p[j]))
        ajust[j] = acumulado
    return ajust
'''),
]
