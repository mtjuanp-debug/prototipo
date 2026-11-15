# -*- coding: utf-8 -*-
"""Métricas de test de los modelos exploratorios del Yanuncay (anexo de la tesis).

Usa las MISMAS definiciones que el notebook oficial ``src/metrics/yanuncay_models_metric.ipynb`` (funciones
copiadas sin cambios de criterio): SS, NSE, KGE, RMSE, MAE, sesgo, PBIAS, R²; Diebold–Mariano (HAC Newey–West +
HLN) con Holm frente a la persistencia; estados de alerta en CAUDAL (Q = A(H − H₀)^p) con F1 macro corregido
(una clase observada que no se pronostica cuenta con F1 = 0); eventos detectados y anticipación efectiva
(simulación secuencial). El RF oficial se recalcula sobre las mismas horas como control de coherencia y la tabla
comparativa toma además sus valores publicados de ``modelos_yanuncay/tablas/metricas_test_por_horizonte.csv``.

Los exploratorios NO cambian la selección oficial: aquí solo se leen sus predicciones de test (ya emitidas una
única vez por ``ejecutar_exploratorios.py --fase final``).

Uso:
  python src/models/exploratorios/metricas_exploratorios.py --unir     # une partes/arboles y partes/redes
  python src/models/exploratorios/metricas_exploratorios.py            # métricas, tablas y figuras
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "Docs" / "resources"
EXPL = RES / "modelos_yanuncay" / "exploratorios"
OFICIAL = RES / "modelos_yanuncay"
FAMILIAS = ["LGBM", "CAT", "GRU", "TCN", "NHITS"]
NOMBRE = {"LGBM": "LightGBM", "CAT": "CatBoost", "GRU": "GRU", "TCN": "TCN", "NHITS": "N-HiTS",
          "RF_oficial": "RF oficial", "Persistencia": "Persistencia"}
SEPARACION_EVENTOS_H, VENTANA_PICO_H, ALFA_DM, EMPATE_SS = 6, 12, 0.05, 0.01
CURVA_A, CURVA_H0, CURVA_P = 1.45966e-6, 20.0, 3.3435      # LIM_Yanuncay-AJ-Tarqui (Tabla 2)
ESTADOS = ["normal", "prealerta", "alerta"]

HZ = json.load(open(RES / "horizontes_yanuncay.json", encoding="utf-8"))
Q_PRE, Q_ALE = float(HZ.get("umbral_prealerta_m3s", 33.0)), float(HZ.get("umbral_alerta_m3s", 50.0))
UMBRAL_ESTADO = {1: Q_PRE, 2: Q_ALE}


# ------------------------------------------------------------------------------------------- unión de partes
def unir_partes(destino: Path = EXPL) -> None:
    """Une las salidas de partes/arboles (CPU) y partes/redes (GPU) en la carpeta exploratorios/."""
    partes = [destino / "partes" / p for p in ("arboles", "redes")]
    for p in partes:
        if not (p / "predicciones_yanuncay.parquet").exists():
            raise FileNotFoundError(f"Falta la fase final de {p}")
    preds = [pd.read_parquet(p / "predicciones_yanuncay.parquet") for p in partes]
    pers = [d[d["modelo"] == "Persistencia"].sort_values(["particion", "horizonte", "timestamp"]).reset_index(drop=True)
            for d in preds]
    pd.testing.assert_frame_equal(pers[0], pers[1])          # misma persistencia = mismas horas y escala
    pred = pd.concat([preds[0], preds[1][preds[1]["modelo"] != "Persistencia"]], ignore_index=True)
    pred.to_parquet(destino / "predicciones_yanuncay.parquet", index=False)
    pd.concat([pd.read_csv(p / "busqueda_resultados.csv") for p in partes], ignore_index=True).to_csv(
        destino / "busqueda_resultados.csv", index=False)
    res = [pd.read_csv(p / "resumen_entrenamiento.csv") for p in partes]
    res = pd.concat([res[0], res[1][res[1]["familia"] != "Persistencia"]], ignore_index=True)
    sel = pd.concat([pd.read_csv(p / "seleccion_modelo.csv") for p in partes], ignore_index=True)
    sel = sel.drop(columns="seleccionado").sort_values("SS_global", ascending=False).reset_index(drop=True)
    emp = sel[sel["SS_global"] >= sel["SS_global"].iloc[0] - EMPATE_SS]   # misma regla que el runner
    ganador = emp.loc[emp["RMSE_alto_global"].idxmin(), "familia"] if len(emp) > 1 else sel["familia"].iloc[0]
    sel["seleccionado"] = sel["familia"] == ganador
    sel.to_csv(destino / "seleccion_modelo.csv", index=False)
    res["SS_global"] = res["familia"].map(sel.set_index("familia")["SS_global"])
    res["seleccionado"] = res["familia"].map(sel.set_index("familia")["seleccionado"]).fillna(False)
    res.to_csv(destino / "resumen_entrenamiento.csv", index=False)
    hist, cfgs = {}, []
    for p in partes:
        hist.update(json.loads((p / "historiales.json").read_text(encoding="utf-8")))
        cfgs.append(json.loads((p / "configuracion.json").read_text(encoding="utf-8")))
    (destino / "historiales.json").write_text(json.dumps(hist), encoding="utf-8")
    cfg = dict(cfgs[0])
    for clave in ("candidatos", "seleccion_por_familia"):
        cfg[clave] = {**cfgs[0][clave], **cfgs[1][clave]}
    cfg["tiempo_total_min_partes"] = {p.name: c["tiempo_total_min"] for p, c in zip(partes, cfgs)}
    cfg["dispositivo_partes"] = {"arboles": "cpu (LightGBM/CatBoost)", "redes": cfgs[1]["dispositivo_redes"]}
    cfg["unido_desde"] = [str(p.relative_to(REPO)) for p in partes]
    cfg.pop("tiempo_total_min", None)
    (destino / "configuracion.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
    (destino / "modelos").mkdir(exist_ok=True)
    for p in partes:
        for f in (p / "modelos").iterdir():
            shutil.copy2(f, destino / "modelos" / f.name)
    print(f"Unido: {len(pred):,} filas, modelos {sorted(pred['modelo'].unique())}; seleccionado {ganador}")


# ------------------------------------------------------------------------------------------- funciones oficiales
def a_caudal(nivel):
    h = np.asarray(nivel, dtype=float)
    return CURVA_A * np.clip(h - CURVA_H0, 0, None) ** CURVA_P


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


def metricas_niveles_altos(o, s, umbral):
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
    sobre = np.flatnonzero(np.nan_to_num(valores, nan=-np.inf) >= umbral)
    if len(sobre) == 0:
        return []
    cortes = np.flatnonzero(np.diff(horas[sobre]) - 1 >= sep_h) + 1
    return np.split(sobre, cortes)


def eventos_pico(t, o, s, umbral=Q_PRE, ventana_h=VENTANA_PICO_H):
    t = pd.DatetimeIndex(t)
    horas = a_horas(t)
    filas = []
    for ev in episodios(horas, a_caudal(o), umbral):
        i_obs = ev[np.argmax(o[ev])]
        idx = np.flatnonzero((horas >= horas[ev[0]] - ventana_h) & (horas <= horas[ev[-1]] + ventana_h) & ~np.isnan(s))
        i_pred = idx[np.argmax(s[idx])]
        filas.append({"t_pico_obs": t[i_obs], "pico_obs_cm": o[i_obs], "pico_pred_cm": s[i_pred],
                      "error_pico_cm": s[i_pred] - o[i_obs], "subestimado": bool(s[i_pred] < o[i_obs]),
                      "desfase_pico_h": horas[i_pred] - horas[i_obs]})
    return pd.DataFrame(filas)


def resumen_picos(ev):
    if ev is None or len(ev) == 0:
        return {"n_eventos": 0, "error_pico_medio_cm": np.nan, "pct_picos_subestimados": np.nan,
                "desfase_medio_h": np.nan}
    return {"n_eventos": len(ev), "error_pico_medio_cm": ev["error_pico_cm"].mean(),
            "pct_picos_subestimados": 100 * ev["subestimado"].mean(), "desfase_medio_h": ev["desfase_pico_h"].mean()}


def a_estado(nivel):
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
    cm = confusion_matrix(obs, prd, labels=[0, 1, 2])
    res = {"n": int(cm.sum())}
    for c, nombre in enumerate(ESTADOS):
        vp = cm[c, c]
        fp, fn = cm[:, c].sum() - vp, cm[c, :].sum() - vp
        pod, prec = _div(vp, vp + fn), _div(vp, vp + fp)
        f1 = _div(2 * prec * pod, prec + pod) if not (np.isnan(pod) or np.isnan(prec)) else np.nan
        if cm[c, :].sum() > 0 and np.isnan(f1):
            f1 = 0.0          # clase observada sin aciertos (nunca pronosticada o sin VP): F1 = 0
        res.update({f"n_obs_{nombre}": int(cm[c, :].sum()), f"POD_{nombre}": pod, f"precision_{nombre}": prec,
                    f"F1_{nombre}": f1})
    observadas = [c for c in range(len(ESTADOS)) if cm[c, :].sum() > 0]
    res["F1_macro"] = (float(f1_score(np.asarray(obs, int), np.asarray(prd, int), labels=observadas,
                                      average="macro", zero_division=0)) if observadas else np.nan)
    for k, etiqueta in [(1, "aviso"), (2, "alerta")]:
        res.update({f"{m}_{etiqueta}": v for m, v in tabla_2x2(obs, prd, k).items()})
    return res


def simular_eventos(estado_emitido, obs, k, anticipacion_max):
    o = obs.asfreq("h")
    e = estado_emitido.reindex(o.index)
    horas = a_horas(o.index)
    filas = []
    for ev in episodios(horas, a_caudal(o.to_numpy()), UMBRAL_ESTADO[k]):
        if ev[0] == 0:
            continue
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
    em = estado_emitido.asfreq("h")
    falsas = 0
    for ep in episodios(a_horas(em.index), em.to_numpy(), k):
        ini, fin = em.index[ep[0]], em.index[ep[-1]] + pd.Timedelta(hours=anticipacion_max)
        if not (a_caudal(obs.loc[ini:fin].to_numpy()) >= UMBRAL_ESTADO[k]).any():
            falsas += 1
    return eventos, falsas


def diebold_mariano(e_mod, e_ref, t, h, rezago=None):
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


# ------------------------------------------------------------------------------------------- carga
def cargar(expl: Path):
    pe = pd.read_parquet(expl / "predicciones_yanuncay.parquet")
    po = pd.read_parquet(OFICIAL / "predicciones_yanuncay.parquet")
    po = po[po["modelo"] == "RF"].assign(modelo="RF_oficial", familia="RF_oficial")
    pred = pd.concat([pe, po], ignore_index=True)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred["fold"] = pred["fold"].astype("Int64")
    esc = pd.read_csv(RES / "escaladores_yanuncay.csv").set_index("columna")
    tmin, tmax = esc.loc["y_obs", ["min", "max"]].astype(float)
    obs = {}
    for p in ("train", "val", "test"):
        s = pd.read_parquet(RES / f"{p}_yanuncay.parquet", columns=["y_obs"])["y_obs"] * (tmax - tmin) + tmin
        s.index = pd.to_datetime(s.index, utc=True)
        obs[p] = s.sort_index()
    total = pd.concat(obs.values()).sort_index()
    total = total[~total.index.duplicated()]
    # Convención temporal (igual que el notebook oficial): timestamp = objetivo o emisión
    hs = sorted(pred["horizonte"].unique())
    pp = pred[(pred["modelo"] == "Persistencia") & pred["fold"].isna() & (pred["horizonte"] == hs[0])].head(20000)
    d_obj = pp["y_real_cm"].to_numpy() - total.reindex(pp["timestamp"]).to_numpy()
    d_emi = pp["y_real_cm"].to_numpy() - total.reindex(pp["timestamp"] + pd.Timedelta(hours=int(hs[0]))).to_numpy()
    es_obj = not (np.nanmedian(np.abs(d_emi)) < np.nanmedian(np.abs(d_obj)))
    desfase = pd.to_timedelta(pred["horizonte"], unit="h")
    pred["t_obj"] = pred["timestamp"] if es_obj else pred["timestamp"] + desfase
    pred["t_emi"] = pred["t_obj"] - desfase
    print("Convención temporal:", "timestamp = instante objetivo" if es_obj else "timestamp = instante de emisión")
    return pred, obs


def interseccion(df, grupo):
    n_mod = df.groupby("horizonte")["modelo"].nunique()
    conteo = df.groupby(grupo + ["timestamp"])["modelo"].transform("nunique")
    return df[conteo == df["horizonte"].map(n_mod)]


# ------------------------------------------------------------------------------------------- cálculo
def calcular(expl: Path) -> None:
    tab_dir, fig_dir = expl / "tablas", expl / "figuras"
    tab_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    cfg = json.loads((expl / "configuracion.json").read_text(encoding="utf-8"))
    p95 = float(cfg["p95_train_cm"])
    pred, obs = cargar(expl)
    test = interseccion(pred[(pred["particion"] == "test") & pred["fold"].isna()], ["horizonte"])
    wf = interseccion(pred[pred["particion"] == "val_fold"], ["horizonte", "fold"])
    hs = sorted(int(h) for h in test["horizonte"].unique())
    modelos = ["Persistencia", "RF_oficial"] + [f for f in FAMILIAS if f in set(test["modelo"])]
    datos = {k: g.set_index("t_obj").sort_index() for k, g in test.groupby(["modelo", "horizonte"])}
    print(f"Test: {len(test):,} filas en la intersección de {modelos}")

    # Regresión, niveles altos y picos
    filas = []
    for (m, h), g in datos.items():
        o, s = g["y_real_cm"].to_numpy(), g["y_pred_cm"].to_numpy()
        filas.append({"modelo": m, "horizonte": h, "n_horas": len(g), **metricas_regresion(o, s),
                      **metricas_niveles_altos(o, s, p95), **resumen_picos(eventos_pico(g.index, o, s))})
    met = pd.DataFrame(filas)
    rp = met[met["modelo"] == "Persistencia"].set_index("horizonte")["RMSE"]
    met["SS"] = 1 - met["RMSE"] / met["horizonte"].map(rp)

    # SS en walk-forward (mismas horas y definición que la búsqueda: media de SS por fold, mejor config.)
    ss_wf = []
    for (m, h, k), g in wf.groupby(["modelo", "horizonte", "fold"]):
        ss_wf.append({"modelo": m, "horizonte": h, "fold": k,
                      "RMSE": np.sqrt(np.mean((g["y_pred_cm"] - g["y_real_cm"]) ** 2))})
    ss_wf = pd.DataFrame(ss_wf)
    ref = ss_wf[ss_wf["modelo"] == "Persistencia"].set_index(["horizonte", "fold"])["RMSE"]
    ss_wf["SS"] = 1 - ss_wf["RMSE"] / ss_wf.set_index(["horizonte", "fold"]).index.map(ref)
    ss_fold = ss_wf.pivot_table(index=["modelo", "horizonte"], columns="fold", values="SS")
    ss_fold.columns = [f"SS_fold{c}" for c in ss_fold.columns]
    ss_fold["SS_medio"] = ss_fold.mean(axis=1)
    ss_fold.reset_index().round(4).to_csv(tab_dir / "ss_por_fold.csv", index=False)
    met["SS_walkforward"] = met.set_index(["modelo", "horizonte"]).index.map(ss_fold["SS_medio"])

    # Diebold–Mariano frente a la persistencia en test
    dm = []
    for m in modelos[1:]:
        for h in hs:
            gm, gp = datos[(m, h)], datos[("Persistencia", h)]
            c = gm.index.intersection(gp.index)
            em = (gm.loc[c, "y_pred_cm"] - gm.loc[c, "y_real_cm"]).to_numpy()
            ep = (gp.loc[c, "y_pred_cm"] - gp.loc[c, "y_real_cm"]).to_numpy()
            dm.append({"modelo": m, "horizonte": h, **diebold_mariano(em, ep, c, h),
                       "SS": 1 - np.sqrt(np.mean(em ** 2)) / np.sqrt(np.mean(ep ** 2))})
    dm = pd.DataFrame(dm)
    es_expl = dm["modelo"].isin(FAMILIAS)
    dm.loc[es_expl, "p_holm_exploratorios"] = holm(dm.loc[es_expl, "p_unilateral"])   # 5 familias × 2 horizontes
    dm["p_holm"] = dm.groupby("modelo")["p_unilateral"].transform(lambda p: pd.Series(holm(p), index=p.index))
    dm["rechaza_H0_5pct"] = (dm["SS"] > 0) & (dm["p_holm"] < ALFA_DM)
    dm.to_csv(tab_dir / "diebold_mariano.csv", index=False)
    met = met.merge(dm[["modelo", "horizonte", "DM_HLN", "p_unilateral", "p_holm", "p_holm_exploratorios"]],
                    on=["modelo", "horizonte"], how="left")

    # Alertas por instante (por horizonte y combinado) y por evento
    def estados_h(m, h):
        g = datos[(m, h)]
        return pd.DataFrame({"est_obs": a_estado(g["y_real_cm"]), "est_pred": a_estado(g["y_pred_cm"])},
                            index=pd.DatetimeIndex(g["t_emi"], name="t_emi")).sort_index()

    def estados_comb(m):
        j = pd.concat([estados_h(m, h).add_suffix(f"_{h}") for h in hs], axis=1).sort_index()
        cp, co = [f"est_pred_{h}" for h in hs], [f"est_obs_{h}" for h in hs]
        return pd.DataFrame({"est_pred": j[cp].max(axis=1, skipna=True), "est_obs": j[co].max(axis=1, skipna=True),
                             "completo": j[cp + co].notna().all(axis=1)}, index=j.index)

    al, ev_filas, detalle = [], [], []
    for m in modelos:
        series = {}
        for h in hs + ["combinado"]:
            e = estados_comb(m) if h == "combinado" else estados_h(m, h).dropna()
            if h == "combinado":
                series[h] = e["est_pred"]
                e = e[e["completo"]]
            else:
                series[h] = e["est_pred"]
            al.append({"modelo": m, "horizonte": h, **metricas_estados(e["est_obs"].to_numpy(int),
                                                                         e["est_pred"].to_numpy(int))})
        for h, est in series.items():
            ant = max(hs) if h == "combinado" else h
            for k in (1, 2):
                ev, falsas = simular_eventos(est, obs["test"], k, ant)
                v = ev[ev["evaluable"]]
                ev_filas.append({"modelo": m, "horizonte": h, "estado": ESTADOS[k], "n_eventos": len(ev),
                                 "n_evaluables": len(v), "detectados": int(v["detectado"].sum()),
                                 "omitidos": int((~v["detectado"]).sum()),
                                 "pct_detectados": 100 * v["detectado"].mean() if len(v) else np.nan,
                                 "anticipacion_media_h": v["anticipacion_h"].mean(),
                                 "anticipacion_max_posible_h": ant, "falsas_alarmas_episodios": falsas})
                detalle.append(ev.assign(modelo=m, horizonte=str(h), estado=ESTADOS[k]))
    al, ev_t = pd.DataFrame(al), pd.DataFrame(ev_filas)
    al.to_csv(tab_dir / "alertas_por_instante.csv", index=False)
    ev_t.to_csv(tab_dir / "alertas_por_evento.csv", index=False)
    pd.concat(detalle, ignore_index=True).to_csv(tab_dir / "alertas_eventos_detalle.csv", index=False)
    al_h = al[al["horizonte"] != "combinado"].astype({"horizonte": int})
    met = met.merge(al_h[["modelo", "horizonte", "F1_macro", "POD_aviso", "FAR_aviso", "POD_alerta", "FAR_alerta"]],
                    on=["modelo", "horizonte"], how="left")
    for estado in ("prealerta", "alerta"):
        sub = ev_t[(ev_t["estado"] == estado) & (ev_t["horizonte"] != "combinado")].astype({"horizonte": int})
        sub = sub.set_index(["modelo", "horizonte"])
        clave = met.set_index(["modelo", "horizonte"]).index
        met[f"eventos_{estado}_detectados"] = clave.map(sub["detectados"])
        met[f"eventos_{estado}_evaluables"] = clave.map(sub["n_evaluables"])
        met[f"anticipacion_{estado}_h"] = clave.map(sub["anticipacion_media_h"])
    met["familia"] = met["modelo"].map(NOMBRE)
    met["exploratorio"] = met["modelo"].isin(FAMILIAS)
    orden = {m: i for i, m in enumerate(modelos)}
    met = met.sort_values(["horizonte", "modelo"], key=lambda c: c.map(orden) if c.name == "modelo" else c)
    cols = ["familia", "modelo", "exploratorio", "horizonte", "n_horas", "SS", "SS_walkforward", "RMSE", "MAE",
            "sesgo", "NSE", "KGE", "KGE_r", "KGE_alfa", "KGE_beta", "PBIAS", "R2", "DM_HLN", "p_unilateral",
            "p_holm", "p_holm_exploratorios", "n_alto", "RMSE_alto", "MAE_alto", "sesgo_alto", "n_eventos",
            "error_pico_medio_cm", "pct_picos_subestimados", "desfase_medio_h", "F1_macro", "POD_aviso",
            "FAR_aviso", "POD_alerta", "FAR_alerta", "eventos_prealerta_detectados", "eventos_prealerta_evaluables",
            "anticipacion_prealerta_h", "eventos_alerta_detectados", "eventos_alerta_evaluables",
            "anticipacion_alerta_h"]
    met[cols].to_csv(expl / "metricas_test_por_horizonte.csv", index=False)
    met[cols].to_csv(tab_dir / "metricas_completas_test.csv", index=False)

    # Comparación con el RF oficial publicado y control de coherencia
    of = pd.read_csv(OFICIAL / "tablas" / "metricas_test_por_horizonte.csv")
    of = of[of["modelo"].isin(["RF", "Persistencia"])].assign(
        modelo=lambda d: d["modelo"].replace({"RF": "RF_oficial"}), fuente="tabla oficial publicada")
    rec = met[met["modelo"].isin(["RF_oficial", "Persistencia"])].set_index(["modelo", "horizonte"])
    dif = of.set_index(["modelo", "horizonte"])[["SS", "RMSE", "NSE", "KGE", "F1_macro"]] - \
        rec[["SS", "RMSE", "NSE", "KGE", "F1_macro"]]
    print("Control: recalculado − publicado (RF oficial y persistencia):\n", dif.round(6))
    comp_cols = ["familia", "modelo", "horizonte", "SS", "SS_walkforward", "RMSE", "NSE", "KGE", "RMSE_alto",
                 "F1_macro", "DM_HLN", "p_holm", "p_holm_exploratorios", "eventos_prealerta_detectados",
                 "eventos_alerta_detectados", "anticipacion_prealerta_h"]
    comp = pd.concat([of[["familia", "modelo", "horizonte", "SS", "RMSE", "NSE", "KGE", "RMSE_alto", "F1_macro",
                          "fuente"]],
                      met[met["exploratorio"]][comp_cols].assign(fuente="exploratorio (anexo)")], ignore_index=True)
    rf_ss = of[of["modelo"] == "RF_oficial"].set_index("horizonte")["SS"]
    comp["dif_SS_vs_RF_oficial"] = comp["SS"] - comp["horizonte"].map(rf_ss)
    comp.sort_values(["horizonte", "fuente"], ascending=[True, False]).to_csv(
        tab_dir / "comparacion_rf_oficial.csv", index=False)
    figuras(met, dm, ev_t, hs, fig_dir)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(met[["modelo", "horizonte", "SS_walkforward", "SS", "RMSE", "NSE", "KGE", "p_holm",
                   "p_holm_exploratorios", "F1_macro", "eventos_prealerta_detectados",
                   "eventos_alerta_detectados"]].round(4).to_string(index=False))
        print(ev_t[ev_t["horizonte"].astype(str) == "combinado"].round(2).to_string(index=False))


def figuras(met, dm, ev_t, hs, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orden = ["RF_oficial"] + [f for f in FAMILIAS if f in set(met["modelo"])]
    fig, ejes = plt.subplots(1, len(hs), figsize=(6.5 * len(hs), 4.3), sharey=True)
    for ax, h in zip(np.atleast_1d(ejes), hs):
        sub = met[met["horizonte"] == h].set_index("modelo").reindex(orden)
        x = np.arange(len(orden))
        ax.bar(x - 0.2, sub["SS_walkforward"], 0.4, label="walk-forward (val, media de folds)", color="#9bb7d4")
        ax.bar(x + 0.2, sub["SS"], 0.4, label="test", color="#2b5d8a")
        for xi, (m, r) in zip(x, sub.iterrows()):
            p = r["p_holm"]
            if pd.notna(p):
                ax.annotate(f"p={p:.3f}" if p >= 0.001 else "p<0.001", (xi + 0.2, r["SS"]), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(sub.loc["RF_oficial", "SS"], color="crimson", ls="--", lw=0.9, label="RF oficial (test)")
        ax.set_xticks(x, [NOMBRE[m] for m in orden], rotation=20)
        ax.set_title(f"τ = {h} h")
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(ejes)[0].set_ylabel("SS = 1 − RMSE / RMSE persistencia")
    np.atleast_1d(ejes)[0].legend(fontsize=8)
    fig.suptitle("Modelos exploratorios: SS en walk-forward y en test (p = DM-HLN con Holm por modelo)")
    fig.tight_layout()
    fig.savefig(fig_dir / "E01_ss_walkforward_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    modelos = ["Persistencia"] + orden
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, estado in zip(ejes, ("prealerta", "alerta")):
        sub = ev_t[(ev_t["estado"] == estado)]
        etiquetas = [str(h) for h in hs] + ["combinado"]
        x = np.arange(len(modelos))
        ancho = 0.8 / len(etiquetas)
        for i, h in enumerate(etiquetas):
            s = sub[sub["horizonte"].astype(str) == h].set_index("modelo").reindex(modelos)
            ax.bar(x + (i - (len(etiquetas) - 1) / 2) * ancho, s["detectados"], ancho,
                   label=f"τ = {h} h" if h != "combinado" else "combinado")
        n_ev = int(sub["n_evaluables"].max())
        ax.axhline(n_ev, color="gray", ls=":", lw=0.9, label=f"evaluables = {n_ev}")
        ax.set_xticks(x, [NOMBRE[m] for m in modelos], rotation=20)
        ax.set_title(f"Eventos de {estado} detectados en test")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "E02_eventos_detectados_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Métricas de test de los exploratorios del Yanuncay")
    p.add_argument("--unir", action="store_true", help="une partes/arboles y partes/redes antes de calcular")
    p.add_argument("--exploratorios", type=Path, default=EXPL)
    a = p.parse_args(argv)
    destino = a.exploratorios.resolve()
    if not str(destino).startswith(str(RES.resolve())):
        p.error("--exploratorios debe estar dentro de Docs/resources")
    if a.unir:
        unir_partes(destino)
    calcular(destino)


if __name__ == "__main__":
    main()
