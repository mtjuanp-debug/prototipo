# -*- coding: utf-8 -*-
"""Evidencia complementaria para las preguntas del revisor (río Yanuncay, modelos finales).

Calcula, sin reentrenar y sin tocar la selección oficial, cinco tablas que alimentan los capítulos 3 a 5:

1. ``revisor_ic_eventos.csv``      intervalos de Wilson (95 %) de la detección de eventos y de la razón de falsas
                                   alarmas: con 13 eventos de prealerta y 7 de alerta la incertidumbre es alta.
2. ``revisor_sensibilidad_curva.csv``  efecto de un error relativo δ de la curva de descarga sobre las alertas: el
                                   prototipo emite con la curva nominal y los eventos «reales» se definen con
                                   Q_real = (1 + δ)·Q_nominal (δ = ±10 %, ±20 %).
3. ``revisor_imputacion.csv``      SS de prueba en todas las horas frente a las horas cuya ventana de 72 h no
                                   contiene ningún dato imputado (posible optimismo de la interpolación bilateral).
4. ``revisor_seleccion.csv``       orden de los candidatos en validación según el criterio global (SS) y según
                                   criterios de crecida (RMSE en niveles altos y puntaje de crecida del prototipo).
5. ``revisor_anticipacion.csv``    distribución de la anticipación efectiva por evento.

Reutiliza las funciones oficiales de ``src/models/exploratorios/metricas_exploratorios.py`` (mismas definiciones
que el notebook de métricas). Salidas en ``Docs/resources/modelos_yanuncay/tablas/``.

Uso:  python src/metrics/robustez_revisor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "models" / "exploratorios"))
import metricas_exploratorios as me  # noqa: E402

RES = REPO / "Docs" / "resources"
MOD = RES / "modelos_yanuncay"
TAB = MOD / "tablas"
CAPTURAS = RES / "capturas_app"
MODELO_SEL = "RF_oficial"          # nombre del RF oficial dentro de me.cargar()
DELTAS_CURVA = (-0.20, -0.10, 0.0, 0.10, 0.20)
VENTANA_IMP_H = 72
Z95 = 1.959964


# ------------------------------------------------------------------------------------------- utilidades
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción k/n (NaN si n = 0)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    medio = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centro - medio), min(1.0, centro + medio))


def estados(nivel, q_pre: float, q_ale: float) -> np.ndarray:
    """Estado 0/1/2 de cada nivel (cm) con umbrales de caudal dados; NaN si falta el nivel."""
    nivel = np.asarray(nivel, dtype=float)
    q = me.a_caudal(nivel)
    est = np.select([q >= q_ale, q >= q_pre], [2.0, 1.0], 0.0)
    return np.where(np.isnan(nivel), np.nan, est)


def eventos(estado_emitido: pd.Series, obs: pd.Series, k: int, anticipacion_max: int,
            umbral_q: float) -> tuple[pd.DataFrame, int]:
    """Igual que ``me.simular_eventos`` pero con el umbral de caudal del evento como parámetro."""
    o = obs.asfreq("h")
    e = estado_emitido.reindex(o.index)
    horas = me.a_horas(o.index)
    filas = []
    for ev in me.episodios(horas, me.a_caudal(o.to_numpy()), umbral_q):
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
        filas.append({"cruce": c, "evaluable": bool(np.any(~np.isnan(previos))), "detectado": racha >= 1,
                      "anticipacion_h": racha if racha >= 1 else np.nan})
    tabla = pd.DataFrame(filas, columns=["cruce", "evaluable", "detectado", "anticipacion_h"])
    em = estado_emitido.asfreq("h")
    falsas = 0
    for ep in me.episodios(me.a_horas(em.index), em.to_numpy(), k):
        ini, fin = em.index[ep[0]], em.index[ep[-1]] + pd.Timedelta(hours=anticipacion_max)
        if not (me.a_caudal(obs.loc[ini:fin].to_numpy()) >= umbral_q).any():
            falsas += 1
    return tabla, falsas


def _datos_test() -> tuple[dict, pd.Series, list[int]]:
    """Predicciones de prueba (intersección oficial de horas) por (modelo, horizonte) y nivel observado."""
    pred, obs = me.cargar(MOD / "exploratorios")
    test = me.interseccion(pred[(pred["particion"] == "test") & pred["fold"].isna()], ["horizonte"])
    hs = sorted(int(h) for h in test["horizonte"].unique())
    datos = {k: g.set_index("t_obj").sort_index() for k, g in test.groupby(["modelo", "horizonte"])}
    return datos, obs["test"], hs


def _combinado(datos: dict, modelo: str, hs: list[int], q_pre_obs: float, q_ale_obs: float,
               q_pre_pred: float = me.Q_PRE, q_ale_pred: float = me.Q_ALE) -> pd.DataFrame:
    """Estado emitido (máximo de los horizontes) y observado por instante de emisión."""
    partes = []
    for h in hs:
        g = datos[(modelo, h)]
        partes.append(pd.DataFrame({f"est_obs_{h}": estados(g["y_real_cm"], q_pre_obs, q_ale_obs),
                                    f"est_pred_{h}": estados(g["y_pred_cm"], q_pre_pred, q_ale_pred)},
                                   index=pd.DatetimeIndex(g["t_emi"], name="t_emi")))
    j = pd.concat(partes, axis=1).sort_index()
    cp, co = [f"est_pred_{h}" for h in hs], [f"est_obs_{h}" for h in hs]
    return pd.DataFrame({"est_pred": j[cp].max(axis=1, skipna=True), "est_obs": j[co].max(axis=1, skipna=True),
                         "completo": j[cp + co].notna().all(axis=1)}, index=j.index)


# ------------------------------------------------------------------------------------------- 1. IC de eventos
def ic_eventos() -> pd.DataFrame:
    ev = pd.read_csv(TAB / "alertas_por_evento.csv")
    ins = pd.read_csv(TAB / "alertas_por_instante.csv")
    filas = []
    for r in ev[ev["particion"].isin(["val", "test"])].itertuples():
        lo, hi = wilson(int(r.detectados), int(r.n_evaluables))
        filas.append({"modelo": r.modelo, "particion": r.particion, "horizonte": r.horizonte,
                      "metrica": f"deteccion_eventos_{r.estado}", "k": int(r.detectados), "n": int(r.n_evaluables),
                      "valor": r.detectados / r.n_evaluables if r.n_evaluables else np.nan,
                      "ic95_inf": lo, "ic95_sup": hi})
    for r in ins[ins["particion"] == "test"].itertuples():
        for etiqueta in ("aviso", "alerta"):
            vp, fp = int(getattr(r, f"VP_{etiqueta}")), int(getattr(r, f"FP_{etiqueta}"))
            lo, hi = wilson(fp, vp + fp)
            filas.append({"modelo": r.modelo, "particion": r.particion, "horizonte": r.horizonte,
                          "metrica": f"FAR_instante_{etiqueta}", "k": fp, "n": vp + fp,
                          "valor": fp / (vp + fp) if vp + fp else np.nan, "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------- 2. curva de descarga
def sensibilidad_curva(datos: dict, obs: pd.Series, hs: list[int], modelo: str = MODELO_SEL) -> pd.DataFrame:
    filas = []
    for d in DELTAS_CURVA:
        q_pre_obs, q_ale_obs = me.Q_PRE / (1 + d), me.Q_ALE / (1 + d)
        comb = _combinado(datos, modelo, hs, q_pre_obs, q_ale_obs)
        completo = comb[comb["completo"]]
        t2 = {k: me.tabla_2x2(completo["est_obs"].to_numpy(int), completo["est_pred"].to_numpy(int), k)
              for k in (1, 2)}
        for k, umbral_obs in ((1, q_pre_obs), (2, q_ale_obs)):
            tabla, falsas = eventos(comb["est_pred"], obs, k, max(hs), umbral_obs)
            v = tabla[tabla["evaluable"]]
            det = int(v["detectado"].sum())
            lo, hi = wilson(det, len(v))
            nivel_eq = me.CURVA_H0 + (umbral_obs / me.CURVA_A) ** (1 / me.CURVA_P)
            filas.append({"delta_curva": d, "estado": me.ESTADOS[k], "umbral_real_m3s_en_curva_nominal": umbral_obs,
                          "nivel_equivalente_real_cm": nivel_eq, "n_eventos_evaluables": len(v), "detectados": det,
                          "POD_eventos": det / len(v) if len(v) else np.nan, "ic95_inf": lo, "ic95_sup": hi,
                          "anticipacion_media_h": v["anticipacion_h"].mean(), "falsas_alarmas_episodios": falsas,
                          "POD_instante": t2[k]["POD"], "FAR_instante": t2[k]["FAR"]})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------- 3. imputación
def _ventana_limpia() -> pd.Series:
    """True en las horas cuya ventana de 72 h (incluida la hora) no tiene ningún dato de entrada imputado."""
    partes = []
    for p in ("val", "test"):
        d = pd.read_parquet(RES / f"{p}_yanuncay.parquet")
        cols = [c for c in d.columns if c.endswith("__imp")]
        partes.append(d[cols].max(axis=1))
    imp = pd.concat(partes).sort_index()
    imp.index = pd.to_datetime(imp.index, utc=True)
    imp = imp[~imp.index.duplicated()].asfreq("h").fillna(1.0)   # hora ausente: se trata como no limpia
    return imp.rolling(VENTANA_IMP_H, min_periods=VENTANA_IMP_H).max().eq(0)


def imputacion(datos: dict, hs: list[int], modelo: str = MODELO_SEL) -> pd.DataFrame:
    limpia = _ventana_limpia()
    filas = []
    for h in hs:
        g, gp = datos[(modelo, h)], datos[("Persistencia", h)]
        comun = g.index.intersection(gp.index)
        g, gp = g.loc[comun], gp.loc[comun]
        mascara = limpia.reindex(pd.DatetimeIndex(g["t_emi"])).fillna(False).to_numpy(bool)
        for nombre, m in (("todas las horas", np.ones(len(g), bool)), ("ventana de 72 h sin imputación", mascara),
                          ("ventana con algún dato imputado", ~mascara)):
            o, s, sp = (g["y_real_cm"].to_numpy()[m], g["y_pred_cm"].to_numpy()[m], gp["y_pred_cm"].to_numpy()[m])
            rmse, rmse_p = np.sqrt(np.mean((s - o) ** 2)), np.sqrt(np.mean((sp - o) ** 2))
            filas.append({"modelo": "RF" if modelo == MODELO_SEL else modelo, "horizonte": h, "subconjunto": nombre,
                          "n_horas": int(m.sum()), "pct_horas": 100 * m.mean(), "RMSE_modelo_cm": rmse,
                          "RMSE_persistencia_cm": rmse_p, "SS": 1 - rmse / rmse_p,
                          "n_horas_sobre_prealerta": int((me.a_caudal(o) >= me.Q_PRE).sum())})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------- 4. selección
def seleccion() -> pd.DataFrame:
    sel = pd.read_csv(MOD / "seleccion_modelo.csv")[["modelo", "SS_global", "RMSE_alto_global", "seleccionado"]]
    rk = pd.read_csv(CAPTURAS / "confianza_ranking.csv")[["modelo", "puntaje", "eventos_h4", "eventos_h7",
                                                           "far_h4", "far_h7"]]
    t = sel.merge(rk, on="modelo", how="left").rename(columns={"puntaje": "puntaje_crecida"})
    t["rango_SS"] = t["SS_global"].rank(ascending=False, method="min").astype(int)
    t["rango_RMSE_alto"] = t["RMSE_alto_global"].rank(ascending=True, method="min").astype(int)
    t["rango_puntaje_crecida"] = t["puntaje_crecida"].rank(ascending=False, method="min").astype(int)
    lider = t["puntaje_crecida"].max()
    t["empata_en_crecida"] = (lider - t["puntaje_crecida"]) < me.EMPATE_SS
    return t.sort_values("rango_SS")


# ------------------------------------------------------------------------------------------- 5. anticipación
def anticipacion() -> pd.DataFrame:
    det = pd.read_csv(TAB / "alertas_eventos_detalle.csv")
    det = det[(det["particion"] == "test") & det["evaluable"]]
    filas = []
    for (m, h, e), g in det.groupby(["modelo", "horizonte", "estado"]):
        a = g["anticipacion_h"].fillna(0)
        filas.append({"modelo": m, "horizonte": h, "estado": e, "n_eventos": len(g),
                      "detectados": int(g["detectado"].sum()), "anticipacion_media_h": g["anticipacion_h"].mean(),
                      "anticipacion_mediana_h": g["anticipacion_h"].median(),
                      "anticipacion_max_h": g["anticipacion_h"].max(),
                      "eventos_con_1h_o_mas": int((a >= 1).sum()), "eventos_con_2h_o_mas": int((a >= 2).sum()),
                      "eventos_con_3h_o_mas": int((a >= 3).sum()),
                      "pct_eventos_con_2h_o_mas": 100 * (a >= 2).mean()})
    return pd.DataFrame(filas)


def main() -> None:
    datos, obs, hs = _datos_test()
    salidas = {"revisor_ic_eventos.csv": ic_eventos(),
               "revisor_sensibilidad_curva.csv": sensibilidad_curva(datos, obs, hs),
               "revisor_imputacion.csv": imputacion(datos, hs),
               "revisor_seleccion.csv": seleccion(),
               "revisor_anticipacion.csv": anticipacion()}
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for nombre, tabla in salidas.items():
            tabla.to_csv(TAB / nombre, index=False)
            vista = tabla
            if "modelo" in tabla and nombre != "revisor_seleccion.csv":
                vista = tabla[tabla["modelo"].isin(["RF", "RF_oficial"])]
            print(f"\n== {nombre} ({len(tabla)} filas)\n{vista.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
