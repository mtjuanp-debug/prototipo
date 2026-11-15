# -*- coding: utf-8 -*-
"""Repetición causal del conjunto de prueba del Yanuncay con el simulador (pregunta del revisor sobre fuga).

El entrenamiento y la evaluación fuera de línea usan entradas reconstruidas con métodos que pueden mirar ambos
extremos de un vacío (p. ej. PCHIP). Aquí se reproduce TODA la prueba hora a hora con el pronosticador del
prototipo, que solo usa información disponible hasta t (imputación causal: arrastre limitado, climatología de
entrenamiento), y se compara con la evaluación fuera de línea en las mismas horas:

- SS frente a la persistencia (nivel observado), en todas las horas y en las horas con/sin imputación en la
  ventana de 72 h;
- detección de eventos de prealerta y alerta con el estado combinado (más severo de los dos horizontes).

Salidas: ``Docs/resources/modelos_yanuncay/tablas/revisor_repeticion_causal.csv`` y
``revisor_repeticion_causal_eventos.csv``; el registro hora a hora en ``revisor_repeticion_causal_registro.parquet``.

Uso:  python src/metrics/repeticion_causal.py [--familia RF] [--horas N]
"""
from __future__ import annotations

import torch  # noqa: F401  (importar antes que xgboost/sklearn evita el error de c10.dll en Windows)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "apps" / "simulador_yanuncay"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configuracion import cargar_config, columnas_entrada, ruta  # noqa: E402
from pronosticador import Pronosticador  # noqa: E402
from sensores import SimuladorSensores, cargar_datos_observados  # noqa: E402

import robustez_revisor as rr  # noqa: E402

TAB = rr.TAB


def reproducir(familia: str, horas: int | None) -> pd.DataFrame:
    config = cargar_config("yanuncay")
    datos = cargar_datos_observados(config)
    test = datos.index[datos["particion"] == "test"]
    inicio, n = test[0], horas or len(test)
    pron = Pronosticador(config, datos[datos["particion"] == "train"], familia)
    sim = SimuladorSensores(datos, inicio, columnas_entrada(config))
    pron.precargar(sim.historia(pron.ventana_h + pron.margen_h))
    t0 = time.time()
    for i in range(n):
        if sim.agotado:
            break
        pron.recibir(sim.siguiente())
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{n:,} h ({time.time() - t0:,.0f} s)", flush=True)
    return pron.tabla_registro()


def comparar(reg: pd.DataFrame, familia: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = cargar_config("yanuncay")
    pred = pd.read_parquet(ruta(config, "predicciones"))
    pred = pred[(pred["particion"] == "test") & pred["fold"].isna()]
    limpia = rr._ventana_limpia()
    filas = []
    estados_causal, estados_offline = {}, {}
    for h in (4, 7):
        ref = pred[(pred["modelo"] == familia) & (pred["horizonte"] == h)].set_index("timestamp")
        per = pred[(pred["modelo"] == "Persistencia") & (pred["horizonte"] == h)].set_index("timestamp")
        c = reg[[f"nivel_h{h}"]].join(ref[["y_pred_cm", "y_real_cm"]], how="inner") \
            .join(per[["y_pred_cm"]].rename(columns={"y_pred_cm": "persistencia_cm"}), how="inner").dropna()
        estados_causal[h] = pd.Series(rr.estados(c[f"nivel_h{h}"], rr.me.Q_PRE, rr.me.Q_ALE), index=c.index)
        estados_offline[h] = pd.Series(rr.estados(c["y_pred_cm"], rr.me.Q_PRE, rr.me.Q_ALE), index=c.index)
        m_limpia = limpia.reindex(c.index).fillna(False).to_numpy(bool)
        for nombre, m in (("todas las horas", np.ones(len(c), bool)), ("ventana de 72 h sin imputación", m_limpia),
                          ("ventana con algún dato imputado", ~m_limpia)):
            o = c["y_real_cm"].to_numpy()[m]
            rp = np.sqrt(np.mean((c["persistencia_cm"].to_numpy()[m] - o) ** 2))
            r_off = np.sqrt(np.mean((c["y_pred_cm"].to_numpy()[m] - o) ** 2))
            r_cau = np.sqrt(np.mean((c[f"nivel_h{h}"].to_numpy()[m] - o) ** 2))
            dif = c[f"nivel_h{h}"].to_numpy()[m] - c["y_pred_cm"].to_numpy()[m]
            filas.append({"modelo": familia, "horizonte": h, "subconjunto": nombre, "n_horas": int(m.sum()),
                          "RMSE_persistencia_cm": rp, "RMSE_fuera_de_linea_cm": r_off, "RMSE_causal_cm": r_cau,
                          "SS_fuera_de_linea": 1 - r_off / rp, "SS_causal": 1 - r_cau / rp,
                          "RMSE_causal_vs_fuera_de_linea_cm": float(np.sqrt(np.mean(dif ** 2))),
                          "pct_horas_identicas": 100 * float((np.abs(dif) < 1e-3).mean())})
    obs = rr.me.cargar(rr.MOD / "exploratorios")[1]["test"]
    ev_filas = []
    for etiqueta, est in (("fuera de línea", estados_offline), ("causal", estados_causal)):
        comb = pd.concat(est, axis=1).max(axis=1, skipna=True).sort_index()
        for k, umbral in ((1, rr.me.Q_PRE), (2, rr.me.Q_ALE)):
            tabla, falsas = rr.eventos(comb, obs, k, 7, umbral)
            v = tabla[tabla["evaluable"]]
            det = int(v["detectado"].sum())
            lo, hi = rr.wilson(det, len(v))
            ev_filas.append({"modelo": familia, "evaluacion": etiqueta, "estado": rr.me.ESTADOS[k],
                             "n_eventos_evaluables": len(v), "detectados": det,
                             "POD_eventos": det / len(v) if len(v) else np.nan, "ic95_inf": lo, "ic95_sup": hi,
                             "anticipacion_media_h": v["anticipacion_h"].mean(), "falsas_alarmas_episodios": falsas})
    return pd.DataFrame(filas), pd.DataFrame(ev_filas)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--familia", default="RF")
    ap.add_argument("--horas", type=int, default=None, help="limitar la repetición (pruebas rápidas)")
    a = ap.parse_args(argv)
    reg = reproducir(a.familia, a.horas)
    sufijo = "" if a.horas is None else f"_{a.horas}h"
    reg.to_parquet(TAB / f"revisor_repeticion_causal_registro{sufijo}.parquet")
    ss, ev = comparar(reg, a.familia)
    ss.to_csv(TAB / f"revisor_repeticion_causal{sufijo}.csv", index=False)
    ev.to_csv(TAB / f"revisor_repeticion_causal_eventos{sufijo}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(ss.round(4).to_string(index=False))
        print(ev.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
