"""Tablas de evaluación del top de modelos y de su nivel de confianza (para el Capítulo 4).

Genera en ``--salida`` (por defecto ``Docs/resources/capturas_app``):
- ``confianza_ranking.csv``: ranking de modelos por desempeño en crecidas en validación (val_fold).
- ``confianza_cobertura_test.csv``: cobertura real de la banda al 90 % y error de Brier de la probabilidad
  de prealerta en el conjunto de prueba, que no se usó para calibrar.
- ``confianza_crecida.csv``: pronósticos del top (+4 h y +7 h), con banda, probabilidad y consenso, hora a
  hora durante una crecida real reproducida con el simulador (por defecto la del 2025-03-02).

Para los demás ríos (``--rio tomebamba`` o ``--rio tarqui``) los archivos llevan el sufijo ``_<rio>`` y la
crecida por defecto es la de ``inicio_crecida`` de su configuración.

Uso (desde la raíz del repositorio):
    python src/apps/simulador_yanuncay/evaluar_confianza.py [--rio yanuncay] [--inicio "2025-03-02 00:00"] [--horas 30]
"""
from __future__ import annotations

import torch  # noqa: F401  (importar antes que xgboost/sklearn evita el error de c10.dll en Windows)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from confianza import calibrar, ranking  # noqa: E402
from configuracion import RAIZ_REPO, cargar_config, columnas_entrada, ruta  # noqa: E402
from hidrologia import umbrales_cm  # noqa: E402
from imputacion_causal import ajustar_parametros  # noqa: E402
from pronosticador import Pronosticador  # noqa: E402
from sensores import SimuladorSensores, cargar_datos_observados  # noqa: E402


def cobertura_test(pred: pd.DataFrame, cal: dict, um: dict[str, float], p95: float) -> pd.DataFrame:
    """Cobertura de la banda al 90 % y Brier de P(prealerta) en test, por modelo y horizonte."""
    filas = []
    t = pred[(pred["particion"] == "test") & pred["y_real_cm"].notna() & pred["y_pred_cm"].notna()]
    for (m, h), g in t.groupby(["modelo", "horizonte"]):
        if (m, h) not in cal:
            continue
        ev = [cal[(m, h)].evaluar(v, um) for v in g["y_pred_cm"]]
        inf = np.array([e["inf90"] for e in ev])
        sup = np.array([e["sup90"] for e in ev])
        pp = np.array([e["p_prealerta"] for e in ev])
        y = g["y_real_cm"].to_numpy()
        dentro = (y >= inf) & (y <= sup)
        alto = g["y_pred_cm"].to_numpy() >= p95
        o = (y >= um["prealerta"]).astype(float)
        filas.append({"modelo": m, "horizonte": int(h), "n": len(g), "cobertura90": dentro.mean(),
                      "cobertura90_alto": dentro[alto].mean() if alto.any() else np.nan,
                      "ancho90_alto_cm": (sup - inf)[alto].mean() if alto.any() else np.nan,
                      "brier_prealerta": np.mean((pp - o) ** 2), "brier_climatologia": np.mean((o.mean() - o) ** 2),
                      "p_media_con_prealerta": pp[o == 1].mean() if (o == 1).any() else np.nan,
                      "p_media_sin_prealerta": pp[o == 0].mean()})
    return pd.DataFrame(filas)


def crecida(config: dict, cal: dict, rk: pd.DataFrame, inicio: str, horas: int) -> pd.DataFrame:
    """Reproduce ``horas`` desde ``inicio`` con el top de 5 modelos y registra pronósticos y confianza."""
    datos = cargar_datos_observados(config)
    cols = columnas_entrada(config)
    params = ajustar_parametros(datos.loc[datos["particion"] == "train", cols], config)
    top = [m for m in rk["modelo"] if m != "Persistencia"][:5]
    pron = Pronosticador(config, datos, top[0], parametros=params)
    pron.configurar_top(top, cal, dict(zip(rk["modelo"], rk["puntaje"])))
    t0 = pd.Timestamp(inicio)
    sim = SimuladorSensores(datos, t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC"), cols)
    pron.precargar(sim.historia(pron.ventana_h + pron.margen_h))
    obj = datos[config["objetivo"]]
    filas = []
    for _ in range(horas):
        r = pron.recibir(sim.siguiente())
        for h in pron.horizontes:
            c = r.consenso.get(h) or {}
            votos = c.get("votos", {})
            fila = {"t_utc": r.ts, "horizonte": h, "nivel_obs_cm": r.nivel_obs,
                    "nivel_real_t_mas_h_cm": obj.get(r.ts + pd.Timedelta(hours=h), np.nan),
                    "consenso_cm": c.get("nivel_cm", np.nan), "consenso_estado": c.get("estado", ""),
                    "votos_prealerta_o_mas": votos.get("prealerta", 0) + votos.get("alerta", 0),
                    "n_modelos": c.get("n", 0)}
            for m, p in r.top[h].items():
                fila[f"{m}_cm"] = p["nivel_cm"]
                fila[f"{m}_inf90"] = p.get("inf90", np.nan)
                fila[f"{m}_sup90"] = p.get("sup90", np.nan)
                fila[f"{m}_p_prealerta"] = p.get("p_prealerta", np.nan)
            filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rio", default="yanuncay")
    ap.add_argument("--inicio", default=None,
                    help="hora UTC de inicio de la crecida a reproducir (por defecto, inicio_crecida del río)")
    ap.add_argument("--horas", type=int, default=30)
    ap.add_argument("--salida", default=str(RAIZ_REPO / "Docs" / "resources" / "capturas_app"))
    args = ap.parse_args()
    salida = Path(args.salida).resolve()
    if RAIZ_REPO.resolve() not in salida.parents:
        raise SystemExit("--salida debe quedar dentro del repositorio")
    salida.mkdir(parents=True, exist_ok=True)

    config = cargar_config(args.rio)
    um = umbrales_cm(config)
    p95 = float(json.loads(ruta(config, "configuracion_modelos").read_text(encoding="utf-8"))["p95_train_cm"])
    pred = pd.read_parquet(ruta(config, "predicciones"))
    cal = calibrar(pred, um, p95, modelos=config["familias"])
    rk = ranking(cal, [int(h) for h in config["horizontes"]], config["familias"])
    sufijo = "" if args.rio == "yanuncay" else f"_{args.rio}"  # el Capítulo 4 lee los nombres sin sufijo
    inicio = args.inicio or config.get("inicio_crecida", config["inicio_defecto"])
    rk.to_csv(salida / f"confianza_ranking{sufijo}.csv", index=False)
    cobertura_test(pred, cal, um, p95).to_csv(salida / f"confianza_cobertura_test{sufijo}.csv", index=False)
    crecida(config, cal, rk, inicio, args.horas).to_csv(salida / f"confianza_crecida{sufijo}.csv", index=False)
    print("Tablas escritas en", salida)


if __name__ == "__main__":
    main()
