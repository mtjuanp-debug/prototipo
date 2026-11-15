# -*- coding: utf-8 -*-
"""Une las salidas de ``modelos_<slug>.ipynb`` ejecutado por partes (``<PREFIJO>FAMILIAS``) en una sola carpeta.

Uso (desde la raíz del repositorio)::

    python src/models/unir_partes.py --rio tomebamba PARTE1 PARTE2 ... [--destino Docs/resources/modelos_tomebamba]

Sin ``--destino`` se usa ``rutas.modelos`` del río en ``config/rios.json``. Concatena búsqueda, predicciones,
importancias e historiales, copia los modelos, une los contratos de carga (``modelos/formulacion_<slug>.json``; todas
las partes deben usar la misma formulación) y rehace la selección global (Tesis §1.3, P57): mayor SS promedio entre
folds y horizontes; si otra familia queda a menos de 0,01 del líder, gana la de menor RMSE en niveles altos
(> P95 de train).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

EMPATE_SS = 0.01
RAIZ = Path(__file__).resolve().parents[2]
RUTA_RIOS = RAIZ / "config" / "rios.json"


def rios_disponibles(ruta: Path = RUTA_RIOS) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))["rios"]


def leer_csv(partes, nombre):
    tablas = [pd.read_csv(p / nombre) for p in partes if (p / nombre).exists()]
    return pd.concat(tablas, ignore_index=True) if tablas else None


def _argumentos(argv: Sequence[str] | None, rios: dict) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Une las partes del entrenamiento de un río.")
    ap.add_argument("--rio", required=True, choices=sorted(rios), help="slug del río en config/rios.json")
    ap.add_argument("partes", nargs="+", type=Path, help="carpetas <PREFIJO>OUT_DIR de cada parte")
    ap.add_argument("--destino", type=Path, help="carpeta de salida (por defecto rutas.modelos del río)")
    args = ap.parse_args(argv)
    if not re.match(r"^[a-z][a-z0-9_]{1,30}$", args.rio):
        ap.error(f"slug no válido: {args.rio!r}")
    faltan = [str(p) for p in args.partes if not p.is_dir()]
    if faltan:
        ap.error(f"no existen las carpetas de partes: {faltan}")
    if args.destino is None:
        args.destino = RAIZ / rios[args.rio]["rutas"]["modelos"]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    rios = rios_disponibles()
    args = _argumentos(argv, rios)
    slug, partes, destino = args.rio, args.partes, args.destino
    nombre_pred = f"predicciones_{slug}.parquet"
    nombre_formulacion = f"formulacion_{slug}.json"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "modelos").mkdir(exist_ok=True)

    # Búsqueda e importancias: cada familia aparece en una sola parte
    for nombre in ("busqueda_resultados.csv", "importancia_permutacion.csv", "shap_importancia.csv",
                   "rf_curva_profundidad.csv"):
        t = leer_csv(partes, nombre)
        if t is not None:
            t.drop_duplicates().to_csv(destino / nombre, index=False)

    # Predicciones: la persistencia se repite en cada parte; se conserva una sola copia
    pred = pd.concat([pd.read_parquet(p / nombre_pred) for p in partes], ignore_index=True)
    claves = ["timestamp", "particion", "horizonte", "modelo", "fold"]
    pred = pred.drop_duplicates(subset=claves, keep="first")
    pred.to_parquet(destino / nombre_pred, index=False)

    # Selección global
    sel = leer_csv(partes, "seleccion_modelo.csv").drop_duplicates(subset=["familia"])
    sel = sel.sort_values("SS_global", ascending=False).reset_index(drop=True)
    lider = sel.iloc[0]
    empatados = sel[sel["SS_global"] >= lider["SS_global"] - EMPATE_SS]
    ganador = empatados.sort_values("RMSE_alto_global").iloc[0]
    sel["seleccionado"] = sel["familia"] == ganador["familia"]
    sel.to_csv(destino / "seleccion_modelo.csv", index=False)

    res = leer_csv(partes, "resumen_entrenamiento.csv")
    res = res.drop_duplicates(subset=["familia", "horizonte"], keep="first")
    res["seleccionado"] = res["familia"] == ganador["familia"]
    res.to_csv(destino / "resumen_entrenamiento.csv", index=False)

    # Historiales y configuración
    hist, conf = {}, None
    for p in partes:
        hist.update(json.loads((p / "historiales.json").read_text(encoding="utf-8")))
        c = json.loads((p / "configuracion.json").read_text(encoding="utf-8"))
        if conf is None:
            conf = c
            conf["seleccion_por_familia"] = {}
            conf["candidatos"] = {}
        conf["seleccion_por_familia"].update(c.get("seleccion_por_familia", {}))
        conf["candidatos"].update(c.get("candidatos", {}))
    (destino / "historiales.json").write_text(json.dumps(hist), encoding="utf-8")

    cols_cfg = [c for c in sel.columns if c.startswith("config_h")]
    conf["seleccion_global"] = {
        "familia": ganador["familia"], "modelo": ganador["modelo"],
        "criterio": "mayor SS promedio entre folds y horizontes",
        "SS_global": float(ganador["SS_global"]),
        "P95_train_cm": conf.get("p95_train_cm"),
        "configuraciones": {c.replace("config_h", ""): ganador[c] for c in cols_cfg},
        "empate_resuelto_por_RMSE_alto": bool(len(empatados) > 1),
    }
    conf["partes_unidas"] = [str(p) for p in partes]
    (destino / "configuracion.json").write_text(json.dumps(conf, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")

    # Modelos finales y contrato de carga (formulacion_<slug>.json): se unen los modelos de todas las partes
    formulacion = None
    for p in partes:
        for f in (p / "modelos").glob("*"):
            if f.name == nombre_formulacion:
                continue
            shutil.copy2(f, destino / "modelos" / f.name)
        ruta_f = p / "modelos" / nombre_formulacion
        if ruta_f.exists():
            fp = json.loads(ruta_f.read_text(encoding="utf-8"))
            if formulacion is None:
                formulacion = {**fp, "modelos": {}}
            claves = ("objetivo", "variables", "features_tabulares", "features_secuencia", "escaladores")
            distintas = [k for k in claves if fp.get(k) != formulacion.get(k)]
            if distintas:
                raise ValueError(f"La parte {p} usa otra formulación ({distintas}); no se pueden unir.")
            if fp.get("pesos", {}).get("alfa") != formulacion.get("pesos", {}).get("alfa"):
                raise ValueError(f"La parte {p} usa otro α de pesos; no se pueden unir.")
            formulacion["modelos"].update(fp.get("modelos", {}))
    if formulacion is not None:
        formulacion["seleccion_global"] = conf["seleccion_global"]
        formulacion["partes_unidas"] = conf["partes_unidas"]
        (destino / "modelos" / nombre_formulacion).write_text(
            json.dumps(formulacion, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Contrato de carga: {len(formulacion['modelos'])} modelos -> {destino / 'modelos' / nombre_formulacion}")

    print(f"Partes unidas ({slug}): {len(partes)} -> {destino}")
    print(sel[["familia", "SS_global", "RMSE_alto_global", "seleccionado"]].to_string(index=False))


if __name__ == "__main__":
    main()
