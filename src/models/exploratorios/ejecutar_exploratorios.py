# -*- coding: utf-8 -*-
"""Ejecuta los modelos exploratorios del Yanuncay (LightGBM, CatBoost, GRU, TCN, N-HiTS) para el anexo.

Reproduce el contrato de ``src/models/modelos_yanuncay.ipynb`` (secciones 5–9): búsqueda aleatoria con
walk-forward expansivo en val (5 bloques, margen ``margen_h``), early stopping en el último 10 % de cada fold,
selección por SS medio frente a la persistencia (desempate < 0,01 con el RMSE en niveles > P95 de train),
reentrenamiento con train + val y una única predicción de test. Las salidas usan los mismos nombres y columnas
que las oficiales (``busqueda_resultados.csv``, ``predicciones_yanuncay.parquet`` en cm, ``resumen_entrenamiento.csv``,
``seleccion_modelo.csv``, ``historiales.json``, ``configuracion.json`` y ``modelos/``), de modo que el notebook de
métricas puede leerlas apuntando a la carpeta de salida. Esa lógica solo existe dentro del notebook oficial, así
que aquí se reimplementa de forma fiel (ver ``datos_exploratorios.py``).

Fases
  humo         2 configuraciones, fold 5, pocas épocas/rondas; corre todo el flujo (incluido test) para verificar.
  walkforward  búsqueda + selección; guarda búsqueda y predicciones val_fold (test no se toca).
  final        reanuda la búsqueda de la carpeta de salida (corre lo que falte), selecciona, reentrena con
               train + val y predice test una sola vez.

Ejemplo:  python src/models/exploratorios/ejecutar_exploratorios.py --fase walkforward --objetivo delta \
              --pesos-alpha 4 --datos Docs/resources --salida Docs/resources/modelos_exploratorios_delta
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import torch  # antes que numpy/sklearn (Windows: c10.dll)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_exploratorios as de       # noqa: E402
import espacios_busqueda as eb         # noqa: E402
import modelos_exploratorios as me     # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FAMILIAS_DEFECTO = "LGBM,GRU,TCN,NHITS"
N_CURVA_TRAIN = 10000
FORMULACION_OFICIAL = REPO / "Docs" / "resources" / "modelos_yanuncay" / "modelos" / "formulacion_yanuncay.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modelos exploratorios del Yanuncay (anexo)")
    p.add_argument("--fase", choices=["humo", "walkforward", "final"], required=True)
    p.add_argument("--objetivo", choices=list(de.OBJETIVOS), default="nivel")
    p.add_argument("--pesos-alpha", type=float, default=0.0, help="w = 1 + α·1[y(t+τ) > umbral]; 0 = sin pesos")
    p.add_argument("--pesos-umbral", choices=list(de.UMBRALES_PESO), default="p95")
    p.add_argument("--pesos-columna", default=None, help="columna del parquet que multiplica los pesos")
    p.add_argument("--datos", type=Path, default=REPO / "Docs" / "resources")
    p.add_argument("--horizontes-json", type=Path, default=None)
    p.add_argument("--formulacion", type=Path, default=FORMULACION_OFICIAL,
                   help="JSON de la formulación oficial: fija la lista y el orden de variables ('ninguna' = catálogo)")
    p.add_argument("--salida", type=Path, default=None)
    p.add_argument("--familias", default=FAMILIAS_DEFECTO, help=f"subconjunto de {','.join(eb.FAMILIAS)}")
    p.add_argument("--n-config", type=int, default=None, help="configuraciones por familia (todas)")
    p.add_argument("--n-config-arboles", type=int, default=30)
    p.add_argument("--n-config-redes", type=int, default=12)
    p.add_argument("--folds", default=None, help="p. ej. 1,2,3,4,5 (humo: 5)")
    p.add_argument("--epocas-humo", type=int, default=3)
    p.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--reanudar", action="store_true", help="reutiliza la búsqueda de --salida (final: siempre)")
    a = p.parse_args(argv)
    # Validación en la frontera del sistema
    a.familias = [f.strip().upper() for f in a.familias.split(",") if f.strip()]
    if not a.familias or not set(a.familias) <= set(eb.FAMILIAS):
        p.error(f"--familias debe ser un subconjunto de {eb.FAMILIAS}")
    if a.pesos_alpha < 0 or not np.isfinite(a.pesos_alpha):
        p.error("--pesos-alpha debe ser un número >= 0")
    humo = a.fase == "humo"
    a.folds = [int(k) for k in (a.folds or ("5" if humo else "1,2,3,4,5")).split(",")]
    if not all(1 <= k <= de.N_BLOQUES for k in a.folds) or len(set(a.folds)) != len(a.folds):
        p.error("--folds debe contener valores distintos entre 1 y 5")
    if a.n_config is not None:
        a.n_config_arboles = a.n_config_redes = a.n_config
    elif humo:
        a.n_config_arboles = a.n_config_redes = 2
    if min(a.n_config_arboles, a.n_config_redes) < 1:
        p.error("--n-config debe ser >= 1")
    if a.formulacion is not None and str(a.formulacion).lower() == "ninguna":
        a.formulacion = None
    elif a.formulacion is not None:
        if not a.formulacion.is_file():
            p.error(f"No existe {a.formulacion}")
        a.formulacion = a.formulacion.resolve()
        j = json.load(open(a.formulacion, encoding="utf-8"))
        if j.get("objetivo") != a.objetivo or float(j.get("pesos", {}).get("alfa", 0.0)) != a.pesos_alpha:
            print(f"AVISO: la formulación oficial es objetivo={j.get('objetivo')}, α={j.get('pesos', {}).get('alfa')};"
                  f" esta corrida usa objetivo={a.objetivo}, α={a.pesos_alpha}")
    a.datos = a.datos.resolve()
    for nombre in ("train_yanuncay.parquet", "catalogo_features_yanuncay.csv", "escaladores_yanuncay.csv"):
        if not (a.datos / nombre).is_file():
            p.error(f"No existe {a.datos / nombre}")
    if a.salida is None:
        sufijo = f"_a{a.pesos_alpha:g}" if a.pesos_alpha else ""
        a.salida = REPO / "Docs" / "resources" / f"modelos_exploratorios_{'humo_' if humo else ''}{a.objetivo}{sufijo}"
    a.salida = a.salida.resolve()
    a.reanudar = a.reanudar or a.fase == "final"
    return a


class Ejecucion:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a, self.humo = a, a.fase == "humo"
        self.out = a.salida
        self.dir_modelos, self.dir_preds = self.out / "modelos", self.out / "pred_folds"
        self.dir_modelos.mkdir(parents=True, exist_ok=True)
        self.dir_preds.mkdir(exist_ok=True)
        self.m = de.cargar_malla(a.datos, a.horizontes_json, a.pesos_columna, a.formulacion)
        cuda = torch.cuda.is_available() and a.dispositivo != "cpu"
        self.disp = torch.device("cuda" if cuda else "cpu")
        self.x_seq = me.a_tensor(self.m.x_seq, self.disp)
        self.idx_obj = self.m.feat_seq.index(de.TARGET)
        self.firma = {"objetivo": a.objetivo, "pesos_alpha": a.pesos_alpha, "pesos_umbral": a.pesos_umbral,
                      "pesos_columna": a.pesos_columna, "datos": str(a.datos), "humo": self.humo,
                      "feat_tab": self.m.feat_tab}
        self.candidatos = {f: eb.generar_candidatos(
            f, a.n_config_arboles if f in eb.FAMILIAS_ARBOL else a.n_config_redes, self.humo) for f in a.familias}
        self.busqueda, self.pred_folds = [], {}
        self.predicciones, self.resumen, self.historiales, self.seleccion = [], [], {}, {}
        self._cargar_previo()

    # -------------------------------------------------------------------------------------------- estado previo
    def _cargar_previo(self) -> None:
        ruta_cfg, ruta_bus = self.out / "configuracion.json", self.out / "busqueda_resultados.csv"
        if not self.a.reanudar or not ruta_bus.exists():
            if self.a.fase == "final":
                raise FileNotFoundError(f"La fase final necesita la búsqueda previa en {ruta_bus}")
            return
        if ruta_cfg.exists():
            previa = json.load(open(ruta_cfg, encoding="utf-8")).get("firma", {})
            distintos = [k for k in self.firma if k in previa and previa[k] != self.firma[k]]
            if distintos:
                raise ValueError(f"La salida existente se generó con otra formulación ({distintos}); use otra --salida")
        self.busqueda = pd.read_csv(ruta_bus).to_dict("records")
        for archivo in self.dir_preds.glob("*.npz"):
            familia, tau = archivo.stem.rsplit("_h", 1)
            with np.load(archivo) as z:
                for clave in z.files:
                    cid, k = clave.rsplit("__f", 1)
                    self.pred_folds.setdefault((familia, int(tau), cid), {})[int(k)] = z[clave]
        print(f"Reanudando: {len(self.busqueda)} filas de búsqueda previas")

    def _guardar_pred_folds(self, familia: str, tau: int) -> None:
        arrays = {f"{cid}__f{k}": v for (f, t, cid), d in self.pred_folds.items() if f == familia and t == tau
                  for k, v in d.items()}
        np.savez_compressed(self.dir_preds / f"{familia}_h{tau}.npz", **arrays)

    # -------------------------------------------------------------------------------------------- registro
    def registrar(self, modelo: str, familia: str, H: de.Horizonte, particion: str, pos: np.ndarray,
                  pred_nivel: np.ndarray, fold=None) -> None:
        self.predicciones.append(pd.DataFrame({
            "timestamp": self.m.ts[pos], "particion": particion, "horizonte": H.tau, "modelo": modelo,
            "familia": familia, "fold": pd.array([fold] * len(pos), dtype="Int64"),
            "y_real_cm": self.m.a_cm(H.y_nivel[pos]), "y_pred_cm": self.m.a_cm(pred_nivel)}))

    def rmse_nivel(self, H: de.Horizonte, pos: np.ndarray, pred_nivel: np.ndarray) -> float:
        return self.m.rmse_cm(H.y_nivel[pos], pred_nivel)

    # -------------------------------------------------------------------------------------------- ajuste
    def ajustar(self, familia: str, cfg: dict, H: de.Horizonte, pos_ent: np.ndarray, datos_red, final=False):
        """Ajusta con early stopping en el último 10 % de pos_ent. Devuelve (predictor de nivel, info, objeto)."""
        pos_fit, pos_es = de.division_es(pos_ent)
        etiqueta = f"{familia} h={H.tau}"
        if familia in eb.FAMILIAS_ARBOL:
            X = self.m.x_tab
            w = None if H.pesos is None else H.pesos[pos_fit]
            curva = de.submuestra(pos_fit, 3000 if self.humo else N_CURVA_TRAIN) if final else None
            res = me.AJUSTADORES_ARBOL[familia](
                cfg, eb.comun_arbol(familia, self.humo), X[pos_fit], H.y[pos_fit], w, X[pos_es], H.y[pos_es],
                None if curva is None else X[curva], None if curva is None else H.y[curva])
            info = {"mejor_iteracion": res.mejor_iteracion}
            hist = {"tipo": "rondas", "train": [v * self.m.rango_cm for v in res.curva_train],
                    "val": [v * self.m.rango_cm for v in res.curva_control], "mejor_iteracion": res.mejor_iteracion}
            return (lambda p: H.a_nivel(res.predecir(X[p]), p)), info, res, hist
        modelo, hist, mejor = me.entrenar_red(familia, cfg, datos_red, pos_fit, pos_es, self.idx_obj, H.tau,
                                              de.SEED, self.m.rango_cm, etiqueta, verboso=final)
        info = {"mejor_iteracion": mejor, "epocas": len(hist["train"]),
                "s_por_epoca": round(float(np.mean(hist["s_epoca"])), 3)}
        hist = {"tipo": "epocas", **hist, "mejor_iteracion": mejor}
        return (lambda p: H.a_nivel(me.predecir_red(modelo, datos_red, p), p)), info, modelo, hist

    def datos_red(self, H: de.Horizonte) -> me.DatosRed:
        pesos = None if H.pesos is None else me.a_tensor(H.pesos, self.disp)
        return me.DatosRed(self.x_seq, me.a_tensor(np.nan_to_num(H.y), self.disp), pesos, de.LOOKBACK_H)

    def _liberar(self) -> None:
        gc.collect()
        if self.disp.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------------------------- búsqueda
    def _config_completa(self, familia: str, tau: int, cid: str, cfg: dict, folds: dict) -> bool:
        filas = [f for f in self.busqueda if f["familia"] == familia and int(f["horizonte"]) == tau
                 and f["config_id"] == cid and f["config_json"] == json.dumps(cfg) and np.isfinite(f["RMSE"])]
        preds = self.pred_folds.get((familia, tau, cid), {})
        return {int(f["fold"]) for f in filas} >= set(folds) and set(preds) >= set(folds)

    def buscar(self, familia: str, H: de.Horizonte, folds: dict, rmse_pers: dict, datos_red) -> None:
        tau = H.tau
        for cid, cfg in self.candidatos[familia]:
            if self._config_completa(familia, tau, cid, cfg, folds):
                continue
            self.busqueda = [f for f in self.busqueda if not (f["familia"] == familia and int(f["horizonte"]) == tau
                                                              and f["config_id"] == cid)]
            preds = {}
            for k, (pos_ent, pos_eva) in folds.items():
                t0 = time.time()
                try:
                    predecir, info, _, _ = self.ajustar(familia, cfg, H, pos_ent, datos_red)
                    preds[k] = predecir(pos_eva).astype(np.float32)
                    rmse = self.rmse_nivel(H, pos_eva, preds[k])
                except Exception as error:          # una configuración fallida no detiene la búsqueda
                    if self.humo:
                        raise
                    print(f"    Falla {cid} fold {k}: {error}")
                    rmse, info = np.nan, {}
                self.busqueda.append({"familia": familia, "modelo": familia, "horizonte": tau, "config_id": cid,
                                      "config_json": json.dumps(cfg), "fold": k, "RMSE": rmse,
                                      "SS": 1 - rmse / rmse_pers[k], "n_eval": len(pos_eva),
                                      "tiempo_s": round(time.time() - t0, 2), **info})
                self._liberar()
            self.pred_folds[(familia, tau, cid)] = preds
            filas = [f for f in self.busqueda if f["familia"] == familia and f["horizonte"] == tau
                     and f["config_id"] == cid]
            print(f"  {cid:10s} h={tau} | SS medio {np.nanmean([f['SS'] for f in filas]):+.4f} | RMSE medio "
                  f"{np.nanmean([f['RMSE'] for f in filas]):.2f} cm | {np.nansum([f['tiempo_s'] for f in filas]):.0f} s")
            pd.DataFrame(self.busqueda).to_csv(self.out / "busqueda_resultados.csv", index=False)
            self._guardar_pred_folds(familia, tau)

    def elegir_mejor(self, familia: str, H: de.Horizonte, folds: dict):
        tau = H.tau
        r = pd.DataFrame(self.busqueda)
        r = r[(r["familia"] == familia) & (r["horizonte"] == tau) & r["fold"].isin(list(folds))
              & r["config_id"].isin([cid for cid, _ in self.candidatos[familia]])]
        g = r.groupby("config_id").agg(SS_medio=("SS", "mean"), SS_std=("SS", "std"),
                                       RMSE_medio=("RMSE", "mean"), n_ok=("SS", "count"))
        g = g[g["n_ok"] == len(folds)]
        if g.empty:
            print(f"  {familia} h={tau}: ninguna configuración completó todos los folds")
            return None
        cid = g["SS_medio"].idxmax()
        y_real, y_pred = [], []
        for k, (_, pos_eva) in folds.items():
            pred = self.pred_folds[(familia, tau, cid)][k]
            self.registrar(familia, familia, H, "val_fold", pos_eva, pred, fold=k)
            y_real.append(self.m.a_cm(H.y_nivel[pos_eva]))
            y_pred.append(self.m.a_cm(pred))
        y_real, y_pred = np.concatenate(y_real), np.concatenate(y_pred)
        alto = y_real > self.m.p95_train_cm
        rmse_alto = float(np.sqrt(np.mean((y_pred[alto] - y_real[alto]) ** 2))) if alto.any() else np.nan
        mejor_it = r.loc[r["config_id"] == cid, "mejor_iteracion"] if "mejor_iteracion" in r else pd.Series(dtype=float)
        self.seleccion[(familia, tau)] = {
            "config_id": cid, "cfg": dict(self.candidatos[familia])[cid],
            "SS_medio_folds": float(g.loc[cid, "SS_medio"]),
            "SS_std_folds": float(g.loc[cid, "SS_std"]) if len(folds) > 1 else np.nan,
            "RMSE_medio_folds": float(g.loc[cid, "RMSE_medio"]), "RMSE_alto_folds": rmse_alto,
            "n_alto_folds": int(alto.sum()),
            "mejor_iteracion_media_folds": float(mejor_it.mean()) if mejor_it.notna().any() else np.nan,
            "n_configs": int(len(g))}
        print(f"  >> {familia} h={tau}: mejor {cid} | SS medio {g.loc[cid, 'SS_medio']:+.4f} | RMSE > P95 "
              f"{rmse_alto:.2f} cm")
        return cid

    # -------------------------------------------------------------------------------------------- modelo final
    def entrenar_final(self, familia: str, H: de.Horizonte, datos_red) -> None:
        sel = self.seleccion[(familia, H.tau)]
        pos, part = H.pos, self.m.part[H.pos]
        pos_tv = pos[np.isin(part, ["train", "val"])]
        t0 = time.time()
        predecir, info, objeto, hist = self.ajustar(familia, sel["cfg"], H, pos_tv, datos_red, final=True)
        tiempo = time.time() - t0
        self.historiales[f"{familia}|h{H.tau}"] = {
            "origen": "final: ajuste vs control (último 10 % de train+val)", **hist}
        base = self.dir_modelos / f"{familia}_h{H.tau}"
        if familia == "LGBM":
            objeto.modelo.booster_.save_model(str(base.with_suffix(".txt")))
        elif familia == "CAT":
            objeto.modelo.save_model(str(base.with_suffix(".cbm")))
        else:
            torch.save({"state_dict": objeto.state_dict(), "familia": familia, "cfg": sel["cfg"], "tau": H.tau,
                        "objetivo": H.objetivo, "feat_seq": self.m.feat_seq, "idx_objetivo": self.idx_obj,
                        "lookback": de.LOOKBACK_H}, base.with_suffix(".pt"))
        fila = {"familia": familia, "modelo": familia, "horizonte": H.tau, "config_id": sel["config_id"],
                "parametros": json.dumps(sel["cfg"]), "objetivo": H.objetivo, "tiempo_final_s": round(tiempo, 1)}
        for nombre in ("train", "val", "test"):
            p = pos[part == nombre]
            pred = predecir(p)            # test: única predicción del modelo final
            self.registrar(familia, familia, H, nombre, p, pred)
            if nombre != "test":
                fila[f"RMSE_{nombre}_ajuste_final"] = self.rmse_nivel(H, p, pred)
        fila.update({k: v for k, v in sel.items() if k != "cfg"})
        fila.update({k: v for k, v in info.items() if k != "epocas"})
        self.resumen.append(fila)
        print(f"  Final {familia:5s} h={H.tau} ({sel['config_id']}) | RMSE train {fila['RMSE_train_ajuste_final']:.2f}"
              f" cm | val (en ajuste) {fila['RMSE_val_ajuste_final']:.2f} cm | {tiempo:.0f} s")
        del objeto
        self._liberar()

    # -------------------------------------------------------------------------------------------- ejecución
    def correr(self) -> None:
        a, m = self.a, self.m
        print(f"Fase {a.fase} | objetivo {a.objetivo} | pesos α={a.pesos_alpha} ({a.pesos_umbral}) | redes en "
              f"{self.disp} | folds {a.folds} | familias {a.familias}")
        print(f"Variables: {len(m.feat_tab)} tabulares, {len(m.feat_seq)} por paso | horizontes {m.horizontes} | "
              f"P95 train {m.p95_train_cm:.1f} cm | salida {self.out}")
        entrenar = a.fase in ("humo", "final")
        t_global = time.time()
        for tau in m.horizontes:
            print(f"\n{'=' * 90}\nHORIZONTE τ = {tau} h\n{'=' * 90}")
            H = de.preparar_horizonte(m, tau, a.objetivo, a.pesos_alpha, a.pesos_umbral)
            datos_red = self.datos_red(H) if set(a.familias) & set(eb.FAMILIAS_RED) else None
            folds = {k: de.particion_fold(m, H.pos, k) for k in a.folds}
            rmse_pers = {}
            for k, (pos_ent, pos_eva) in folds.items():
                assert len(pos_eva) > 0, f"El bloque {k} no tiene filas evaluables"
                assert k == 1 or m.ts[pos_ent].max() < m.bordes[k - 1] - pd.Timedelta(hours=m.margen_h)
                assert not (m.part[np.concatenate([pos_ent, pos_eva])] == "test").any()
                rmse_pers[k] = self.rmse_nivel(H, pos_eva, m.pers[pos_eva])
                self.registrar("Persistencia", "Persistencia", H, "val_fold", pos_eva, m.pers[pos_eva], fold=k)
                print(f"  fold {k}: entrena {len(pos_ent):,} filas | evalúa {len(pos_eva):,} | RMSE persistencia "
                      f"{rmse_pers[k]:.2f} cm")
            if entrenar:
                for p in ("train", "val", "test"):
                    sel = H.pos[m.part[H.pos] == p]
                    self.registrar("Persistencia", "Persistencia", H, p, sel, m.pers[sel])
            self.resumen.append({"familia": "Persistencia", "modelo": "Persistencia", "horizonte": tau,
                                 "config_id": "-", "SS_medio_folds": 0.0,
                                 "RMSE_medio_folds": float(np.mean(list(rmse_pers.values())))})
            for familia in a.familias:
                print(f"\n{eb.NOMBRE_LARGO[familia]} — búsqueda ({len(self.candidatos[familia])} configuraciones × "
                      f"{len(folds)} folds)")
                self.buscar(familia, H, folds, rmse_pers, datos_red)
                if self.elegir_mejor(familia, H, folds) is not None and entrenar:
                    self.entrenar_final(familia, H, datos_red)
            del H, datos_red
            self._liberar()
        self.tiempo_total_min = (time.time() - t_global) / 60
        print(f"\nTiempo total: {self.tiempo_total_min:.1f} min")
        self.guardar()

    def seleccion_global(self) -> pd.DataFrame:
        b, filas = pd.DataFrame(self.busqueda), []
        for familia in self.a.familias:
            if not all((familia, t) in self.seleccion for t in self.m.horizontes):
                continue
            ss = [s for t in self.m.horizontes for s in b.loc[
                (b["familia"] == familia) & (b["horizonte"] == t) & b["fold"].isin(self.a.folds)
                & (b["config_id"] == self.seleccion[(familia, t)]["config_id"]), "SS"]]
            filas.append({"familia": familia, "modelo": familia, "SS_global": float(np.mean(ss)),
                          "RMSE_alto_global": float(np.nanmean([self.seleccion[(familia, t)]["RMSE_alto_folds"]
                                                                for t in self.m.horizontes])),
                          **{f"config_h{t}": self.seleccion[(familia, t)]["config_id"] for t in self.m.horizontes}})
        if not filas:
            return pd.DataFrame()
        tabla = pd.DataFrame(filas).sort_values("SS_global", ascending=False).reset_index(drop=True)
        empatados = tabla[tabla["SS_global"] >= tabla["SS_global"].iloc[0] - de.EMPATE_SS]
        ganador = (empatados.loc[empatados["RMSE_alto_global"].idxmin(), "familia"]
                   if len(empatados) > 1 and empatados["RMSE_alto_global"].notna().any() else tabla["familia"].iloc[0])
        tabla["seleccionado"] = tabla["familia"] == ganador
        return tabla

    def guardar(self) -> None:
        pred = pd.concat(self.predicciones, ignore_index=True)
        pred["fold"] = pred["fold"].astype("Int64")
        assert not pred[pred["particion"] == "test"].duplicated(["modelo", "horizonte", "timestamp"]).any()
        assert pred.loc[pred["particion"] == "val_fold", "fold"].notna().all()
        assert pred.loc[pred["particion"] != "val_fold", "fold"].isna().all()
        tabla = self.seleccion_global()
        resumen = pd.DataFrame(self.resumen)
        if not tabla.empty:
            resumen["SS_global"] = resumen["familia"].map(tabla.set_index("familia")["SS_global"])
            resumen["seleccionado"] = resumen["familia"].map(tabla.set_index("familia")["seleccionado"]).fillna(False)
        pred.to_parquet(self.out / "predicciones_yanuncay.parquet", index=False)
        resumen.to_csv(self.out / "resumen_entrenamiento.csv", index=False)
        pd.DataFrame(self.busqueda).to_csv(self.out / "busqueda_resultados.csv", index=False)
        tabla.to_csv(self.out / "seleccion_modelo.csv", index=False)
        with open(self.out / "historiales.json", "w", encoding="utf-8") as f:
            json.dump(self.historiales, f)
        cfg = {"exploratorio": True, "fase": self.a.fase, "firma": self.firma, "target": de.TARGET,
               "horizontes": self.m.horizontes, "margen_h": self.m.margen_h, "lookback_h": de.LOOKBACK_H,
               "seed": de.SEED, "walk_forward": {"n_bloques": de.N_BLOQUES, "folds_evaluados": self.a.folds,
                                                 "bordes_bloques": [str(b) for b in self.m.bordes],
                                                 "fraccion_early_stopping": de.FRACCION_ES},
               "presupuesto": {"n_iter_arboles": self.a.n_config_arboles, "n_iter_redes": self.a.n_config_redes},
               "candidatos": {f: dict(lista) for f, lista in self.candidatos.items()},
               "arbol_comun": {f: eb.comun_arbol(f, self.humo) for f in eb.FAMILIAS_ARBOL},
               "pesos": {"alpha": self.a.pesos_alpha, "umbral": self.a.pesos_umbral,
                         "umbral_cm": self.m.umbral_cm(self.a.pesos_umbral), "columna": self.a.pesos_columna},
               "p95_train_cm": self.m.p95_train_cm, "empate_ss": de.EMPATE_SS,
               "formulacion": str(self.a.formulacion) if self.a.formulacion else None,
               "escala_objetivo": {
                   "min_cm": self.m.tgt_min, "max_cm": self.m.tgt_max,
                   "reconstruccion_nivel": "nivel_cm(t+τ) = (x[TARGET](t) + salida) * (max - min) + min"
                                           if self.a.objetivo == "delta" else "nivel_cm = salida * (max - min) + min",
                   "arboles": "aprenden el objetivo en la escala min-max oficial (Δ = y_obs(t+τ) − x[TARGET](t))",
                   "redes": "aprenden el objetivo estandarizado (media y desviación de las filas de ajuste, guardadas "
                            "como buffers); forward devuelve la escala min-max oficial antes de reconstruir el nivel"},
               "seleccion_por_familia": {f"{f}|h{t}": s for (f, t), s in self.seleccion.items()},
               "features_tabulares": self.m.feat_tab, "features_secuencia": self.m.feat_seq,
               "dispositivo_redes": str(self.disp), "tiempo_total_min": round(self.tiempo_total_min, 2),
               "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "torch": torch.__version__}}
        with open(self.out / "configuracion.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
        print(f"Predicciones: {len(pred):,} filas | modelos {sorted(pred['modelo'].unique())} -> {self.out}")


def main(argv=None) -> None:
    for flujo in (sys.stdout, sys.stderr):    # consola de Windows (cp1252): τ, α, Δ
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")
    a = parse_args(argv)
    np.random.seed(de.SEED)
    torch.manual_seed(de.SEED)
    Ejecucion(a).correr()


if __name__ == "__main__":
    main()
