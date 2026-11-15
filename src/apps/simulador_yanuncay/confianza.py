"""Nivel de confianza de los pronósticos y ranking de modelos por desempeño en crecidas.

El modelo predice un número (nivel en cm), no una clase, así que no hay un "accuracy". La confianza se
mide con los errores reales que cada modelo cometió en validación, sobre predicciones fuera de muestra
(``particion == 'val_fold'``: cada fold del walk-forward se pronostica con un modelo que no lo vio).

1. Banda de incertidumbre (conformal empírico, condicionado al nivel pronosticado): para un pronóstico
   ŷ se toman los residuos r = y − ŷ de validación con ŷ en el mismo tramo de nivel. La banda al 80 % o
   90 % es [ŷ + q_inf(r), ŷ + q_sup(r)]. Los tramos altos tienen más error y la banda se ensancha.
2. Probabilidad de prealerta y alerta: proporción de los ŷ + r_i (residuos del mismo tramo) que superan
   el umbral en cm, que es equivalente a superar el umbral de caudal porque la curva es creciente.
3. Puntaje de crecidas para el ranking (también en validación):
   - POD_eventos: fracción de eventos de prealerta observados en validación que el modelo anticipó, es
     decir, que pronosticó la prealerta al menos 1 h antes del cruce observado;
   - FAR: fracción de las horas con prealerta pronosticada en las que no se observó prealerta en t+τ;
   - puntaje = POD_eventos · (1 − FAR), promediado entre horizontes. Si dos puntajes difieren en menos de
     0,01, gana el de menor RMSE en niveles altos (≥ P95 de train), la misma regla que la tesis.
   La persistencia no anticipa ningún evento por construcción, así que su puntaje es 0.

Limitación: los residuos vienen de los modelos de cada fold, que se entrenaron con menos datos que el
modelo final (train + val). Por eso las bandas tienden a ser algo más anchas que el error real, lo que
las hace conservadoras.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_RESIDUOS_TRAMO = 40  # si un tramo tiene menos residuos, se fusiona con el tramo inferior
MAX_RESIDUOS_GUARDADOS = 4000
NIVELES_BANDA = (0.80, 0.90)
EMPATE = 0.01  # diferencia de puntaje considerada empate (misma regla que la selección de la tesis)


@dataclass
class CalibracionModelo:
    """Residuos de validación por tramo de nivel pronosticado para un modelo y un horizonte."""
    modelo: str
    horizonte: int
    cortes_cm: list[float]  # límites inferiores de cada tramo (el primero es -inf)
    residuos: list[np.ndarray]
    metricas: dict[str, float] = field(default_factory=dict)

    def _tramo(self, nivel_cm: float) -> int:
        return max(i for i, c in enumerate(self.cortes_cm) if nivel_cm >= c)

    def evaluar(self, nivel_cm: float, umbrales_cm: dict[str, float]) -> dict[str, float]:
        """Banda al 80/90 % y probabilidad de superar cada umbral para un pronóstico ``nivel_cm``."""
        if nivel_cm is None or not math.isfinite(nivel_cm):
            return {}
        r = self.residuos[self._tramo(nivel_cm)]
        salida: dict[str, float] = {"n_residuos": float(len(r))}
        for nivel in NIVELES_BANDA:
            a = (1 - nivel) / 2
            salida[f"inf{int(nivel * 100)}"] = nivel_cm + float(np.quantile(r, a))
            salida[f"sup{int(nivel * 100)}"] = nivel_cm + float(np.quantile(r, 1 - a))
        muestras = nivel_cm + r
        for nombre, umbral in umbrales_cm.items():
            salida[f"p_{nombre}"] = float(np.mean(muestras >= umbral))
        return salida


def _episodios(mask: np.ndarray) -> list[int]:
    """Índices de inicio de cada racha de True."""
    m = np.asarray(mask, dtype=bool)
    previo = np.concatenate([[False], m[:-1]])
    return list(np.flatnonzero(m & ~previo))


def _metricas_crecida(df: pd.DataFrame, h: int, umbral_cm: float, p95_cm: float) -> dict[str, float]:
    """POD de eventos anticipados, FAR por horas, RMSE en niveles altos y SS para una serie val_fold.

    ``df`` está indexado por la hora de emisión t, con y_real = nivel(t+h) e y_pred = pronóstico de t+h.
    """
    df = df.sort_index()
    malla = pd.date_range(df.index.min(), df.index.max() + pd.Timedelta(hours=h), freq="h")
    obs = pd.Series(np.nan, index=malla)
    obs.loc[df.index + pd.Timedelta(hours=h)] = df["y_real_cm"].to_numpy()
    pred_emision = df["y_pred_cm"].reindex(malla)
    detectados, eventos, anticip = 0, 0, []
    for i in _episodios((obs >= umbral_cm).to_numpy()):
        t0 = malla[i]
        if i == 0 or not np.isfinite(obs.iloc[i - 1]):
            continue  # el cruce no es evaluable: no se sabe si empezó antes del hueco
        eventos += 1
        # emisiones entre t0 − h y t0 − 1 h: su objetivo cae antes o en el cruce
        ventana = pred_emision.loc[t0 - pd.Timedelta(hours=h): t0 - pd.Timedelta(hours=1)]
        aciertos = ventana[ventana >= umbral_cm]
        if len(aciertos):
            detectados += 1
            anticip.append((t0 - aciertos.index[0]) / pd.Timedelta(hours=1))
    alarmas = df["y_pred_cm"] >= umbral_cm
    far = float((df.loc[alarmas, "y_real_cm"] < umbral_cm).mean()) if alarmas.any() else 0.0
    alto = df["y_real_cm"] >= p95_cm
    err = df["y_real_cm"] - df["y_pred_cm"]
    return {"eventos": float(eventos), "detectados": float(detectados),
            "pod_eventos": detectados / eventos if eventos else math.nan,
            "far": far, "anticipacion_h": float(np.mean(anticip)) if anticip else math.nan,
            "rmse_alto": float(np.sqrt(np.mean(err[alto] ** 2))) if alto.any() else math.nan,
            "rmse": float(np.sqrt(np.mean(err ** 2)))}


def _cortes(niveles: list[float], pred: np.ndarray) -> list[float]:
    """Tramos [-inf, n1), [n1, n2), ...; se fusionan los que tengan menos de MIN_RESIDUOS_TRAMO."""
    cortes = [-math.inf] + sorted(niveles)
    while len(cortes) > 1:
        conteos = [np.sum((pred >= c) & (pred < (cortes[i + 1] if i + 1 < len(cortes) else math.inf)))
                   for i, c in enumerate(cortes)]
        pobres = [i for i, n in enumerate(conteos) if n < MIN_RESIDUOS_TRAMO and i > 0]
        if not pobres:
            break
        cortes.pop(pobres[-1])
    return cortes


def calibrar(predicciones: Path | pd.DataFrame, umbrales_cm: dict[str, float], p95_cm: float,
             modelos: list[str] | None = None, semilla: int = 42) -> dict[tuple[str, int], CalibracionModelo]:
    """Calibra banda, probabilidades y métricas de crecida con las predicciones ``val_fold``."""
    p = predicciones if isinstance(predicciones, pd.DataFrame) else pd.read_parquet(predicciones)
    p = p[(p["particion"] == "val_fold") & p["y_real_cm"].notna() & p["y_pred_cm"].notna()]
    if modelos:
        p = p[p["modelo"].isin(modelos)]
    rng = np.random.default_rng(semilla)
    niveles = [p95_cm, umbrales_cm["prealerta"]]
    salida: dict[tuple[str, int], CalibracionModelo] = {}
    for (modelo, h), g in p.groupby(["modelo", "horizonte"]):
        g = g.set_index("timestamp")
        pred = g["y_pred_cm"].to_numpy()
        res = (g["y_real_cm"] - g["y_pred_cm"]).to_numpy()
        cortes = _cortes(niveles, pred)
        residuos = []
        for i, c in enumerate(cortes):
            sup = cortes[i + 1] if i + 1 < len(cortes) else math.inf
            r = res[(pred >= c) & (pred < sup)]
            if len(r) > MAX_RESIDUOS_GUARDADOS:
                r = rng.choice(r, MAX_RESIDUOS_GUARDADOS, replace=False)
            residuos.append(np.sort(r))
        metr = _metricas_crecida(g, int(h), umbrales_cm["prealerta"], p95_cm)
        metr["puntaje"] = (0.0 if not math.isfinite(metr["pod_eventos"])
                           else metr["pod_eventos"] * (1 - metr["far"]))
        salida[(str(modelo), int(h))] = CalibracionModelo(str(modelo), int(h), cortes, residuos, metr)
    return salida


def ranking(calibracion: dict[tuple[str, int], CalibracionModelo], horizontes: list[int],
            familias: list[str] | None = None) -> pd.DataFrame:
    """Tabla ordenada por puntaje de crecidas (media de horizontes); desempate por menor RMSE alto."""
    filas = []
    modelos = sorted({m for m, _ in calibracion})
    for m in modelos:
        if familias and m not in familias:
            continue
        mh = [calibracion[(m, h)].metricas for h in horizontes if (m, h) in calibracion]
        if not mh:
            continue
        fila: dict[str, Any] = {"modelo": m,
                                "puntaje": float(np.mean([x["puntaje"] for x in mh])),
                                "rmse_alto": float(np.nanmean([x["rmse_alto"] for x in mh]))}
        for h in horizontes:
            x = calibracion.get((m, h))
            if x is None:
                continue
            fila[f"eventos_h{h}"] = f"{int(x.metricas['detectados'])}/{int(x.metricas['eventos'])}"
            fila[f"far_h{h}"] = x.metricas["far"]
            fila[f"anticip_h{h}"] = x.metricas["anticipacion_h"]
            fila[f"puntaje_h{h}"] = x.metricas["puntaje"]
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    # misma regla de desempate de la tesis: entre los que quedan a < EMPATE del mejor, gana el de menor RMSE alto
    pendientes, orden = tabla.copy(), []
    while len(pendientes):
        mejor = pendientes["puntaje"].max()
        cand = pendientes[pendientes["puntaje"] >= mejor - EMPATE]
        elegido = cand.sort_values(["rmse_alto", "puntaje"], ascending=[True, False]).index[0]
        orden.append(elegido)
        pendientes = pendientes.drop(index=elegido)
    tabla = tabla.loc[orden].reset_index(drop=True)
    tabla.insert(0, "rango", np.arange(1, len(tabla) + 1))
    return tabla


def consenso(pronosticos: dict[str, dict[str, Any]], pesos: dict[str, float] | None = None) -> dict[str, Any]:
    """Consenso de un horizonte entre varios modelos: votos por estado y nivel combinado.

    ``pronosticos``: ``{modelo: {"nivel_cm", "estado", ...}}``. El nivel combinado es la mediana ponderada
    por puntaje de crecidas (o la mediana simple si no hay pesos positivos).
    """
    validos = {m: p for m, p in pronosticos.items() if p.get("nivel_cm") is not None
               and math.isfinite(p["nivel_cm"])}
    if not validos:
        return {}
    votos: dict[str, int] = {}
    for p in validos.values():
        votos[p["estado"]] = votos.get(p["estado"], 0) + 1
    niveles = np.array([p["nivel_cm"] for p in validos.values()])
    w = np.array([max((pesos or {}).get(m, 0.0), 0.0) for m in validos])
    if w.sum() <= 0:
        combinado = float(np.median(niveles))
    else:
        orden = np.argsort(niveles)
        acum = np.cumsum(w[orden]) / w.sum()
        combinado = float(niveles[orden][np.searchsorted(acum, 0.5)])
    return {"votos": votos, "n": len(validos), "nivel_cm": combinado,
            "min_cm": float(niveles.min()), "max_cm": float(niveles.max())}
