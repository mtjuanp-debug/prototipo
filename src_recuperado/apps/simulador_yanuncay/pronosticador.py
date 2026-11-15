"""Pronosticador en tiempo real: ventana móvil, imputación causal, pronóstico +τ1/+τ2 y alertas.

Flujo por cada lectura horaria:
1. Se agrega la lectura cruda a la historia (se conservan ≥ 72 h + ventana de las redes).
2. Control de calidad e imputación causal de la ventana (``imputacion_causal``).
3. Variables con ``modelo_adaptador.calcular_variables`` y pronóstico con ``modelo_adaptador.predecir``
   (el adaptador escala con los escaladores de train y devuelve NIVEL en cm).
4. Nivel -> caudal con la curva de descarga -> estado (normal / prealerta / alerta).
5. Registro de alertas: escalamientos del estado pronosticado y observado, con la anticipación
   prevista (τ) y la real (hora observada del cruce − hora de emisión).
6. Top de modelos: además del modelo principal (que emite las alertas) se pueden correr otros modelos
   en paralelo. Cada pronóstico lleva su banda de incertidumbre y su probabilidad de prealerta/alerta
   (``confianza``), y por horizonte se calcula el consenso entre los modelos del top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import modelo_adaptador as adaptador
from configuracion import columnas_entrada
from confianza import CalibracionModelo, consenso
from hidrologia import SIN_DATO, caudal_desde_nivel, estado_desde_caudal, rango_estado, umbrales_cm
from imputacion_causal import ParametrosImputacion, ajustar_parametros, imputar

HORAS_FIN_EVENTO = 6  # horas seguidas en normal para cerrar un evento y rearmar las alertas
TOLERANCIA_CONFIRMACION_H = 3  # margen tras t+τ para considerar confirmada una alerta pronosticada


@dataclass
class ResultadoPaso:
    ts: pd.Timestamp
    nivel_obs: float
    caudal_obs: float
    estado_obs: str
    nivel_entrada: float
    pronosticos: dict[int, dict[str, Any]]
    imputadas: list[str]
    valido: bool
    mensaje: str = ""
    alertas_nuevas: list[dict[str, Any]] = field(default_factory=list)
    top: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)  # {h: {modelo: pronóstico}}
    consenso: dict[int, dict[str, Any]] = field(default_factory=dict)


class Pronosticador:
    """Mantiene la ventana de datos y emite pronósticos y estados de alerta hora a hora."""

    def __init__(self, config: dict[str, Any], datos_train: pd.DataFrame, familia: str | None = None,
                 parametros: ParametrosImputacion | None = None) -> None:
        self.config = config
        self.columnas = columnas_entrada(config)
        self.objetivo = config["objetivo"]
        self.contrato = adaptador.cargar_contrato(config)
        cfg_var = adaptador.config_variables(self.contrato)
        faltan = [c for c in (cfg_var.entradas if cfg_var is not None else []) if c not in self.columnas]
        if faltan:
            raise ValueError(f"Los sensores simulados no entregan estaciones que piden los modelos: {faltan}")
        self.horizontes = [int(h) for h in self.contrato.horizontes]
        self.ventana_h = max(int(config.get("ventana_historia_h", 96)), self.contrato.ventana_minima_h)
        self.margen_h = 24 + int(config["imputacion"]["ffill_nivel_max_h"])
        self.params = parametros or ajustar_parametros(datos_train[self.columnas], config)
        self.curva = config["curva_descarga"]
        self.umbrales = config["umbrales_m3s"]
        self.crudo = pd.DataFrame(columns=self.columnas, dtype=float)
        self.registro: list[dict[str, Any]] = []
        self.alertas: list[dict[str, Any]] = []
        self._armado: dict[str, int] = {}
        self._normales: dict[str, int] = {}
        self.umbrales_cm = umbrales_cm(config)
        self.calibracion: dict[tuple[str, int], CalibracionModelo] = {}
        self.pesos: dict[str, float] = {}
        self._cache: dict[str, dict[int, adaptador.ModeloCargado]] = {}
        self.familia = ""
        self.modelos: dict[int, adaptador.ModeloCargado] = {}
        self.top: list[str] = []
        self.cambiar_familia(familia or config.get("familia_defecto", "XGB"))

    # ------------------------------------------------------------------ configuración
    def _modelos_de(self, familia: str) -> dict[int, adaptador.ModeloCargado]:
        if familia not in self._cache:
            self._cache[familia] = {h: adaptador.cargar_modelo(self.config["rio"], familia, h, self.config)
                                    for h in self.horizontes}
        return self._cache[familia]

    def cambiar_familia(self, familia: str) -> None:
        self.modelos = self._modelos_de(familia)
        self.familia = familia
        if self.familia not in self.top:
            self.top = [self.familia] + [f for f in self.top if f != self.familia]

    def configurar_top(self, familias: list[str], calibracion: dict[tuple[str, int], CalibracionModelo] | None = None,
                       pesos: dict[str, float] | None = None) -> None:
        """Modelos que se corren en cada hora; el primero es el principal (emite las alertas)."""
        if not familias:
            return
        if calibracion is not None:
            self.calibracion = calibracion
        if pesos is not None:
            self.pesos = pesos
        for f in familias:
            self._modelos_de(f)
        self.top = list(dict.fromkeys(familias))
        if self.top[0] != self.familia:
            self.cambiar_familia(self.top[0])

    def precargar(self, historia: pd.DataFrame) -> None:
        """Carga la historia previa (simula que el sistema ya venía operando)."""
        self.crudo = historia[self.columnas].astype(float).copy()

    # ------------------------------------------------------------------ paso
    def _estado(self, nivel: float) -> tuple[float, str]:
        if nivel is None or not math.isfinite(nivel):
            return math.nan, SIN_DATO
        q = float(caudal_desde_nivel(nivel, self.curva))
        return q, estado_desde_caudal(q, self.umbrales)

    def recibir(self, lectura: pd.Series) -> ResultadoPaso:
        """Procesa la lectura de la hora ``lectura.name`` y devuelve el resultado del paso."""
        ts = pd.Timestamp(lectura.name)
        fila = lectura.reindex(self.columnas).astype(float).to_frame().T
        fila.index = pd.DatetimeIndex([ts])
        self.crudo = pd.concat([self.crudo[self.crudo.index < ts], fila]) if len(self.crudo) else fila
        self.crudo = self.crudo.iloc[-(self.ventana_h + self.margen_h):]
        malla = pd.date_range(end=ts, periods=len(self.crudo), freq="h")
        ventana = self.crudo.reindex(malla)
        return self._procesar(ts, ventana)

    def _procesar(self, ts: pd.Timestamp, ventana: pd.DataFrame) -> ResultadoPaso:
        imputado, indicadores = imputar(ventana, self.params, self.config)
        sub = pd.concat([imputado, indicadores], axis=1).iloc[-self.ventana_h:]
        nivel_obs = float(ventana[self.objetivo].iloc[-1]) if pd.notna(ventana[self.objetivo].iloc[-1]) else math.nan
        caudal_obs, estado_obs = self._estado(nivel_obs)
        nivel_ent = float(imputado[self.objetivo].iloc[-1]) if pd.notna(imputado[self.objetivo].iloc[-1]) else math.nan
        imputadas = [c[:-5] for c in indicadores.columns if indicadores[c].iloc[-1] == 1]
        pronosticos: dict[int, dict[str, Any]] = {}
        mensaje = ""
        if len(sub) < self.ventana_h:
            mensaje = "Historia insuficiente para calcular variables."
        variables = adaptador.calcular_variables(sub, self.contrato) if not mensaje else None
        top: dict[int, dict[str, dict[str, Any]]] = {h: {} for h in self.horizontes}
        for fam in self.top:
            for h, modelo in self._modelos_de(fam).items():
                top[h][fam] = self._pronostico(fam, h, modelo, variables, nivel_ent, ts)
        for h in self.horizontes:
            pronosticos[h] = top[h][self.familia]
        valido = all(math.isfinite(p["nivel_cm"]) for p in pronosticos.values())
        if not valido and not mensaje:
            mensaje = "Sin pronóstico: hueco de datos no imputable de forma causal (p. ej. nivel > límite)."
        res = ResultadoPaso(ts, nivel_obs, caudal_obs, estado_obs, nivel_ent, pronosticos, imputadas, valido,
                            mensaje)
        res.top = top
        res.consenso = {h: consenso(top[h], self.pesos) for h in self.horizontes}
        for h, c in res.consenso.items():
            if c:
                c["caudal_m3s"], c["estado"] = self._estado(c["nivel_cm"])
        res.alertas_nuevas = self._actualizar_alertas(res)
        self._registrar(res)
        return res

    def _pronostico(self, familia: str, h: int, modelo: adaptador.ModeloCargado, variables: pd.DataFrame | None,
                    nivel_ent: float, ts: pd.Timestamp) -> dict[str, Any]:
        """Pronóstico de un modelo y horizonte con su banda y probabilidades de superar los umbrales."""
        nivel_p = math.nan
        if variables is not None and math.isfinite(nivel_ent):
            nivel_p = adaptador.predecir(modelo, variables, nivel_ent)
        q_p, est_p = self._estado(nivel_p)
        salida: dict[str, Any] = {"nivel_cm": nivel_p, "caudal_m3s": q_p, "estado": est_p,
                                  "t_objetivo": ts + pd.Timedelta(hours=h)}
        cal = self.calibracion.get((familia, h))
        if cal is not None and math.isfinite(nivel_p):
            salida.update(cal.evaluar(nivel_p, self.umbrales_cm))
        return salida

    # ------------------------------------------------------------------ alertas
    def _escalamiento(self, clave: str, estado: str) -> bool:
        """True si ``estado`` supera el máximo ya alertado en el evento en curso de ``clave``."""
        r = rango_estado(estado)
        if r <= 0:
            self._normales[clave] = self._normales.get(clave, 0) + (1 if r == 0 else 0)
            if self._normales[clave] >= HORAS_FIN_EVENTO:
                self._armado[clave] = 0
            return False
        self._normales[clave] = 0
        if r > self._armado.get(clave, 0):
            self._armado[clave] = r
            return True
        return False

    def _actualizar_alertas(self, res: ResultadoPaso) -> list[dict[str, Any]]:
        nuevas: list[dict[str, Any]] = []
        for h, p in res.pronosticos.items():
            if self._escalamiento(f"h{h}", p["estado"]):
                nuevas.append({"emitida": res.ts, "origen": f"pronóstico +{h} h", "estado": p["estado"],
                               "nivel_cm": round(p["nivel_cm"], 1), "caudal_m3s": round(p["caudal_m3s"], 1),
                               "t_objetivo": p["t_objetivo"], "anticipacion_prevista_h": h,
                               "confirmada": None, "anticipacion_real_h": None, "modelo": self.familia})
        if self._escalamiento("obs", res.estado_obs):
            nuevas.append({"emitida": res.ts, "origen": "observado", "estado": res.estado_obs,
                           "nivel_cm": round(res.nivel_obs, 1), "caudal_m3s": round(res.caudal_obs, 1),
                           "t_objetivo": res.ts, "anticipacion_prevista_h": 0, "confirmada": res.ts,
                           "anticipacion_real_h": 0.0, "resultado": "observada", "modelo": "-"})
        for a in nuevas:
            a.setdefault("resultado", "pendiente")
        self.alertas.extend(nuevas)
        self._verificar_pendientes(res)
        return nuevas

    def _verificar_pendientes(self, res: ResultadoPaso) -> None:
        """Una alerta pronosticada se confirma si el nivel observado alcanza su estado entre la emisión y
        ``t_objetivo + TOLERANCIA_CONFIRMACION_H``; si no, se marca como falsa alarma."""
        r_obs = rango_estado(res.estado_obs)
        for a in self.alertas:
            if a["resultado"] != "pendiente":
                continue
            limite = a["t_objetivo"] + pd.Timedelta(hours=TOLERANCIA_CONFIRMACION_H)
            if r_obs >= rango_estado(a["estado"]) and res.ts <= limite:
                a["confirmada"] = res.ts
                a["anticipacion_real_h"] = (res.ts - a["emitida"]) / pd.Timedelta(hours=1)
                a["resultado"] = "confirmada"
            elif res.ts > limite:
                a["resultado"] = "falsa alarma"

    # ------------------------------------------------------------------ registro
    def _registrar(self, res: ResultadoPaso) -> None:
        fila: dict[str, Any] = {"ts": res.ts, "nivel_obs": res.nivel_obs, "caudal_obs": res.caudal_obs,
                                "estado_obs": res.estado_obs, "nivel_entrada": res.nivel_entrada,
                                "n_imputadas": len(res.imputadas)}
        for h, p in res.pronosticos.items():
            fila[f"nivel_h{h}"] = p["nivel_cm"]
            fila[f"caudal_h{h}"] = p["caudal_m3s"]
            fila[f"estado_h{h}"] = p["estado"]
            for k in ("inf90", "sup90", "inf80", "sup80", "p_prealerta", "p_alerta"):
                fila[f"{k}_h{h}"] = p.get(k, math.nan)
            for fam, pf in res.top.get(h, {}).items():
                if fam != self.familia:
                    fila[f"nivel_h{h}_{fam}"] = pf["nivel_cm"]
        fila.update({c: self.crudo[c].iloc[-1] for c in self.columnas})
        self.registro.append(fila)

    def tabla_registro(self) -> pd.DataFrame:
        return pd.DataFrame(self.registro).set_index("ts") if self.registro else pd.DataFrame()

    def tabla_alertas(self) -> pd.DataFrame:
        return pd.DataFrame(self.alertas)
