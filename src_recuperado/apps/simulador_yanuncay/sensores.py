"""Simulador de sensores hidrometeorológicos de la cuenca del Yanuncay.

Reproduce hora a hora el registro real (por defecto el periodo de prueba) como si los datos llegaran
de las estaciones en tiempo real, y permite inyectar escenarios:

- ``Tormenta``: hietograma sintético (mm totales y duración configurables) sumado a los pluviómetros.
- ``FallaSensor``: el sensor deja de transmitir (NaN) durante un intervalo.
- ``ValorCongelado``: el sensor repite el mismo valor durante un intervalo.

Fuente de datos: particiones train/val/test de la corrida preliminar, **desescaladas** con los
escaladores de train. Los valores que el preprocesamiento había imputado (``__imp = 1``) se vuelven
NaN, de modo que el simulador entrega solo lo que el sensor midió realmente (tras el control de
calidad). La imputación en tiempo real la hace ``pronosticador.py`` de forma causal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from configuracion import columnas_entrada, ruta


# --------------------------------------------------------------------------------------------
# Datos observados
# --------------------------------------------------------------------------------------------
def cargar_datos_observados(config: dict[str, Any]) -> pd.DataFrame:
    """Malla horaria (UTC) con las lecturas observadas en unidades físicas y la columna ``particion``."""
    carpeta = ruta(config, "datos_dir")
    patron = config["rutas"]["prefijo_particiones"]
    esc = pd.read_csv(ruta(config, "escaladores")).set_index("columna")
    cols = columnas_entrada(config)
    objetivo = config["objetivo"]
    partes = []
    for particion in ("train", "val", "test"):
        d = pd.read_parquet(carpeta / patron.format(particion=particion))
        salida = pd.DataFrame(index=d.index)
        for col in cols:
            mn, mx = float(esc.loc[col, "min"]), float(esc.loc[col, "max"])
            valor = d[col].astype(float) * (mx - mn) + mn
            imp = f"{col}__imp"
            if imp in d.columns:
                valor = valor.mask(d[imp] == 1)
            salida[col] = valor
        if "y_obs" in d.columns:  # nivel observado real, nunca imputado
            mn, mx = float(esc.loc["y_obs", "min"]), float(esc.loc["y_obs", "max"])
            salida[objetivo] = d["y_obs"].astype(float) * (mx - mn) + mn
        salida["particion"] = particion
        partes.append(salida)
    datos = pd.concat(partes).sort_index()
    datos = datos[~datos.index.duplicated()]
    malla = pd.date_range(datos.index.min(), datos.index.max(), freq="h", name="timestamp")
    datos = datos.reindex(malla)
    datos["particion"] = datos["particion"].fillna("fuera")
    return datos


# --------------------------------------------------------------------------------------------
# Escenarios
# --------------------------------------------------------------------------------------------
@dataclass
class Escenario:
    inicio: pd.Timestamp
    duracion_h: int
    columnas: list[str]

    @property
    def fin(self) -> pd.Timestamp:
        return self.inicio + pd.Timedelta(hours=self.duracion_h)

    def activo(self, ts: pd.Timestamp) -> bool:
        return self.inicio <= ts < self.fin

    def aplicar(self, ts: pd.Timestamp, fila: pd.Series, datos: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def describir(self) -> str:
        raise NotImplementedError


@dataclass
class Tormenta(Escenario):
    total_mm: float = 40.0
    forma: str = "triangular"

    def hietograma(self) -> np.ndarray:
        """Lámina horaria (mm) que suma ``total_mm``; triangular con pico en el primer tercio."""
        n = max(int(self.duracion_h), 1)
        if self.forma == "uniforme" or n < 3:
            pesos = np.ones(n)
        else:
            pico = max(int(round(n / 3)) - 1, 0)
            pesos = np.array([(i + 1) / (pico + 1) if i <= pico else (n - i) / (n - pico) for i in range(n)])
        return self.total_mm * pesos / pesos.sum()

    def aplicar(self, ts: pd.Timestamp, fila: pd.Series, datos: pd.DataFrame) -> pd.Series:
        k = int((ts - self.inicio) / pd.Timedelta(hours=1))
        lamina = float(self.hietograma()[k])
        for col in self.columnas:
            base = fila[col]
            fila[col] = (0.0 if pd.isna(base) else float(base)) + lamina
        return fila

    def describir(self) -> str:
        return f"Tormenta {self.total_mm:.0f} mm / {self.duracion_h} h ({self.forma})"


@dataclass
class FallaSensor(Escenario):
    def aplicar(self, ts: pd.Timestamp, fila: pd.Series, datos: pd.DataFrame) -> pd.Series:
        for col in self.columnas:
            fila[col] = np.nan
        return fila

    def describir(self) -> str:
        return f"Falla de sensor ({self.duracion_h} h)"


@dataclass
class ValorCongelado(Escenario):
    valores: dict[str, float] = field(default_factory=dict)

    def aplicar(self, ts: pd.Timestamp, fila: pd.Series, datos: pd.DataFrame) -> pd.Series:
        for col in self.columnas:
            if col not in self.valores:
                previos = datos.loc[:self.inicio, col].dropna()
                self.valores[col] = float(previos.iloc[-1]) if len(previos) else np.nan
            fila[col] = self.valores[col]
        return fila

    def describir(self) -> str:
        return f"Valor congelado ({self.duracion_h} h)"


# --------------------------------------------------------------------------------------------
# Simulador
# --------------------------------------------------------------------------------------------
class SimuladorSensores:
    """Entrega una lectura por hora a partir de ``inicio`` aplicando los escenarios activos."""

    def __init__(self, datos: pd.DataFrame, inicio: pd.Timestamp, columnas: list[str]) -> None:
        self.datos = datos
        self.columnas = list(columnas)
        self.escenarios: list[Escenario] = []
        self.t_siguiente = pd.Timestamp(inicio).floor("h")
        if self.t_siguiente.tzinfo is None:
            self.t_siguiente = self.t_siguiente.tz_localize("UTC")

    @property
    def t_ultimo(self) -> pd.Timestamp:
        return self.t_siguiente - pd.Timedelta(hours=1)

    @property
    def agotado(self) -> bool:
        return self.t_siguiente > self.datos.index.max()

    def agregar_escenario(self, escenario: Escenario) -> None:
        self.escenarios.append(escenario)

    def limpiar_escenarios(self) -> None:
        self.escenarios.clear()

    def lectura(self, ts: pd.Timestamp) -> pd.Series:
        """Lectura de todas las estaciones en ``ts`` (con escenarios). Determinista."""
        if ts in self.datos.index:
            fila = self.datos.loc[ts, self.columnas].astype(float).copy()
        else:
            fila = pd.Series(np.nan, index=self.columnas, dtype=float)
        for esc in self.escenarios:
            if esc.activo(ts):
                fila = esc.aplicar(ts, fila, self.datos)
        fila.name = ts
        return fila

    def historia(self, n_horas: int) -> pd.DataFrame:
        """Lecturas de las ``n_horas`` anteriores al instante siguiente (precarga del sistema)."""
        indice = pd.date_range(end=self.t_ultimo, periods=n_horas, freq="h")
        return pd.DataFrame([self.lectura(ts) for ts in indice], index=indice)

    def siguiente(self) -> pd.Series:
        """Emite la lectura de la próxima hora y avanza el reloj simulado."""
        fila = self.lectura(self.t_siguiente)
        self.t_siguiente += pd.Timedelta(hours=1)
        return fila
