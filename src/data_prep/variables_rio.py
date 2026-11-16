# -*- coding: utf-8 -*-
"""Ingeniería de variables de los modelos de nivel de los cuatro ríos (Yanuncay, Tomebamba, Tarqui, Machángara).

Módulo único y genérico, importable desde los notebooks de preprocesamiento (``preprocesamiento_<slug>.ipynb``) y
desde la aplicación de pronóstico en tiempo real. Nada depende del río: el objetivo y las entradas llegan en
``ConfigVariables`` (normalmente desde ``horizontes_<slug>.json``). ``variables_yanuncay.py`` reexporta este módulo
con los valores por defecto del Yanuncay para la aplicación ``src/apps/simulador_yanuncay``.

Todas las variables son **causales**: la variable del instante t solo usa datos de t o anteriores. Si una ventana
contiene un NaN, la variable resultante es NaN (regla de 72 h: los huecos largos no se imputan y las filas que los
tocan quedan como no válidas).

Uso típico en tiempo real::

    import pandas as pd
    import variables_rio as vy

    cfg = vy.ConfigVariables.desde_horizontes_json("Docs/resources/horizontes_<slug>.json")
    X_cm = vy.calcular_variables(df_ventana, cfg)          # unidades físicas, índice horario continuo
    esc = vy.leer_escaladores("Docs/resources/escaladores_<slug>.csv")
    X = vy.escalar(X_cm, esc)                               # Min-Max ajustado en train
    fila_t = X.iloc[[-1]]                                   # variables del último instante de la ventana

``df_ventana`` debe tener índice horario continuo (sin horas faltantes) y, como mínimo, las columnas de
``cfg.entradas`` ya imputadas (huecos ≤ 72 h). Para obtener variables completas en la última hora hacen falta
``cfg.ventana_requerida_h`` horas de historia.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Fila de escaladores_<slug>.csv con la escala del objetivo (y_obs comparte min y max con la columna objetivo)
ESCALA_OBJETIVO = "y_obs"
NOMBRE_AREAL = "AREAL_Media__Lluvia_Tot"
SUFIJO_IMP = "__imp"
COLUMNAS_TIEMPO = ("hora_sin", "hora_cos", "mes_sin", "mes_cos")

# Conjunto al que pertenece cada tipo de variable. "base" reproduce las variables de la corrida preliminar;
# "lluvia_extendida" agrupa las variables de lluvia añadidas en la fase A (rezagos, acumulados cortos y de 12 h,
# máximos horarios, índice de precipitación antecedente y lluvia areal).
CONJUNTO_BASE = "base"
CONJUNTO_EXTENDIDO = "lluvia_extendida"


# ---------------------------------------------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigVariables:
    """Parámetros de construcción de variables.

    Attributes:
        entradas: columnas base (unidades físicas) en el orden de salida; la primera suele ser el objetivo.
        target: columna del nivel objetivo (cm); None = la primera de ``entradas``.
        ventanas_lluvia_base_h: acumulados móviles de la corrida preliminar.
        ventanas_lluvia_extra_h: acumulados móviles añadidos (el de 1 h es la propia lluvia horaria y se omite).
        maximos_lluvia_h: ventanas del máximo de la lluvia horaria.
        lags_lluvia_h: rezagos de la lluvia horaria.
        lags_lluvia_por_estacion: si es True los rezagos se calculan por pluviómetro además de la lluvia areal.
        api_k: constante de recesión horaria del índice de precipitación antecedente (API).
        api_ventana_h: horas que suma el API (se trunca para respetar la regla de 72 h; 0,9^72 ≈ 5e-4).
        lags_nivel_h: rezagos del nivel.
        deltas_nivel_h: cambios del nivel.
        estaciones_areales: pluviómetros promediados en la lluvia areal (None = todos los de ``entradas``).
        variables_extendidas: si es False solo se generan las variables del conjunto base.
    """

    entradas: tuple[str, ...]
    target: str | None = None
    ventanas_lluvia_base_h: tuple[int, ...] = (3, 6, 24, 72)
    ventanas_lluvia_extra_h: tuple[int, ...] = (2, 12)
    maximos_lluvia_h: tuple[int, ...] = (3, 6)
    lags_lluvia_h: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    lags_lluvia_por_estacion: bool = False
    api_k: float = 0.9
    api_ventana_h: int = 72
    lags_nivel_h: tuple[int, ...] = (1, 3, 6, 12, 24)
    deltas_nivel_h: tuple[int, ...] = (1, 3)
    estaciones_areales: tuple[str, ...] | None = None
    variables_extendidas: bool = True

    def __post_init__(self) -> None:
        if not self.entradas:
            raise ValueError("entradas no puede estar vacía")
        if self.target is None:
            object.__setattr__(self, "target", self.entradas[0])
        if self.target not in self.entradas:
            raise ValueError(f"El objetivo {self.target} debe estar en las entradas")
        if not 0.0 < self.api_k < 1.0:
            raise ValueError("api_k debe estar en (0, 1)")
        todas = [*self.ventanas_lluvia_base_h, *self.ventanas_lluvia_extra_h, *self.maximos_lluvia_h,
                 *self.lags_lluvia_h, *self.lags_nivel_h, *self.deltas_nivel_h, self.api_ventana_h]
        if any(int(h) < 1 for h in todas):
            raise ValueError("Todas las ventanas y rezagos deben ser >= 1 h")

    @property
    def columnas_lluvia(self) -> list[str]:
        return [c for c in self.entradas if dominio(c) == "lluvia"]

    @property
    def columnas_areales(self) -> list[str]:
        return list(self.estaciones_areales) if self.estaciones_areales else self.columnas_lluvia

    @property
    def ventana_lluvia_h(self) -> int:
        """Horas de historia (incluida t) que usa la variable de lluvia más larga."""
        h = list(self.ventanas_lluvia_base_h)
        if self.variables_extendidas:
            h += [*self.ventanas_lluvia_extra_h, *self.maximos_lluvia_h, self.api_ventana_h,
                  *(l + 1 for l in self.lags_lluvia_h)]
        return max(h)

    @property
    def ventana_nivel_h(self) -> int:
        return max(max(self.lags_nivel_h), max(self.deltas_nivel_h)) + 1

    @property
    def ventana_requerida_h(self) -> int:
        """Horas mínimas de historia (incluida t) para que todas las variables de t estén definidas."""
        return max(self.ventana_lluvia_h, self.ventana_nivel_h)

    @classmethod
    def desde_horizontes_json(cls, ruta: str | Path, **kwargs) -> "ConfigVariables":
        """Construye la configuración con las entradas guardadas por el preprocesamiento."""
        with open(ruta, encoding="utf-8") as f:
            hz = json.load(f)
        extra = dict(hz.get("config_variables", {}))
        extra.update(kwargs)
        for k, v in list(extra.items()):
            if isinstance(v, list):
                extra[k] = tuple(v)
        return cls(entradas=tuple(hz["entradas"]), **extra)

    def a_dict(self) -> dict:
        """Parámetros serializables (para guardarlos en un json junto a los modelos)."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items() if k != "entradas"}


# ---------------------------------------------------------------------------------------------------------------
# Utilidades de nombres
# ---------------------------------------------------------------------------------------------------------------
def dominio(col: str) -> str:
    """'nivel', 'lluvia' o 'meteo' según el nombre ``ESTACION__Variable``."""
    variable = col.split("__")[1] if "__" in col else col
    if variable in ("Level_Avg", "Caudal"):
        return "nivel"
    if variable == "Lluvia_Tot":
        return "lluvia"
    return "meteo"


def estacion_de(col: str) -> str:
    return col.split("__")[0]


def _validar_ventana(df: pd.DataFrame, columnas: Iterable[str]) -> None:
    faltan = [c for c in columnas if c not in df.columns]

    if faltan:
        raise KeyError(f"Faltan columnas en la ventana: {faltan}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser un DatetimeIndex horario")

    if len(df) > 1:
        diferencias = df.index[1:] - df.index[:-1]

        if not (diferencias == pd.Timedelta(hours=1)).all():
            raise ValueError(
                "El índice debe ser horario, ordenado y sin horas faltantes "
                "(reindexar con NaN)"
            )


# ---------------------------------------------------------------------------------------------------------------
# Operadores causales (NaN en la ventana -> NaN)
# ---------------------------------------------------------------------------------------------------------------
def acumulado(serie: pd.Series, h: int) -> pd.Series:
    """Suma de las últimas h horas (t-h+1 … t)."""
    return serie.rolling(window=h, min_periods=h).sum()


def maximo(serie: pd.Series, h: int) -> pd.Series:
    """Máximo de la lluvia horaria en las últimas h horas (t-h+1 … t)."""
    return serie.rolling(window=h, min_periods=h).max()


def rezago(serie: pd.Series, h: int) -> pd.Series:
    """Valor en t-h."""
    return serie.shift(h)


def cambio(serie: pd.Series, h: int) -> pd.Series:
    """y(t) - y(t-h)."""
    return serie.diff(h)


def indice_precipitacion_antecedente(serie: pd.Series, k: float = 0.9, ventana_h: int = 72) -> pd.Series:
    """API(t) = Σ_{j=0}^{ventana_h-1} k^j · P(t-j).

    Versión truncada (FIR) del API recursivo API(t) = k·API(t-1) + P(t): con k = 0,9 y 72 h el peso omitido es
    0,9^72 ≈ 5e-4, y así la variable respeta la misma ventana de dependencia de 72 h que los demás acumulados.
    """
    x = serie.to_numpy(dtype=float)
    salida = np.zeros_like(x)
    for j in range(ventana_h):
        desplazado = np.full_like(x, np.nan)
        desplazado[j:] = x[:len(x) - j] if j else x
        salida = salida + (k ** j) * desplazado
    return pd.Series(salida, index=serie.index, name=serie.name)


def lluvia_areal(df: pd.DataFrame, columnas: Sequence[str]) -> pd.Series:
    """Media aritmética horaria de los pluviómetros (NaN si falta alguno, igual que la regla de 72 h)."""
    return df[list(columnas)].mean(axis=1, skipna=False).rename(NOMBRE_AREAL)


def codificacion_temporal(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Seno y coseno de la hora del día y del mes (la hora se toma del índice tal como viene, en UTC)."""
    h, m = idx.hour.values, idx.month.values
    return pd.DataFrame({
        "hora_sin": np.sin(2 * np.pi * h / 24), "hora_cos": np.cos(2 * np.pi * h / 24),
        "mes_sin": np.sin(2 * np.pi * (m - 1) / 12), "mes_cos": np.cos(2 * np.pi * (m - 1) / 12),
    }, index=idx)


# ---------------------------------------------------------------------------------------------------------------
# Construcción de variables
# ---------------------------------------------------------------------------------------------------------------
def _variables_lluvia(serie: pd.Series, col: str, cfg: ConfigVariables, es_areal: bool,
                      features: dict, catalogo: list) -> None:
    def agregar(nombre: str, valores: pd.Series, tipo: str, h: int, conjunto: str) -> None:
        features[nombre] = valores
        catalogo.append({"feature": nombre, "columna_base": col, "dominio": "lluvia", "tipo": tipo,
                         "ventana_horas": h, "conjunto": conjunto})

    if not es_areal:
        for h in cfg.ventanas_lluvia_base_h:
            agregar(f"{col}_acum_{h}h", acumulado(serie, h), "acumulado_rodante", h, CONJUNTO_BASE)
    if not cfg.variables_extendidas:
        return
    if es_areal:
        features[col] = serie
        catalogo.append({"feature": col, "columna_base": "|".join(cfg.columnas_areales), "dominio": "lluvia",
                         "tipo": "lluvia_areal", "ventana_horas": 1, "conjunto": CONJUNTO_EXTENDIDO})
        for h in cfg.ventanas_lluvia_base_h:
            agregar(f"{col}_acum_{h}h", acumulado(serie, h), "acumulado_rodante", h, CONJUNTO_EXTENDIDO)
    for h in cfg.ventanas_lluvia_extra_h:
        agregar(f"{col}_acum_{h}h", acumulado(serie, h), "acumulado_rodante", h, CONJUNTO_EXTENDIDO)
    for h in cfg.maximos_lluvia_h:
        agregar(f"{col}_max_{h}h", maximo(serie, h), "maximo_horario", h, CONJUNTO_EXTENDIDO)
    if es_areal or cfg.lags_lluvia_por_estacion:
        for h in cfg.lags_lluvia_h:
            agregar(f"{col}_lag_{h}h", rezago(serie, h), "lag", h, CONJUNTO_EXTENDIDO)
    agregar(f"{col}_api", indice_precipitacion_antecedente(serie, cfg.api_k, cfg.api_ventana_h),
            "indice_precipitacion_antecedente", cfg.api_ventana_h, CONJUNTO_EXTENDIDO)


def construir_derivadas(base: pd.DataFrame, cfg: ConfigVariables) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variables derivadas (lluvia, nivel y lluvia areal) a partir de las series base en unidades físicas.

    Returns:
        (derivadas, catalogo): DataFrame con las variables en orden fijo y su catálogo (una fila por variable).
    """
    features: dict[str, pd.Series] = {}
    catalogo: list[dict] = []
    for col in cfg.entradas:
        dom = dominio(col)
        serie = base[col].astype(float)
        if dom == "lluvia":
            _variables_lluvia(serie, col, cfg, False, features, catalogo)
        elif dom == "nivel":
            for h in cfg.lags_nivel_h:
                nombre = f"{col}_lag_{h}h"
                features[nombre] = rezago(serie, h)
                catalogo.append({"feature": nombre, "columna_base": col, "dominio": dom, "tipo": "lag",
                                 "ventana_horas": h, "conjunto": CONJUNTO_BASE})
            for h in cfg.deltas_nivel_h:
                nombre = f"{col}_delta_{h}h"
                features[nombre] = cambio(serie, h)
                catalogo.append({"feature": nombre, "columna_base": col, "dominio": dom,
                                 "tipo": "tasa_de_cambio", "ventana_horas": h, "conjunto": CONJUNTO_BASE})
    if cfg.variables_extendidas and cfg.columnas_areales:
        _variables_lluvia(lluvia_areal(base, cfg.columnas_areales), NOMBRE_AREAL, cfg, True, features, catalogo)
    return pd.DataFrame(features, index=base.index), pd.DataFrame(catalogo)


def catalogo_base(cfg: ConfigVariables, columnas_indicador: Sequence[str]) -> pd.DataFrame:
    """Catálogo de las columnas base, indicadores de imputación y codificación cíclica."""
    filas = [{"feature": c, "columna_base": c, "dominio": dominio(c), "tipo": "base", "ventana_horas": 0,
              "conjunto": CONJUNTO_BASE} for c in cfg.entradas]
    filas += [{"feature": c, "columna_base": c[:-len(SUFIJO_IMP)], "dominio": dominio(c[:-len(SUFIJO_IMP)]),
               "tipo": "indicador_imputacion", "ventana_horas": 0, "conjunto": CONJUNTO_BASE}
              for c in columnas_indicador]
    filas += [{"feature": c, "columna_base": "TIMESTAMP", "dominio": "tiempo", "tipo": "codificacion_ciclica",
               "ventana_horas": 0, "conjunto": CONJUNTO_BASE} for c in COLUMNAS_TIEMPO]
    return pd.DataFrame(filas)


def calcular_variables(df_ventana: pd.DataFrame, config: ConfigVariables,
                       indicadores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcula todas las variables de entrada (sin escalar) sobre una ventana horaria en unidades físicas.

    Args:
        df_ventana: índice horario continuo; columnas ``config.entradas`` ya imputadas (NaN = hueco no imputable).
            Si trae columnas ``<entrada>__imp`` se usan como indicadores de imputación.
        config: parámetros de construcción.
        indicadores: indicadores ``__imp`` (0/1) explícitos; tienen prioridad sobre los de ``df_ventana``.
            Si no hay indicadores para una entrada, se asume dato observado (0).

    Returns:
        DataFrame con el mismo índice y las columnas en el orden del preprocesamiento:
        entradas base, derivadas, codificación cíclica e indicadores ``__imp``. Las filas sin historia suficiente
        (las primeras ``config.ventana_requerida_h - 1``) o que tocan un NaN quedan con NaN en alguna columna.
    """
    _validar_ventana(df_ventana, config.entradas)
    base = df_ventana[list(config.entradas)].astype(float)
    derivadas, _ = construir_derivadas(base, config)
    tiempo = codificacion_temporal(base.index)
    imp = pd.DataFrame(index=base.index)
    for col in config.entradas:
        nombre = f"{col}{SUFIJO_IMP}"
        if indicadores is not None and nombre in indicadores.columns:
            imp[nombre] = indicadores[nombre].reindex(base.index).fillna(0).astype(np.int8)
        elif nombre in df_ventana.columns:
            imp[nombre] = df_ventana[nombre].fillna(0).astype(np.int8)
        else:
            imp[nombre] = np.int8(0)
    return pd.concat([base, derivadas, tiempo, imp], axis=1)


def ventana_dependencia(col: str, config: ConfigVariables) -> int:
    """Horas hacia atrás (sin contar t) de las que depende una entrada base a través de sus variables."""
    dom = dominio(col)
    if dom == "lluvia":
        return config.ventana_lluvia_h - 1
    if dom == "nivel":
        return config.ventana_nivel_h - 1
    return 0


def filas_completas(variables: pd.DataFrame, columnas: Sequence[str] | None = None) -> pd.Series:
    """True en las horas sin NaN en las columnas de entrada (condición necesaria de ``valido``)."""
    cols = list(columnas) if columnas is not None else list(variables.columns)
    return variables[cols].notna().all(axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Escalado Min-Max (ajustado en train por el preprocesamiento)
# ---------------------------------------------------------------------------------------------------------------
def leer_escaladores(ruta: str | Path) -> pd.DataFrame:
    """Lee ``escaladores_<slug>.csv`` (columnas: columna, metodo, min, max) indexado por columna."""
    return pd.read_csv(ruta).set_index("columna")


def ajustar_escaladores(train: pd.DataFrame, columnas: Sequence[str]) -> pd.DataFrame:
    """Mínimo y máximo por columna (solo con las filas que se le pasen, normalmente train válido)."""
    filas = [{"columna": c, "metodo": "minmax", "min": train[c].min(), "max": train[c].max()} for c in columnas]
    return pd.DataFrame(filas).set_index("columna")


def escalar(variables: pd.DataFrame, escaladores: pd.DataFrame) -> pd.DataFrame:
    """Aplica (x - min) / (max - min) a las columnas presentes en ``escaladores``; el resto queda igual.

    Las columnas con rango nulo se ponen en 0, como en el preprocesamiento. No se recorta a [0, 1]: valores
    fuera del rango de train se conservan (los árboles los tratan como el extremo; las redes extrapolan).
    """
    salida = variables.copy()
    for col in salida.columns:
        if col == "y_obs" or col not in escaladores.index:
            continue
        mn, mx = float(escaladores.at[col, "min"]), float(escaladores.at[col, "max"])
        rango = mx - mn
        salida[col] = 0.0 if (not np.isfinite(rango) or rango == 0) else (salida[col] - mn) / rango
    return salida


def desescalar(valores: np.ndarray | pd.Series, escaladores: pd.DataFrame,
               columna: str = ESCALA_OBJETIVO) -> np.ndarray:
    """Vuelve a unidades físicas (cm para el nivel; por defecto con la escala de ``y_obs``, igual a la del objetivo)."""
    mn, mx = float(escaladores.at[columna, "min"]), float(escaladores.at[columna, "max"])
    return np.asarray(valores, dtype=float) * (mx - mn) + mn


def desescalar_cambio(delta: np.ndarray | pd.Series, escaladores: pd.DataFrame,
                      columna: str = ESCALA_OBJETIVO) -> np.ndarray:
    """Convierte un cambio Δ escalado a cm (solo se multiplica por el rango; el mínimo se cancela)."""
    mn, mx = float(escaladores.at[columna, "min"]), float(escaladores.at[columna, "max"])
    return np.asarray(delta, dtype=float) * (mx - mn)


__all__ = [
    "ConfigVariables", "ESCALA_OBJETIVO", "NOMBRE_AREAL", "CONJUNTO_BASE", "CONJUNTO_EXTENDIDO", "COLUMNAS_TIEMPO",
    "SUFIJO_IMP",
    "dominio", "estacion_de", "acumulado", "maximo", "rezago", "cambio", "indice_precipitacion_antecedente",
    "lluvia_areal", "codificacion_temporal", "construir_derivadas", "catalogo_base", "calcular_variables",
    "ventana_dependencia", "filas_completas", "leer_escaladores", "ajustar_escaladores", "escalar", "desescalar",
    "desescalar_cambio",
]
