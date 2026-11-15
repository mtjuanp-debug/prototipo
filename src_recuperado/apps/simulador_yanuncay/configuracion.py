"""Carga de la configuración del simulador, parametrizable por río.

Orden de precedencia (de menor a mayor):
1. ``config_simulador.json`` de esta carpeta (valores por defecto del Yanuncay).
2. ``rios[<rio>]`` del mismo archivo para los demás ríos: sus claves sustituyen a las del Yanuncay
   (``corridas``, ``rutas``, ``estaciones`` y ``etiquetas`` se reemplazan completas, no se fusionan).
   Si el río no define ``estaciones`` o ``etiquetas``, se derivan de las ``entradas`` de su
   ``horizontes_<rio>.json``.
3. ``<repo>/config/rios.json`` si existe: se fusiona la entrada del río solicitado
   (``{"yanuncay": {...}}`` o ``{"rios": {"yanuncay": {...}}}``). Si el archivo no existe o no
   tiene el río, se ignora sin error.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

CARPETA_APP = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA_APP.parents[2]
RUTA_CONFIG_LOCAL = CARPETA_APP / "config_simulador.json"
RUTA_RIOS = RAIZ_REPO / "config" / "rios.json"


def _fusionar(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Fusión recursiva: los valores de ``extra`` sobrescriben a los de ``base``."""
    salida = copy.deepcopy(base)
    for clave, valor in extra.items():
        if isinstance(valor, dict) and isinstance(salida.get(clave), dict):
            salida[clave] = _fusionar(salida[clave], valor)
        else:
            salida[clave] = copy.deepcopy(valor)
    return salida


def _config_rio_global(rio: str) -> dict[str, Any]:
    if not RUTA_RIOS.exists():
        return {}
    try:
        datos = json.loads(RUTA_RIOS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(datos, dict) and isinstance(datos.get("rios"), dict):
        datos = datos["rios"]
    if isinstance(datos, list):
        datos = {str(d.get("rio", d.get("nombre", ""))).lower(): d for d in datos if isinstance(d, dict)}
    if not isinstance(datos, dict):
        return {}
    entrada = datos.get(rio) or datos.get(rio.lower()) or datos.get(rio.capitalize())
    return entrada if isinstance(entrada, dict) else {}


def _traducir_rio_global(entrada: dict[str, Any]) -> dict[str, Any]:
    """Toma de ``config/rios.json`` solo los parámetros físicos de la estación (curva, umbrales, nombre).

    Las rutas de datos y modelos NO se toman de ahí: el simulador usa las de su propia configuración
    (o las del contrato de modelos), para no mezclar datos de una corrida con modelos de otra.
    """
    extra: dict[str, Any] = {}
    curva = entrada.get("curva") or entrada.get("curva_descarga")
    if isinstance(curva, dict):
        extra["curva_descarga"] = {"a": float(curva.get("A", curva.get("a"))), "H0": float(curva["H0"]),
                                   "p": float(curva["p"])}
    if isinstance(entrada.get("umbrales_m3s"), dict):
        extra["umbrales_m3s"] = {k: float(v) for k, v in entrada["umbrales_m3s"].items()}
    if entrada.get("nombre"):
        extra["nombre_rio"] = str(entrada["nombre"])
    if entrada.get("columna_objetivo"):
        extra["objetivo"] = str(entrada["columna_objetivo"])
    return extra


def corridas_disponibles(config: dict[str, Any]) -> dict[str, str]:
    """``{clave: etiqueta}`` de las corridas de modelos definidas en ``config['corridas']``."""
    return {k: str(v.get("etiqueta", k)) for k, v in config.get("corridas", {}).items()}


def _aplicar_corrida(base: dict[str, Any], corrida: str | None) -> dict[str, Any]:
    """Fusiona ``corridas[corrida]`` (rutas, familia por defecto, familias) sobre la configuración."""
    corridas = base.get("corridas", {})
    clave = corrida or base.get("corrida_defecto")
    if not clave:
        return base
    if clave not in corridas:
        raise KeyError(f"Corrida desconocida: {clave}. Disponibles: {sorted(corridas)}")
    extra = {k: v for k, v in corridas[clave].items() if k != "etiqueta"}
    config = _fusionar(base, extra)
    config["corrida"] = clave
    return config


CLAVES_REEMPLAZO = ("corridas", "rutas", "estaciones", "etiquetas")
NOMBRES_VARIABLE = {"Level_Avg": ("Nivel", "cm"), "Lluvia_Tot": ("Lluvia", "mm"), "HumAire_Avg": ("Humedad", "%"),
                    "RS_W_Avg": ("Radiación", "W/m²"), "TempAire_Avg": ("Temperatura", "°C")}


def _leer_local() -> dict[str, Any]:
    return json.loads(RUTA_CONFIG_LOCAL.read_text(encoding="utf-8"))


def rios_disponibles() -> dict[str, str]:
    """``{slug: nombre}`` de los ríos que el simulador puede cargar (el Yanuncay primero)."""
    local = _leer_local()
    salida = {str(local["rio"]): str(local.get("nombre_rio", local["rio"]))}
    salida.update({k: str(v.get("nombre_rio", k.capitalize())) for k, v in local.get("rios", {}).items()})
    return salida


def _etiqueta(col: str) -> str:
    """``'PLU_Ricaurte__Lluvia_Tot'`` -> ``'Lluvia Ricaurte (mm)'``."""
    estacion, _, variable = col.partition("__")
    nombre = estacion.split("_", 1)[-1].replace("PTAP", "").replace("-", " ").strip()
    texto, unidad = NOMBRES_VARIABLE.get(variable, (variable, ""))
    return f"{texto} {nombre} ({unidad})" if unidad else f"{texto} {nombre}"


def _estaciones_desde_entradas(entradas: list[str], objetivo: str) -> dict[str, list[str]]:
    """Agrupa las entradas del modelo en nivel (el objetivo primero), lluvia y meteorología."""
    niveles = [objetivo] + [c for c in entradas if c.endswith("__Level_Avg") and c != objetivo]
    lluvia = [c for c in entradas if c.endswith("__Lluvia_Tot")]
    meteo = [c for c in entradas if c not in niveles and c not in lluvia]
    return {"nivel": niveles, "lluvia": lluvia, "meteo": meteo}


def _aplicar_rio(base: dict[str, Any], rio: str) -> dict[str, Any]:
    """Sustituye los valores del Yanuncay por los de ``rios[rio]`` (sin cambios si ``rio`` es el base)."""
    if rio == base.get("rio"):
        return base
    extra = base.get("rios", {}).get(rio)
    if extra is None:
        raise KeyError(f"Río desconocido: {rio}. Disponibles: {sorted(rios_disponibles())}")
    config = _fusionar({k: v for k, v in base.items() if k not in CLAVES_REEMPLAZO},
                       {k: v for k, v in extra.items() if k not in CLAVES_REEMPLAZO})
    config.update({k: copy.deepcopy(extra[k]) for k in CLAVES_REEMPLAZO if k in extra})
    if "estaciones" not in config or "etiquetas" not in config:
        p = RAIZ_REPO / extra["horizontes_json"]
        entradas = list(json.loads(p.read_text(encoding="utf-8"))["entradas"])
        config.setdefault("estaciones", _estaciones_desde_entradas(entradas, config["objetivo"]))
        config.setdefault("etiquetas", {c: _etiqueta(c) for c in entradas})
    return config


def cargar_config(rio: str = "yanuncay", corrida: str | None = None) -> dict[str, Any]:
    """Configuración efectiva del simulador para ``rio`` y la corrida de modelos ``corrida``
    (por defecto ``corrida_defecto``: los modelos finales)."""
    base = _aplicar_corrida(_aplicar_rio(_leer_local(), rio), corrida)
    base.pop("rios", None)
    try:
        extra = _traducir_rio_global(_config_rio_global(rio))
    except (KeyError, TypeError, ValueError):
        extra = {}
    config = _fusionar(base, extra)
    config["fuente_parametros_rio"] = str(RUTA_RIOS) if extra else str(RUTA_CONFIG_LOCAL)
    config["rio"] = rio
    return config


def ruta(config: dict[str, Any], clave: str) -> Path | None:
    """Ruta absoluta de ``config['rutas'][clave]`` (relativa a la raíz del repositorio)."""
    valor = config.get("rutas", {}).get(clave)
    if not valor:
        return None
    p = Path(valor)
    return p if p.is_absolute() else RAIZ_REPO / p


def columnas_entrada(config: dict[str, Any]) -> list[str]:
    """Todas las columnas de sensores (nivel + lluvia + meteorología)."""
    e = config["estaciones"]
    return list(e["nivel"]) + list(e["lluvia"]) + list(e["meteo"])


def variable_de(col: str) -> str:
    """``'MET_X__Lluvia_Tot'`` -> ``'Lluvia_Tot'``."""
    return col.split("__")[1] if "__" in col else col


def estacion_de(col: str) -> str:
    return col.split("__")[0]
