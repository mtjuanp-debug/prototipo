# -*- coding: utf-8 -*-
"""Utilidades comunes de los generadores de notebooks por río.

Los notebooks de cada río salen de una única plantilla (``generadores/plantillas``) en la que los valores propios
del río aparecen como marcadores ``@@clave@@``. Este módulo lee ``config/rios.json``, calcula el contexto de un río
(nombres, curva de descarga, umbrales, rutas y prefijo de las variables de entorno) y sustituye los marcadores.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[2]
RUTA_RIOS = RAIZ / "config" / "rios.json"
PATRON_SLUG = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
PATRON_MARCADOR = re.compile(r"@@[A-Za-z0-9_]+@@")
CLAVES_OBLIGATORIAS = ("slug", "nombre", "estacion_objetivo", "columna_objetivo", "curva", "umbrales_m3s",
                       "subcuenca", "subcuencas_entrada", "rutas")

Celda = tuple[str, str]


class ConfiguracionInvalida(ValueError):
    """``config/rios.json`` no tiene lo que necesita el generador (o el río pedido no existe)."""


# ---------------------------------------------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------------------------------------------
def cargar_rios(ruta: Path = RUTA_RIOS) -> dict:
    """Devuelve el objeto completo de ``rios.json`` (con las claves ``_meta`` y ``rios``)."""
    try:
        config = json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfiguracionInvalida(f"No se pudo leer {ruta}: {exc}") from exc
    if not isinstance(config.get("rios"), dict) or not config["rios"]:
        raise ConfiguracionInvalida(f"{ruta} debe tener un objeto 'rios' no vacío")
    return config


def obtener_rio(slug: str, config: Mapping | None = None) -> dict:
    """Entrada de un río, validada (slug con formato seguro y claves obligatorias)."""
    if not isinstance(slug, str) or not PATRON_SLUG.match(slug):
        raise ConfiguracionInvalida(f"Slug de río no válido: {slug!r}")
    config = cargar_rios() if config is None else config
    if slug not in config["rios"]:
        raise ConfiguracionInvalida(f"El río '{slug}' no está en rios.json (disponibles: {sorted(config['rios'])})")
    rio = config["rios"][slug]
    faltan = [k for k in CLAVES_OBLIGATORIAS if rio.get(k) in (None, "", [])]
    if faltan:
        raise ConfiguracionInvalida(f"El río '{slug}' no tiene {faltan} en rios.json")
    if rio["slug"] != slug:
        raise ConfiguracionInvalida(f"El río '{slug}' declara slug '{rio['slug']}'")
    if not rio["columna_objetivo"].startswith(rio["estacion_objetivo"] + "__"):
        raise ConfiguracionInvalida(f"columna_objetivo de '{slug}' no corresponde a estacion_objetivo")
    return rio


def ruta_en_repo(relativa: str, raiz: Path = RAIZ) -> Path:
    """Ruta absoluta de una ruta relativa de ``rios.json``; rechaza rutas que salgan del repositorio."""
    ruta = (raiz / relativa).resolve()
    if Path(relativa).is_absolute() or not ruta.is_relative_to(raiz.resolve()):
        raise ConfiguracionInvalida(f"La ruta {relativa!r} debe ser relativa y estar dentro del repositorio")
    return ruta


# ---------------------------------------------------------------------------------------------------------------
# Contexto de un río
# ---------------------------------------------------------------------------------------------------------------
def numero(x: float) -> str:
    """Literal de Python compacto: 1.45966e-06 -> '1.45966e-6'; 20 -> '20.0'."""
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", repr(float(x)))


def decimal_es(x: float, decimales: int) -> str:
    """Número con coma decimal (convención de la tesis)."""
    return f"{x:.{decimales}f}".replace(".", ",")


def nivel_desde_caudal(q: float, curva: Mapping) -> float:
    """Inversa de la curva de descarga Q = A (H - H0)^p (H en cm, Q en m3/s)."""
    return float(curva["H0"]) + (float(q) / float(curva["A"])) ** (1.0 / float(curva["p"]))


def contexto(rio: Mapping) -> dict[str, str]:
    """Valores comunes a las tres plantillas (los generadores añaden los suyos)."""
    curva, umbrales = rio["curva"], rio["umbrales_m3s"]
    q_pre, q_ale = float(umbrales["prealerta"]), float(umbrales["alerta"])
    if not q_pre < q_ale:
        raise ConfiguracionInvalida(f"Umbrales de '{rio['slug']}': prealerta debe ser menor que alerta")
    cm_pre, cm_ale = nivel_desde_caudal(q_pre, curva), nivel_desde_caudal(q_ale, curva)
    vecinas = [s for s in rio["subcuencas_entrada"] if s != rio["subcuenca"]]
    prefijo = rio["rutas"].get("prefijo_env") or f"{rio['slug'].upper()}_"
    if not re.match(r"^[A-Z][A-Z0-9_]*_$", prefijo):
        raise ConfiguracionInvalida(f"prefijo_env de '{rio['slug']}' no válido: {prefijo!r}")
    return {
        "slug": rio["slug"],
        "PREFIJO": prefijo,
        "nombre": rio["nombre"],
        "estacion": rio["estacion_objetivo"],
        "estacion_tesis": rio.get("estacion_tesis") or rio["estacion_objetivo"],
        "target": rio["columna_objetivo"],
        "subcuenca": rio["subcuenca"],
        "subcuencas_entrada": json.dumps(list(rio["subcuencas_entrada"]), ensure_ascii=False),
        "vecinas_barra": "/".join(vecinas) or "ninguna",
        "vecinas_coma": ", ".join(vecinas) or "ninguna",
        "curva_a": numero(curva["A"]),
        "curva_h0": numero(curva["H0"]),
        "curva_p": numero(curva["p"]),
        "q_pre": numero(q_pre),
        "q_ale": numero(q_ale),
        "q_pre_g": f"{q_pre:g}",
        "q_ale_g": f"{q_ale:g}",
        "cm_pre_1": decimal_es(cm_pre, 1),
        "cm_pre_2": decimal_es(cm_pre, 2),
        "cm_ale_2": decimal_es(cm_ale, 2),
    }


# ---------------------------------------------------------------------------------------------------------------
# Plantillas y notebooks
# ---------------------------------------------------------------------------------------------------------------
def renderizar(texto: str, ctx: Mapping[str, str]) -> str:
    """Sustituye los marcadores ``@@clave@@``; falla si queda alguno sin valor."""
    for clave, valor in ctx.items():
        texto = texto.replace(f"@@{clave}@@", str(valor))
    sobrantes = sorted(set(PATRON_MARCADOR.findall(texto)))
    if sobrantes:
        raise ConfiguracionInvalida(f"Marcadores sin valor en la plantilla: {sobrantes}")
    return texto


def unir_plantillas(modulos: Iterable) -> list[Celda]:
    """Concatena las celdas de los módulos de plantilla (cada uno expone ``CELDAS``)."""
    return [celda for modulo in modulos for celda in modulo.CELDAS]


def construir_notebook(celdas: Sequence[Celda], ctx: Mapping[str, str], metadata: Mapping) -> nbf.NotebookNode:
    """Notebook sin salidas; los ids de celda son deterministas para que regenerar no ensucie el diff."""
    nb = nbf.v4.new_notebook()
    for i, (tipo, texto) in enumerate(celdas):
        fuente = renderizar(texto, ctx).strip("\n")
        celda = nbf.v4.new_code_cell(fuente) if tipo == "code" else nbf.v4.new_markdown_cell(fuente)
        celda["id"] = f"c{i:02d}"
        nb.cells.append(celda)
    nb["metadata"] = json.loads(json.dumps(metadata))
    return nb


def conservar_salidas(nb: nbf.NotebookNode, anterior: nbf.NotebookNode) -> int:
    """Copia las salidas de ``anterior`` a las celdas de código cuyo texto no cambió (misma posición).

    Así regenerar un notebook ya ejecutado no borra resultados que siguen siendo válidos. Devuelve cuántas celdas
    conservaron su salida.
    """
    n = 0
    for nueva, vieja in zip(nb.cells, anterior.cells):
        if nueva.cell_type == vieja.cell_type == "code" and nueva.source == vieja.source and vieja.get("outputs"):
            nueva["outputs"] = vieja["outputs"]
            nueva["execution_count"] = vieja.get("execution_count")
            n += 1
    return n


def escribir_notebook(nb: nbf.NotebookNode, destino: Path, conservar: bool = True) -> Path:
    """Escribe el notebook; con ``conservar`` mantiene las salidas de las celdas que no cambiaron."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if conservar and destino.exists():
        try:
            conservar_salidas(nb, nbf.read(str(destino), as_version=4))
        except Exception as exc:  # noqa: BLE001 - un notebook anterior ilegible no debe impedir regenerar
            print(f"[aviso] No se pudieron conservar las salidas de {destino.name}: {exc}")
    nbf.validate(nb)
    nbf.write(nb, str(destino))
    return destino


def destino_notebook(rio: Mapping, clave_ruta: str, salida: Path | None = None) -> Path:
    """Ruta del notebook: la de ``rutas.<clave_ruta>`` en el repositorio, o el mismo nombre dentro de ``salida``."""
    relativa = rio["rutas"][clave_ruta]
    if salida is not None:
        return Path(salida).resolve() / Path(relativa).name
    return ruta_en_repo(relativa)
