# -*- coding: utf-8 -*-
"""Genera ``src/models/modelos_<slug>.ipynb`` desde la plantilla común.

La formulación por defecto es la adoptada en la fase A del Yanuncay: objetivo Δ = y(t+τ) − nivel(t), variables de
lluvia extendidas y α = 0 (sin pesos de crecida). Se cambia al ejecutar con ``<PREFIJO>OBJETIVO``,
``<PREFIJO>VARIABLES`` y ``<PREFIJO>PESO_ALFA`` (p. ej. ``TOMEBAMBA_OBJETIVO=nivel``).

Uso (desde la raíz del repositorio)::

    python src/generadores/generar_modelos.py --rio tomebamba
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generadores import comun  # noqa: E402
from generadores.plantillas import modelos_1, modelos_2, modelos_3  # noqa: E402

PLANTILLA = comun.unir_plantillas([modelos_1, modelos_2, modelos_3])
METADATA = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
CLAVE_RUTA = "nb_modelos"
RIO_ABLACION = "yanuncay"   # río donde se hizo la ablación que fijó la formulación por defecto

ABLACION_PROPIA = (
    "- **Ablación (solo walk-forward en val, RF y XGBoost, 10 configuraciones):** base 0,115 → +Δ 0,138 →\n"
    "  +Δ + lluvia extendida 0,143 → + pesos α = 2: 0,126; α = 5: 0,113 (SS medio entre familias, folds y\n"
    "  horizontes). Formulación final: **Δ + lluvia extendida, α = 0** (`tablas/ablacion_walkforward.csv`).")
ABLACION_HEREDADA = (
    "- **Formulación por defecto (fase A del Yanuncay):** Δ + lluvia extendida, α = 0. Es la ganadora de la ablación\n"
    "  walk-forward del Yanuncay (base 0,115 → +Δ 0,138 → +Δ + lluvia extendida 0,143; los pesos α = 2 y α = 5 la\n"
    "  empeoraron; `Docs/resources/modelos_yanuncay/tablas/ablacion_walkforward.csv`). Se aplica igual en todos los\n"
    "  ríos; las otras formulaciones quedan disponibles con las variables de entorno de la tabla.")


def contexto(rio: Mapping) -> dict[str, str]:
    ctx = comun.contexto(rio)
    ctx["bloque_ablacion"] = ABLACION_PROPIA if rio["slug"] == RIO_ABLACION else ABLACION_HEREDADA
    return ctx


def construir(rio: Mapping):
    return comun.construir_notebook(PLANTILLA, contexto(rio), METADATA)


def generar(slug: str, salida: Path | None = None, config: Mapping | None = None, conservar: bool = True) -> Path:
    """Escribe el notebook del río; ``conservar`` mantiene las salidas de las celdas que no cambiaron."""
    rio = comun.obtener_rio(slug, config)
    return comun.escribir_notebook(construir(rio), comun.destino_notebook(rio, CLAVE_RUTA, salida), conservar)


if __name__ == "__main__":
    from generadores.generar_notebooks import ejecutar
    ejecutar(["--solo", "modelos", *sys.argv[1:]])
