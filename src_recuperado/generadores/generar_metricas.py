# -*- coding: utf-8 -*-
"""Genera ``src/metrics/<slug>_models_metric.ipynb`` desde la plantilla común.

Uso (desde la raíz del repositorio)::

    python src/generadores/generar_metricas.py --rio tomebamba
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generadores import comun  # noqa: E402
from generadores.plantillas import metricas_1, metricas_2, metricas_3  # noqa: E402

PLANTILLA = comun.unir_plantillas([metricas_1, metricas_2, metricas_3])
METADATA = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
CLAVE_RUTA = "nb_metricas"


def contexto(rio: Mapping) -> dict[str, str]:
    return comun.contexto(rio)


def construir(rio: Mapping):
    return comun.construir_notebook(PLANTILLA, contexto(rio), METADATA)


def generar(slug: str, salida: Path | None = None, config: Mapping | None = None, conservar: bool = True) -> Path:
    """Escribe el notebook del río; ``conservar`` mantiene las salidas de las celdas que no cambiaron."""
    rio = comun.obtener_rio(slug, config)
    return comun.escribir_notebook(construir(rio), comun.destino_notebook(rio, CLAVE_RUTA, salida), conservar)


if __name__ == "__main__":
    from generadores.generar_notebooks import ejecutar
    ejecutar(["--solo", "metricas", *sys.argv[1:]])
