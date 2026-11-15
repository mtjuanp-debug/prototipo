# -*- coding: utf-8 -*-
"""Genera ``src/data_prep/preprocesamiento_<slug>.ipynb`` desde la plantilla común.

Uso (desde la raíz del repositorio)::

    python src/generadores/generar_preprocesamiento.py --rio tomebamba
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generadores import comun  # noqa: E402
from generadores.plantillas import (preprocesamiento_1, preprocesamiento_2, preprocesamiento_3,  # noqa: E402
                                    preprocesamiento_4)

PLANTILLA = comun.unir_plantillas([preprocesamiento_1, preprocesamiento_2, preprocesamiento_3, preprocesamiento_4])
METADATA = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}}
CLAVE_RUTA = "nb_preprocesamiento"


def frase_literatura(rio: Mapping) -> str:
    """Cierre del contraste con la literatura (la referencia publicada es de Matadero-Sayausí, en el Tomebamba)."""
    if (rio.get("horizontes") or {}).get("referencia_literatura"):
        return f"Para la estación objetivo ({rio['estacion_objetivo']}) no hay valores publicados."
    return f"Para el {rio['nombre']} no hay valores publicados."


def aviso_falta_objetivo(rio: Mapping) -> str:
    """Mensaje que muestra el notebook si la serie objetivo no está en la tabla maestra."""
    texto = "Revise estacion_objetivo y columna_objetivo en config/rios.json."
    alternativa = rio.get("fuente_alternativa")
    if alternativa:
        texto += (f" La serie parece estar en {alternativa.get('archivo')} (nombre de archivo desplazado). "
                  f"Acción: {alternativa.get('accion')}. El notebook no usa datos sustitutos.")
    return texto


def contexto(rio: Mapping) -> dict[str, str]:
    ctx = comun.contexto(rio)
    ctx["frase_literatura"] = frase_literatura(rio)
    ctx["aviso_falta_objetivo"] = repr(aviso_falta_objetivo(rio))
    return ctx


def construir(rio: Mapping):
    return comun.construir_notebook(PLANTILLA, contexto(rio), METADATA)


def generar(slug: str, salida: Path | None = None, config: Mapping | None = None, conservar: bool = True) -> Path:
    """Escribe el notebook del río; ``conservar`` mantiene las salidas de las celdas que no cambiaron."""
    rio = comun.obtener_rio(slug, config)
    return comun.escribir_notebook(construir(rio), comun.destino_notebook(rio, CLAVE_RUTA, salida), conservar)


if __name__ == "__main__":
    from generadores.generar_notebooks import ejecutar
    ejecutar(["--solo", "preprocesamiento", *sys.argv[1:]])
