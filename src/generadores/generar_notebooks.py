# -*- coding: utf-8 -*-
"""CLI: genera los notebooks de la línea base de uno o varios ríos a partir de ``config/rios.json``.

Uso (desde la raíz del repositorio)::

    python src/generadores/generar_notebooks.py --rio machangara
    python src/generadores/generar_notebooks.py --todos
    python src/generadores/generar_notebooks.py --rio tarqui --solo preprocesamiento
    python src/generadores/generar_notebooks.py --listar
    python src/generadores/generar_notebooks.py --rio yanuncay --salida <carpeta>   # para comparar sin pisar

Los notebooks se escriben en las rutas ``rutas.nb_*`` del río. Si el notebook ya existe, se conservan las salidas de
las celdas cuyo código no cambió (``--sin-salidas`` las descarta). No edite los notebooks generados a mano:
cambie la plantilla (``src/generadores/plantillas``) o ``config/rios.json`` y vuelva a generar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generadores import comun, generar_metricas, generar_modelos, generar_preprocesamiento  # noqa: E402

GENERADORES = {"preprocesamiento": generar_preprocesamiento, "modelos": generar_modelos,
               "metricas": generar_metricas}


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Genera los notebooks de la línea base por río (config/rios.json).")
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--rio", action="append", metavar="SLUG", help="slug del río (se puede repetir)")
    grupo.add_argument("--todos", action="store_true", help="todos los ríos de rios.json")
    grupo.add_argument("--listar", action="store_true", help="lista los ríos y su estado")
    ap.add_argument("--solo", choices=sorted(GENERADORES), action="append",
                    help="genera solo este notebook (se puede repetir)")
    ap.add_argument("--salida", type=Path, help="carpeta alternativa (no toca los notebooks del repositorio)")
    ap.add_argument("--sin-salidas", action="store_true",
                    help="no conservar las salidas de las celdas sin cambios (por defecto se conservan)")
    ap.add_argument("--config", type=Path, default=comun.RUTA_RIOS, help="ruta de rios.json")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    """Genera los notebooks pedidos y devuelve sus rutas (lanza ``ConfiguracionInvalida`` si algo falla)."""
    args = _argumentos(argv)
    config = comun.cargar_rios(args.config)
    if args.listar:
        for slug, rio in config["rios"].items():
            print(f"{slug:12s} {rio.get('nombre', ''):12s} {rio.get('responsable', ''):8s} {rio.get('estado', '')}")
        return []
    slugs = list(config["rios"]) if args.todos else list(dict.fromkeys(args.rio))
    tipos = args.solo or list(GENERADORES)
    escritos = []
    for slug in slugs:
        for tipo in tipos:
            ruta = GENERADORES[tipo].generar(slug, salida=args.salida, config=config,
                                              conservar=not args.sin_salidas)
            escritos.append(ruta)
            print(f"[{slug}] {tipo:16s} -> {ruta}")
    return escritos


def ejecutar(argv: Sequence[str] | None = None) -> None:
    """Punto de entrada de consola: errores de configuración como mensaje y código de salida 1."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # consolas de Windows en cp1252 (τ, Δ, ...)
    try:
        main(argv)
    except comun.ConfiguracionInvalida as exc:
        sys.exit(f"Error de configuración: {exc}")


if __name__ == "__main__":
    ejecutar()
