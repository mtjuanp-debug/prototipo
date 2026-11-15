# -*- coding: utf-8 -*-
"""Contraste conjunto de la hipótesis de trabajo (Tesis §1.5, P120-P121) para los cuatro ríos.

Lee, para cada río de ``config/rios.json``, la tabla ``diebold_mariano.csv`` que produce el notebook de
métricas (modelo seleccionado frente a la persistencia, una fila por horizonte, p-valor unilateral de la
prueba de Diebold-Mariano con varianza HAC de Newey-West y corrección de Harvey-Leybourne-Newbold).
Aplica la corrección de Holm-Bonferroni sobre **todas** las comparaciones río-horizonte disponibles
(hasta 8: cuatro ríos x dos horizontes; Tesis P109) y decide:

- H1: todos los ríos incluidos están evaluados y, en cada río y horizonte, SS > 0 y p_holm < alfa.
- H0: al menos una comparación disponible no cumple. Es definitivo aunque falten ríos, porque añadir
  comparaciones a Holm nunca reduce los p-valores ajustados ya calculados.
- pendiente: todas las comparaciones disponibles cumplen, pero faltan ríos por evaluar.

Uso (desde la raíz del repositorio):
    python src/metrics/hipotesis_conjunta.py --destino Docs/resources/hipotesis_conjunta.csv
    python src/metrics/hipotesis_conjunta.py --destino Docs/resources/hipotesis_conjunta.csv \
        --dm yanuncay=Docs/resources/preliminar/modelos_yanuncay/tablas/diebold_mariano.csv
    python src/metrics/hipotesis_conjunta.py --destino ... --excluir tarqui="serie objetivo no disponible"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

ALFA_POR_DEFECTO = 0.05
COLUMNAS_SALIDA = ["rio", "slug", "estacion", "modelo", "horizonte", "n", "SS", "DM", "p", "p_holm",
                   "cumple", "estado", "detalle"]
COLUMNAS_REQUERIDAS = {"horizonte", "SS", "p_unilateral"}
ESTADO_EVALUADO, ESTADO_PENDIENTE, ESTADO_EXCLUIDO = "evaluado", "pendiente", "excluido"


class EntradaInvalida(ValueError):
    """Error de validación de rutas, configuración o tablas de entrada."""


@dataclass(frozen=True)
class Comparacion:
    """Una comparación río-horizonte (modelo seleccionado frente a la persistencia)."""

    slug: str
    rio: str
    estacion: str
    modelo: str
    horizonte: int
    n: Optional[int]
    ss: float
    dm: float
    p: float


@dataclass
class EstadoRio:
    """Situación de un río: evaluado, pendiente (sin tabla) o excluido del análisis."""

    slug: str
    rio: str
    estacion: str
    estado: str
    detalle: str = ""
    comparaciones: list[Comparacion] = field(default_factory=list)


@dataclass(frozen=True)
class Resultado:
    tabla: pd.DataFrame
    veredicto: str
    mensaje: str
    pendientes: list[str]
    excluidos: list[str]


# ----------------------------------------------------------------------------------------------
# Estadística
# ----------------------------------------------------------------------------------------------
def holm_bonferroni(pvalores: Sequence[float]) -> list[float]:
    """p-valores ajustados por Holm (1979), en el mismo orden de entrada.

    p_holm(i) = max_{j <= i} min(1, (m - j + 1) p(j)) con los p-valores ordenados de menor a mayor.
    """
    p = [float(x) for x in pvalores]
    for x in p:
        if math.isnan(x) or not 0.0 <= x <= 1.0:
            raise EntradaInvalida(f"p-valor fuera de [0, 1]: {x}")
    m = len(p)
    orden = sorted(range(m), key=lambda i: p[i])
    ajustados = [0.0] * m
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - rango) * p[i]))
        ajustados[i] = acumulado
    return ajustados


# ----------------------------------------------------------------------------------------------
# Rutas y configuración
# ----------------------------------------------------------------------------------------------
def raiz_repositorio() -> Path:
    """Raíz del repositorio: la carpeta que contiene ``config/rios.json`` sobre este archivo."""
    for candidato in Path(__file__).resolve().parents:
        if (candidato / "config" / "rios.json").is_file():
            return candidato
    return Path.cwd().resolve()


def ruta_dentro_de(raiz: Path, relativa: str) -> Path:
    """Resuelve una ruta de la configuración y exige que quede dentro de ``raiz`` (sin '..' que escape)."""
    if not isinstance(relativa, str) or not relativa.strip():
        raise EntradaInvalida(f"Ruta vacía o no textual en la configuración: {relativa!r}")
    if "\x00" in relativa:
        raise EntradaInvalida("Ruta con carácter nulo")
    candidata = Path(relativa)
    if candidata.is_absolute():
        raise EntradaInvalida(f"La configuración debe usar rutas relativas a la raíz: {relativa}")
    resuelta = (raiz / candidata).resolve()
    try:
        resuelta.relative_to(raiz.resolve())
    except ValueError as exc:
        raise EntradaInvalida(f"Ruta fuera del repositorio: {relativa}") from exc
    return resuelta


def ruta_csv_usuario(texto: str, raiz: Path, debe_existir: bool) -> Path:
    """Valida una ruta CSV pasada por línea de comandos (relativa a la raíz o absoluta)."""
    if not texto or "\x00" in texto:
        raise EntradaInvalida(f"Ruta inválida: {texto!r}")
    ruta = Path(texto)
    ruta = (ruta if ruta.is_absolute() else raiz / ruta).resolve()
    if ruta.suffix.lower() != ".csv":
        raise EntradaInvalida(f"Se esperaba un archivo .csv: {ruta}")
    if debe_existir and not ruta.is_file():
        raise EntradaInvalida(f"No existe el archivo: {ruta}")
    if not debe_existir and ruta.exists() and not ruta.is_file():
        raise EntradaInvalida(f"El destino existe y no es un archivo: {ruta}")
    return ruta


def cargar_config(ruta: Path) -> dict:
    if not ruta.is_file():
        raise EntradaInvalida(f"No existe la configuración de ríos: {ruta}")
    try:
        config = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EntradaInvalida(f"rios.json no es un JSON válido: {exc}") from exc
    rios = config.get("rios")
    if not isinstance(rios, dict) or not rios:
        raise EntradaInvalida("rios.json debe tener un objeto 'rios' no vacío")
    for slug, rio in rios.items():
        for clave in ("nombre", "estacion_objetivo", "rutas"):
            if clave not in rio:
                raise EntradaInvalida(f"Río '{slug}' sin la clave '{clave}'")
        if "diebold_mariano" not in rio["rutas"]:
            raise EntradaInvalida(f"Río '{slug}' sin rutas.diebold_mariano")
    return config


def parsear_pares(pares: Iterable[str], opcion: str) -> dict[str, str]:
    """Convierte ['clave=valor', ...] en un diccionario (claves en minúsculas)."""
    salida: dict[str, str] = {}
    for par in pares:
        clave, sep, valor = par.partition("=")
        if opcion == "--dm" and not (sep and valor):
            raise EntradaInvalida(f"{opcion} espera RIO=RUTA, se recibió: {par}")
        salida[clave.strip().lower()] = valor.strip()
    return salida


def horizontes_esperados(rio: dict) -> set[int]:
    h = rio.get("horizontes") or {}
    return {int(h[k]) for k in ("tau1", "tau2") if h.get(k) is not None}


# ----------------------------------------------------------------------------------------------
# Lectura de las tablas Diebold-Mariano
# ----------------------------------------------------------------------------------------------
def leer_diebold_mariano(ruta: Path, slug: str, rio: dict) -> list[Comparacion]:
    """Lee las filas del modelo seleccionado (una por horizonte) y valida su contenido."""
    df = pd.read_csv(ruta)
    faltan = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltan:
        raise EntradaInvalida(f"{ruta.name} de '{slug}' no tiene las columnas {sorted(faltan)}")
    if "seleccionado" in df.columns:
        df = df[df["seleccionado"].astype(str).str.lower().isin(["true", "1"])]
    if df.empty:
        raise EntradaInvalida(f"{ruta.name} de '{slug}' no tiene filas del modelo seleccionado")
    if df["horizonte"].duplicated().any():
        raise EntradaInvalida(f"{ruta.name} de '{slug}' tiene más de una fila por horizonte")
    columna_dm = "DM_HLN" if "DM_HLN" in df.columns else ("DM" if "DM" in df.columns else None)
    esperados = horizontes_esperados(rio)
    encontrados = {int(h) for h in df["horizonte"]}
    if esperados and encontrados != esperados:
        raise EntradaInvalida(f"Horizontes de '{slug}' en {ruta.name} ({sorted(encontrados)}) no coinciden "
                              f"con rios.json ({sorted(esperados)})")
    comparaciones = []
    for fila in df.sort_values("horizonte").itertuples(index=False):
        d = fila._asdict()
        p, ss = float(d["p_unilateral"]), float(d["SS"])
        if math.isnan(p) or not 0.0 <= p <= 1.0 or math.isnan(ss):
            raise EntradaInvalida(f"Valores inválidos en '{slug}' h={d['horizonte']}: p={p}, SS={ss}")
        comparaciones.append(Comparacion(
            slug=slug, rio=rio["nombre"], estacion=rio["estacion_objetivo"],
            modelo=str(d.get("modelo", "")), horizonte=int(d["horizonte"]),
            n=int(d["n"]) if "n" in d and pd.notna(d["n"]) else None, ss=ss,
            dm=float(d[columna_dm]) if columna_dm else float("nan"), p=p))
    return comparaciones


def recolectar(config: dict, raiz: Path, rutas_dm: dict[str, str],
               excluidos: dict[str, str]) -> list[EstadoRio]:
    """Estado de cada río según la existencia de su tabla y las exclusiones declaradas."""
    rios = config["rios"]
    desconocidos = (set(rutas_dm) | set(excluidos)) - set(rios)
    if desconocidos:
        raise EntradaInvalida(f"Ríos desconocidos (revise rios.json): {sorted(desconocidos)}")
    estados = []
    for slug, rio in rios.items():
        base = EstadoRio(slug=slug, rio=rio["nombre"], estacion=rio["estacion_objetivo"], estado="")
        if slug in excluidos:
            base.estado, base.detalle = ESTADO_EXCLUIDO, excluidos[slug] or "excluido por el usuario"
        else:
            ruta = (ruta_csv_usuario(rutas_dm[slug], raiz, debe_existir=True) if slug in rutas_dm
                    else ruta_dentro_de(raiz, rio["rutas"]["diebold_mariano"]))
            if ruta.is_file():
                base.estado, base.detalle = ESTADO_EVALUADO, str(ruta)
                base.comparaciones = leer_diebold_mariano(ruta, slug, rio)
            else:
                base.estado, base.detalle = ESTADO_PENDIENTE, f"sin {ruta.name} (se espera en {ruta})"
        estados.append(base)
    return estados


# ----------------------------------------------------------------------------------------------
# Contraste conjunto
# ----------------------------------------------------------------------------------------------
def contrastar(estados: Sequence[EstadoRio], alfa: float = ALFA_POR_DEFECTO) -> Resultado:
    if not 0.0 < alfa < 1.0:
        raise EntradaInvalida(f"alfa debe estar en (0, 1): {alfa}")
    comparaciones = [c for e in estados for c in e.comparaciones]
    p_holm = holm_bonferroni([c.p for c in comparaciones])
    filas = []
    for c, ph in zip(comparaciones, p_holm):
        filas.append({"rio": c.rio, "slug": c.slug, "estacion": c.estacion, "modelo": c.modelo,
                      "horizonte": c.horizonte, "n": c.n, "SS": c.ss, "DM": c.dm, "p": c.p, "p_holm": ph,
                      "cumple": bool(c.ss > 0 and ph < alfa), "estado": ESTADO_EVALUADO,
                      "detalle": f"Holm sobre {len(comparaciones)} comparaciones"})
    for e in estados:
        if e.estado != ESTADO_EVALUADO:
            filas.append({"rio": e.rio, "slug": e.slug, "estacion": e.estacion, "estado": e.estado,
                          "detalle": e.detalle})
    tabla = pd.DataFrame(filas, columns=COLUMNAS_SALIDA).astype({"horizonte": "Int64", "n": "Int64"})

    pendientes = [e.rio for e in estados if e.estado == ESTADO_PENDIENTE]
    excluidos = [e.rio for e in estados if e.estado == ESTADO_EXCLUIDO]
    evaluados = tabla[tabla["estado"] == ESTADO_EVALUADO]
    fallan = evaluados[~evaluados["cumple"].astype(bool)]
    if evaluados.empty:
        veredicto = "pendiente"
        mensaje = "No hay ningún río evaluado todavía."
    elif not fallan.empty:
        veredicto = "H0"
        lista = ", ".join(f"{r.rio} h{r.horizonte} (SS={r.SS:.3f}, p_holm={r.p_holm:.3g})"
                          for r in fallan.itertuples())
        mensaje = (f"No se rechaza H0 (P121): no cumplen SS > 0 y p_holm < {alfa} -> {lista}. "
                   "Es definitivo: los ríos pendientes solo pueden aumentar los p-valores ajustados.")
    elif pendientes:
        veredicto = "pendiente"
        mensaje = (f"Las {len(evaluados)} comparaciones disponibles cumplen, pero faltan: "
                   f"{', '.join(pendientes)}. El veredicto final requiere todos los ríos incluidos.")
    else:
        veredicto = "H1"
        mensaje = (f"Se acepta H1 (P120): las {len(evaluados)} comparaciones río-horizonte tienen SS > 0 "
                   f"y p_holm < {alfa}.")
    if excluidos:
        mensaje += f" Ríos excluidos del análisis: {', '.join(excluidos)}."
    return Resultado(tabla=tabla, veredicto=veredicto, mensaje=mensaje, pendientes=pendientes,
                     excluidos=excluidos)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Holm-Bonferroni conjunto y veredicto H1/H0 (Tesis P120-P121).")
    ap.add_argument("--destino", required=True, help="CSV de salida (relativo a la raíz o absoluto)")
    ap.add_argument("--rios", default="config/rios.json", help="Configuración de ríos (por defecto config/rios.json)")
    ap.add_argument("--raiz", default=None, help="Raíz del repositorio (por defecto se detecta)")
    ap.add_argument("--alfa", type=float, default=ALFA_POR_DEFECTO, help="Nivel de significancia (0,05)")
    ap.add_argument("--dm", action="append", default=[], metavar="RIO=RUTA",
                    help="Usa otra tabla diebold_mariano.csv para un río (repetible)")
    ap.add_argument("--excluir", action="append", default=[], metavar="RIO[=MOTIVO]",
                    help="Excluye formalmente un río del análisis (repetible)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        raiz = Path(args.raiz).resolve() if args.raiz else raiz_repositorio()
        if not raiz.is_dir():
            raise EntradaInvalida(f"La raíz no es una carpeta: {raiz}")
        ruta_rios = Path(args.rios)
        ruta_rios = (ruta_rios if ruta_rios.is_absolute() else raiz / ruta_rios).resolve()
        config = cargar_config(ruta_rios)
        destino = ruta_csv_usuario(args.destino, raiz, debe_existir=False)
        estados = recolectar(config, raiz, parsear_pares(args.dm, "--dm"),
                             parsear_pares(args.excluir, "--excluir"))
        resultado = contrastar(estados, args.alfa)
    except EntradaInvalida as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.tabla.to_csv(destino, index=False, encoding="utf-8")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(resultado.tabla.drop(columns=["detalle"]).to_string(index=False))
    print(f"\nVeredicto: {resultado.veredicto}\n{resultado.mensaje}\nTabla: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
