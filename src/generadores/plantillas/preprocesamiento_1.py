# -*- coding: utf-8 -*-
"""Plantilla del notebook de preprocesamiento (src/data_prep/preprocesamiento_<slug>.ipynb): parte 1 de 4.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("md", r"""
# Preprocesamiento del dataset hidrometeorológico — Río @@nombre@@

Derivación del pipeline general (`preprocesamiento_dataset_hidrometeorologico v1.ipynb`)
para el modelo del **río @@nombre@@**. La variable objetivo es el **nivel** (cm) de la estación
`@@estacion@@` (ver `Docs/VariableObjetivo.docx`).

Los capítulos 1 y 2 de la tesis (`Docs/TesisCardenasPlazaSanchezRivadeneira.docx`) son la
**fuente de verdad**. Este notebook aplica en particular:
- **Regla de 72 h** (alcance, párrafo de imputación): los vacíos de más de 72 h **no se imputan** y
  las horas que dependen de ellos se excluyen del entrenamiento y la evaluación (columna `valido`).
- **Métodos por tipo de variable**: nivel (lineal, PCHIP, regresión con vecina, climatología solo
  en 24–72 h fuera de crecidas), lluvia (CCW primero, IDW como alternativa) y meteorología
  (missForest iterativo frente a interpolación y climatología).
- **Validación** con huecos artificiales de 1, 6, 24 y 72 h solo en train; nMAE y nRMSE
  normalizados por el **rango** observado en train.
- **Tiempo de respuesta (Tr)** estimado en train → horizontes τ₂ = ⌊Tr⌋ y τ₁ = max(1, round(τ₂/2)).
- **Partición** 70/15/15 con un **margen** = ventana máxima de rezagos + τ₂ entre conjuntos y
  al menos 3 eventos de prealerta en validación y en prueba.

| # | Paso | Qué hace |
|---|---|---|
| 0 | Configuración | Rutas, objetivo, umbrales de prealerta/alerta desde la curva de descarga |
| 1 | Carga y auditoría | Tabla homologada (grilla horaria con NaN explícitos) + catálogo; auditoría de los `.dat` |
| 2 | Estaciones candidatas | @@subcuenca@@ + vecinas de @@vecinas_barra@@, filtradas por completitud |
| 3 | Control de calidad | Rangos físicos, radiación nocturna, IQR (meteorología sin radiación), niveles congelados |
| 4 | Tiempo de respuesta y horizontes | Eventos de prealerta en train, lluvia areal, Tr, τ₁ y τ₂ |
| 5 | Regla de 72 h y selección por costo | Huecos largos, ventana de dependencia y costo en horas de cada entrada |
| 6 | Métodos de imputación | Ajustados solo con train |
| 7 | Validación con huecos artificiales | 1/6/24/72 h en train, nRMSE por rango, ranking por tipo y banda |
| 8 | Imputación final | Solo huecos ≤ 72 h; indicadores `__imp`; `y_obs` nunca se imputa |
| 9 | Variables derivadas y `valido` | Acumulados, rezagos, deltas, tiempo cíclico; filas válidas sin huecos largos ni información futura |
| 10 | Partición con margen | Cortes fijos 70/15/15 + margen; verificación de ≥ 3 eventos en val y test |
| 11 | Escalado, verificación y guardado | Min-Max en filas válidas de train; parquet, reportes y `horizontes_@@slug@@.json` |
"""),
    ("md", r"""
## Paso 0 — Configuración
"""),
    ("code", r'''
import json
import math
import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", message="\\[IterativeImputer\\] Early stopping criterion not reached")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)


def _raiz_repo():
    # El notebook vive en src/data_prep; se sube hasta encontrar Docs/resources.
    actual = Path.cwd().resolve()
    for candidato in [actual, *actual.parents]:
        if (candidato / "Docs" / "resources").exists():
            return candidato
    raise FileNotFoundError("No se encontró la carpeta Docs/resources del repositorio.")


ROOT = _raiz_repo()
RES_DIR = ROOT / "Docs" / "resources"
# Variables derivadas: módulo compartido con la aplicación de tiempo real (src/data_prep/variables_rio.py)
if str(ROOT / "src" / "data_prep") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import variables_rio as vy
HOMOLOGADA_PATH = RES_DIR / "tabla_maestra_homologada.parquet"
COORDS_PATH = RES_DIR / "coordenadas.xlsx"
DAT_DIR = RES_DIR / "preprocesados_base"
OUT_DIR = RES_DIR

TARGET = "@@target@@"
SUBCUENCA_OBJETIVO = "@@subcuenca@@"
SUBCUENCAS_ENTRADA = @@subcuencas_entrada@@
VARIABLES_METEO_ENTRADA = {"TempAire_Avg", "HumAire_Avg", "RS_W_Avg"}

# Filtro inicial de candidatas (luego decide el costo en horas de la regla de 72 h, Paso 5).
MAX_PCT_NAN_TRAIN = 0.50
MAX_PCT_NAN_VAL_TEST = 0.50

# Cortes base 70/15/15 (mismos que el pipeline general -> comparables con los otros ríos).
FIN_TRAIN = pd.Timestamp("2022-12-13 17:00", tz="UTC")
FIN_VAL = pd.Timestamp("2024-10-23 00:00", tz="UTC")

# Umbrales oficiales de ETAPA EP para el @@nombre@@ (Tabla 3 de la tesis), en caudal. La tesis compara en
# caudal el nivel transformado con la curva Q = a (H - H0)^p (§1.3 P103). Como la curva es creciente,
# Q >= @@q_pre_g@@ m3/s equivale a H >= H0 + (@@q_pre_g@@/a)^(1/p); se usa ese nivel equivalente para filtrar más rápido.
CURVA = {"a": @@curva_a@@, "H0": @@curva_h0@@, "p": @@curva_p@@}
Q_PREALERTA, Q_ALERTA = @@q_pre@@, @@q_ale@@   # m3/s


def nivel_desde_caudal(q, a=CURVA["a"], h0=CURVA["H0"], p=CURVA["p"]):
    return h0 + (q / a) ** (1.0 / p)


UMBRAL_PREALERTA_CM = float(nivel_desde_caudal(Q_PREALERTA))
UMBRAL_ALERTA_CM = float(nivel_desde_caudal(Q_ALERTA))

# Faltantes
MAX_HUECO_IMPUTABLE_H = 72          # regla de la tesis: > 72 h no se imputa
HUECO_CORTO_H = 6                   # frontera entre banda "corto" (<= 6 h) y "largo" (7-72 h)
LONGITUDES_HUECO = (1, 6, 24, 72)   # longitudes evaluadas con huecos artificiales
N_HUECOS_POR_LONGITUD = 60
N_DONANTES = 4
MIN_TRASLAPE_H = 1000               # horas comunes mínimas para estimar correlación/regresión
MAX_LAG_VECINA_H = 12               # desfase máximo explorado para la regresión con vecina
MIN_CORR_VECINA = 0.5
UMBRAL_LLUVIA_MM = 0.1
MAX_COSTO_EXTRA = 0.05              # una entrada se conserva si excluye <= 5 % de horas extra en train

# Control de calidad
K_IQR = 3.0
MIN_RACHA_CONGELADA_H = 24          # nivel idéntico durante >= 24 h -> sensor congelado
UMBRAL_RS_NOCHE = 50.0              # W/m2 en horas nocturnas -> error de sensor

# Tiempo de respuesta
VENTANA_TR_H = 24                   # lluvia considerada antes del pico
MIT_TR_H = 6                        # tiempo mínimo entre tormentas (horas secas) para separar la lluvia
SEPARACION_EVENTOS_H = 6            # cruces separados por < 6 h = mismo evento (tesis)
MIN_EVENTOS_UMBRAL = 10             # si hay menos eventos de prealerta en train se usa P95
MIN_LLUVIA_EVENTO_MM = 1.0          # lluvia areal mínima para que el evento sea interpretable
MIN_EVENTOS_PARTICION = 3

# Variables derivadas (los valores por defecto de vy.ConfigVariables; aquí se repiten los que fijan la ventana
# de dependencia de la regla de 72 h). Las variables de lluvia extendidas (rezagos 1-6 h de la lluvia areal,
# acumulados de 2 y 12 h, máximos horarios de 3 y 6 h, API con k = 0,9 truncado a 72 h y lluvia areal) no amplían
# la ventana: ninguna mira más de 72 h hacia atrás.
VENTANAS_LLUVIA_HORAS = vy.ConfigVariables.__dataclass_fields__["ventanas_lluvia_base_h"].default   # (3, 6, 24, 72)
LAGS_NIVEL_HORAS = vy.ConfigVariables.__dataclass_fields__["lags_nivel_h"].default                  # (1, 3, 6, 12, 24)
DELTAS_NIVEL_HORAS = vy.ConfigVariables.__dataclass_fields__["deltas_nivel_h"].default              # (1, 3)
VENTANA_MAX_H = max(max(VENTANAS_LLUVIA_HORAS), max(LAGS_NIVEL_HORAS))   # 72 h

# missForest (IterativeImputer + ExtraTrees)
MF_N_ESTIMADORES = 50
MF_MAX_ITER = 10
# Predictores del missForest: meteorología de las subcuencas de entrada con <= 50 % NaN en train
# (las estaciones excluidas como entrada, p. ej. Pucán, siguen sirviendo como predictores).
MAX_PCT_NAN_MF = 0.50
# Lluvia: la tesis prioriza CCW; otro método solo lo reemplaza si mejora el nRMSE en más de 5 %.
TOLERANCIA_CCW = 0.05

SEED = 42
rng = np.random.default_rng(SEED)

print("Repositorio:", ROOT)
print("Fuente:", HOMOLOGADA_PATH.name, "| existe:", HOMOLOGADA_PATH.exists())
print(f"Umbral de prealerta: {Q_PREALERTA:.0f} m3/s -> {UMBRAL_PREALERTA_CM:.2f} cm")
print(f"Umbral de alerta:    {Q_ALERTA:.0f} m3/s -> {UMBRAL_ALERTA_CM:.2f} cm")
print(f"Ventana máxima de rezagos/acumulados: {VENTANA_MAX_H} h")
'''),
    ("code", r'''
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.strip().lower()


def dominio(col):
    variable = col.split("__")[1]
    if variable in ("Level_Avg", "Caudal"):
        return "nivel"
    if variable == "Lluvia_Tot":
        return "lluvia"
    return "meteo"


def estacion_de(col):
    return col.split("__")[0]


codificacion_temporal = vy.codificacion_temporal   # seno/coseno de hora y mes (módulo compartido)


def guardar_figura(fig, nombre):
    fig.savefig(OUT_DIR / nombre, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def rachas_nan(s):
    """Para cada hora: longitud de la racha de NaN que la contiene (0 si hay dato),
    inicio y fin (posiciones) de esa racha y si la racha toca un extremo de la serie."""
    es_na = s.isna().values
    n = len(es_na)
    largo = np.zeros(n, dtype=int)
    ini = np.full(n, -1)
    fin = np.full(n, -1)
    borde = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if es_na[i]:
            j = i
            while j < n and es_na[j]:
                j += 1
            largo[i:j] = j - i
            ini[i:j] = i
            fin[i:j] = j - 1
            borde[i:j] = (i == 0) or (j == n)
            i = j
        else:
            i += 1
    return largo, ini, fin, borde


def agrupar_eventos(serie, umbral, separacion_h=SEPARACION_EVENTOS_H):
    """Eventos = cruces del umbral por el nivel observado; cruces separados por < separacion_h
    horas se agrupan. Devuelve DataFrame con inicio, fin, pico y hora del pico."""
    s = serie.dropna()
    sobre = s[s > umbral]
    if sobre.empty:
        return pd.DataFrame(columns=["inicio", "fin", "t_pico", "pico", "horas_sobre_umbral"])
    t = sobre.index
    nuevo = np.r_[True, (t[1:] - t[:-1]) >= pd.Timedelta(hours=separacion_h)]
    grupo = np.cumsum(nuevo)
    filas = []
    for g in np.unique(grupo):
        tt = t[grupo == g]
        tramo = s.loc[tt[0]:tt[-1]]
        filas.append({"inicio": tt[0], "fin": tt[-1], "t_pico": tramo.idxmax(), "pico": tramo.max(),
                      "horas_sobre_umbral": len(tt)})
    return pd.DataFrame(filas)
'''),
    ("md", r"""
## Paso 1 — Carga de la tabla homologada y del catálogo

Se parte de `tabla_maestra_homologada.parquet`: es la salida del Paso 3 del pipeline general,
es decir, la grilla horaria continua con NaN **explícitos** y **antes de cualquier imputación**.
Se pivotea a formato ancho (una fila por hora, una columna por `estacion__variable`).
"""),
    ("code", r'''
def leer_catalogo(coords_path):
    cat = pd.read_excel(coords_path)
    renombres = {"Código": "estacion", "Codigo": "estacion", "Tipo": "tipo", "Subcuenca": "subcuenca",
                 "X_UTM": "x", "Y_UTM": "y", "X": "x", "Y": "y", "Z": "z"}
    cat = cat.rename(columns={k: v for k, v in renombres.items() if k in cat.columns})
    cat["clave"] = cat["estacion"].map(_norm)
    return cat


larga = pd.read_parquet(HOMOLOGADA_PATH)
larga["col"] = larga["estacion"] + "__" + larga["variable"]
ancho_full = larga.pivot_table(index="TIMESTAMP", columns="col", values="valor", aggfunc="first").sort_index()
grilla = pd.date_range(ancho_full.index.min(), ancho_full.index.max(), freq="h", tz="UTC")
ancho_full = ancho_full.reindex(grilla)
ancho_full.index.name = "TIMESTAMP"
if TARGET not in ancho_full.columns:
    # Sin la serie objetivo no hay nada que preprocesar: se detiene aquí con la causa y la acción sugerida.
    raise KeyError(f"La serie objetivo {TARGET} no está en {HOMOLOGADA_PATH.name}. "
                   + @@aviso_falta_objetivo@@)

subcuenca_de = larga.drop_duplicates("estacion").set_index("estacion")["subcuenca"]
catalogo = leer_catalogo(COORDS_PATH)
COORDS = {r.clave: (r.x, r.y) for r in catalogo.itertuples() if pd.notna(r.x) and pd.notna(r.y)}
del larga

print(f"Tabla ancha: {ancho_full.shape[0]:,} horas x {ancho_full.shape[1]} columnas "
      f"({ancho_full.index.min()} -> {ancho_full.index.max()})")
print(f"Estaciones con coordenadas en el catálogo: {len(COORDS)}")
'''),
    ("md", r"""
### Auditoría de los archivos `.dat` de `preprocesados_base`

Antes de decidir la fuente se compararon los `.dat` disponibles contra la tabla homologada,
valor por valor. Para cada archivo y variable se busca la estación de la tabla homologada con la
que comparte valores idénticos. Si el nombre del archivo no coincide con esa estación, el archivo
está **mal nombrado** respecto al lote con el que se construyó el pipeline, y no se usa.
"""),
    ("code", r'''
def auditar_nombres_dat(dat_dir, ancho):
    filas = []
    for f in sorted(dat_dir.glob("*.dat")):
        d = pd.read_csv(f, na_values=["NA", ""], low_memory=False)
        if "TIMESTAMP" not in d.columns:
            filas.append({"archivo": f.name, "variable": None, "coincide_con": None,
                          "pct_valores_identicos": None, "nombre_coherente": False,
                          "nota": "sin columna TIMESTAMP"})
            continue
        d["TIMESTAMP"] = pd.to_datetime(d["TIMESTAMP"], utc=True, errors="coerce")
        d = d.dropna(subset=["TIMESTAMP"]).drop_duplicates("TIMESTAMP").set_index("TIMESTAMP")
        estacion_archivo = re.sub(r"\(\d+\)$", "", f.stem)
        for variable in ("Level_Avg", "Lluvia_Tot", "TempAire_Avg"):
            if variable not in d.columns:
                continue
            s = pd.to_numeric(d[variable], errors="coerce").reindex(ancho.index)
            mejor, mejor_pct = None, 0.0
            for c in [c for c in ancho.columns if c.endswith("__" + variable)]:
                comun = s.notna() & ancho[c].notna()
                if comun.sum() < 100:
                    continue
                pct = float((s[comun] - ancho.loc[comun, c]).abs().lt(1e-6).mean())
                if pct > mejor_pct:
                    mejor, mejor_pct = c.split("__")[0], pct
            filas.append({"archivo": f.name, "variable": variable, "coincide_con": mejor,
                          "pct_valores_identicos": round(100 * mejor_pct, 1),
                          "nombre_coherente": mejor is not None and _norm(mejor) == _norm(estacion_archivo),
                          "nota": ""})
    return pd.DataFrame(filas)


if DAT_DIR.exists():
    auditoria_dat = auditar_nombres_dat(DAT_DIR, ancho_full)
    por_archivo = auditoria_dat.groupby("archivo")["nombre_coherente"].all()
    print(f"Archivos .dat cuyo nombre coincide con la estación que realmente contienen: "
          f"{int(por_archivo.sum())} de {len(por_archivo)}")
    auditoria_dat.to_csv(OUT_DIR / "auditoria_nombres_dat.csv", index=False)
    display(auditoria_dat[auditoria_dat["variable"].isin(["Level_Avg"]) | ~auditoria_dat["nombre_coherente"]].head(40))
else:
    print("No se encontró preprocesados_base; se omite la auditoría.")
'''),
    ("md", r"""
**Decisión:** la tabla homologada es la **única fuente**. Los `.dat` de `preprocesados_base`
tienen los nombres desplazados (por ejemplo, la serie de nivel del Yanuncay está en un archivo
llamado `LIM_Tarqui-AJ-Yanuncay.dat`). Antes de volver a usarlos, ETAPA EP o el equipo deben
confirmar la correspondencia entre archivo y estación.

### Recorte al periodo del objetivo
Se recorta la tabla al periodo en que la estación objetivo tiene registros. Fuera de ese periodo,
cualquier valor del objetivo sería 100 % sintético. Las máscaras `es_train/es_val/es_test` usan
los cortes **base** (sin margen); el margen se aplica en el Paso 10, cuando ya se conoce τ₂.
"""),
    ("code", r'''
inicio, fin = ancho_full[TARGET].first_valid_index(), ancho_full[TARGET].last_valid_index()
ancho = ancho_full.loc[inicio:fin].copy()
es_train = ancho.index <= FIN_TRAIN
es_val = (ancho.index > FIN_TRAIN) & (ancho.index <= FIN_VAL)
es_test = ancho.index > FIN_VAL
print(f"Periodo del objetivo: {inicio} -> {fin} ({len(ancho):,} horas)")
print(f"Horas train/val/test (cortes base): {es_train.sum():,} / {es_val.sum():,} / {es_test.sum():,}")
'''),
    ("md", r"""
## Paso 2 — Estaciones candidatas

Criterios, en orden (filtro inicial; la decisión final la toma el **costo en horas** de la regla
de 72 h, Paso 5):
1. Subcuenca @@subcuenca@@ o vecinas inmediatas (@@vecinas_coma@@).
2. Para niveles, solo `Level_Avg`. `Caudal` se descarta porque se calcula del nivel con la curva
   de descarga y sería redundante.
3. En meteorología, solo los promedios (`TempAire_Avg`, `HumAire_Avg`, `RS_W_Avg`); mínimos y
   máximos son casi colineales con el promedio.
4. Como máximo 50 % de NaN en el periodo de entrenamiento.
5. Como máximo 50 % de NaN en validación y en prueba (solo se mira la **disponibilidad** del dato,
   nunca sus valores -> no hay fuga).

Las estaciones excluidas **siguen sirviendo como donantes** en la imputación (Pasos 6 a 8).
"""),
    ("code", r'''
filas = []
for c in ancho.columns:
    estacion, variable = c.split("__")
    pct_nan = float(ancho.loc[es_train, c].isna().mean())
    pct_nan_val = float(ancho.loc[es_val, c].isna().mean())
    pct_nan_test = float(ancho.loc[es_test, c].isna().mean())
    if c == TARGET:
        motivo = "OBJETIVO"
    elif subcuenca_de.get(estacion) not in SUBCUENCAS_ENTRADA:
        motivo = "fuera de subcuencas de entrada"
    elif variable == "Caudal":
        motivo = "caudal derivado del nivel"
    elif dominio(c) == "meteo" and variable not in VARIABLES_METEO_ENTRADA:
        motivo = "min/max redundante con el promedio"
    elif pct_nan > MAX_PCT_NAN_TRAIN:
        motivo = f"> {MAX_PCT_NAN_TRAIN:.0%} NaN en train"
    elif max(pct_nan_val, pct_nan_test) > MAX_PCT_NAN_VAL_TEST:
        motivo = f"> {MAX_PCT_NAN_VAL_TEST:.0%} NaN en val o test"
    else:
        motivo = "OK"
    filas.append({"columna": c, "subcuenca": subcuenca_de.get(estacion), "dominio": dominio(c),
                  "pct_nan_train": round(100 * pct_nan, 1), "pct_nan_val": round(100 * pct_nan_val, 1),
                  "pct_nan_test": round(100 * pct_nan_test, 1), "motivo": motivo})

seleccion = pd.DataFrame(filas).sort_values(["motivo", "dominio", "columna"])
CANDIDATAS = seleccion.loc[seleccion["motivo"].isin(["OK", "OBJETIVO"]), "columna"].tolist()

print(f"Columnas candidatas (incluido el objetivo): {len(CANDIDATAS)}")
display(seleccion[seleccion["motivo"].isin(["OK", "OBJETIVO"])])
print(seleccion["motivo"].value_counts())
'''),
    ("md", r"""
## Paso 3 — Control de calidad

- **Rangos físicos** (todas las columnas, incluidos los donantes): lluvia < 0, humedad fuera de
  [0, 105] (entre 100 y 105 se recorta a 100), temperatura fuera de [-15, 40] °C, radiación < 0
  (se recorta a 0) o > 1500 W/m², nivel ≤ 0. Esos valores pasan a NaN.
- **Radiación sin IQR.** La distribución de la radiación es muy asimétrica (la mitad de las horas
  son noche, con 0 W/m²), así que el IQR marcaba como atípicos los picos reales de mediodía
  (> 900 W/m² en Cebollar y Sayausí). Solo se aplica el límite físico de 0–1500 W/m² y un
  **chequeo nocturno**: en las horas que en train tienen radiación mediana nula (noche), un valor
  > 50 W/m² es un error de sensor y pasa a NaN. Las horas nocturnas se determinan con los datos
  y no con una zona horaria supuesta.
- **IQR (k = 3)**, con límites calculados **solo con train**: en temperatura y humedad el atípico
  pasa a NaN (error de sensor). En nivel y lluvia solo se marca y **no se toca**, porque puede ser
  la crecida que se quiere predecir. Para la lluvia el IQR se calcula sobre horas con lluvia > 0.
- **Niveles congelados.** Un nivel horario promedio que se repite **idéntico** durante 24 h o más no
  es físico (el sensor dejó de actualizar). Esas horas pasan a NaN en el objetivo y en los niveles
  vecinos, de modo que no se usan ni como etiqueta ni como entrada.
"""),
]
