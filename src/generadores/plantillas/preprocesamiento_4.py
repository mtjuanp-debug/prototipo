# -*- coding: utf-8 -*-
"""Plantilla del notebook de preprocesamiento (src/data_prep/preprocesamiento_<slug>.ipynb): parte 4 de 4.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("code", r'''
mf_final = MissForest(COLS_MF)
mf_final.fit_transform(df_train[COLS_MF])          # ajuste solo con train
mf_final_salida = mf_final.transform(ancho_limpio[COLS_MF])
print("missForest final ajustado con train y aplicado a todo el periodo.")

imputado = pd.DataFrame(index=ancho_limpio.index)
indicadores = pd.DataFrame(index=ancho_limpio.index)
interp_futuro = pd.DataFrame(False, index=ancho_limpio.index, columns=ENTRADAS)
hueco_largo = pd.DataFrame(False, index=ancho_limpio.index, columns=ENTRADAS)
filas = []
for col in ENTRADAS:
    dom = dominio(col)
    s = ancho_limpio[col]
    p = ajustar_parametros(col, df_train[col], df_train)
    falta = s.isna()
    largo, ini, fin_r, borde = rachas_nan(s)
    largo_s = pd.Series(largo, index=s.index)
    no_imputable = pd.Series((largo > MAX_HUECO_IMPUTABLE_H) | (falta.values & borde), index=s.index)
    hueco_largo[col] = no_imputable
    imputable = falta & ~no_imputable
    banda = pd.Series(np.where(largo <= HUECO_CORTO_H, "corto", "largo"), index=s.index)
    final = s.copy()
    metodo_usado = pd.Series(np.where(falta, "", "observado"), index=s.index, dtype=object)
    metodo_usado[no_imputable] = "no_imputado_hueco_largo"
    if dom == "nivel":
        # Anclas de cada hueco (valor observado antes y después) para la regla de climatología.
        ancla_izq = pd.Series(np.where(ini > 0, s.values[np.clip(ini - 1, 0, None)], np.nan), index=s.index)
        ancla_der = pd.Series(np.where((fin_r >= 0) & (fin_r < len(s) - 1),
                                       s.values[np.clip(fin_r + 1, 0, len(s) - 1)], np.nan), index=s.index)
        clima_ok = (largo_s >= 24) & (np.fmax(ancla_izq, ancla_der) < P95_TRAIN[col])
    cache = {}
    for b in ("corto", "largo"):
        en_banda = imputable & (banda == b)
        for m in RANKING[(dom, b)]:
            pendiente = en_banda & final.isna()
            if dom == "nivel" and m == "climatologia":
                pendiente &= clima_ok
            if not pendiente.any():
                continue
            if m not in cache:
                cache[m] = estimar(m, s, ancho_limpio, p, mf_final_salida, col)
            ok = pendiente & cache[m].notna()
            final[ok] = cache[m][ok]
            metodo_usado[ok] = m
    pendiente = imputable & final.isna()
    if pendiente.any():
        if dom == "lluvia":
            respaldo, nombre = pd.Series(0.0, index=s.index), "respaldo_cero"
        elif dom == "nivel":
            respaldo, nombre = interpolar(s, "linear"), "respaldo_lineal"
        else:
            respaldo, nombre = climatologia(s.index, p["climatologia"]), "respaldo_climatologia"
        final[pendiente] = respaldo[pendiente]
        metodo_usado[pendiente] = nombre

    if dom == "lluvia":
        final = final.clip(lower=0)
    elif col.split("__")[1].startswith("HumAire"):
        final = final.clip(0, 100)
    elif col.split("__")[1].startswith("RS_"):
        final = final.clip(lower=0)

    # Horas que dependen del ancla derecha de un hueco interpolado (información futura).
    por_interp = metodo_usado.isin(["lineal", "pchip", "respaldo_lineal"]).values
    futuro = por_interp.copy()
    # PCHIP estima la pendiente en el ancla derecha con el punto siguiente: la hora del ancla también depende del futuro.
    por_pchip = np.flatnonzero(metodo_usado.eq("pchip").values)
    if len(por_pchip):
        anclas = np.unique(fin_r[por_pchip] + 1)
        futuro[anclas[anclas < len(s)]] = True
    interp_futuro[col] = futuro

    imputado[col] = final
    indicadores[f"{col}__imp"] = falta.astype(np.int8)
    conteo = metodo_usado.value_counts(normalize=True) * 100
    fila = {"columna": col, "dominio": dom}
    fila.update({f"pct_{k}": round(v, 3) for k, v in conteo.items()})
    for nombre_part, mascara in (("train", es_train), ("val", es_val), ("test", es_test)):
        fila[f"pct_faltante_{nombre_part}"] = round(100 * falta[mascara].mean(), 2)
        fila[f"pct_imputado_{nombre_part}"] = round(100 * imputable[mascara].mean(), 2)
        fila[f"pct_hueco_largo_{nombre_part}"] = round(100 * no_imputable[mascara].mean(), 2)
    filas.append(fila)

reporte_imputacion = pd.DataFrame(filas).fillna(0.0)
reporte_imputacion.to_csv(OUT_DIR / "reporte_imputacion_@@slug@@.csv", index=False)

y_obs = ancho_limpio[TARGET].rename("y_obs")
assert imputado.isna().equals(hueco_largo), "Los NaN restantes deben coincidir con los huecos largos"
assert np.allclose(imputado.loc[y_obs.notna(), TARGET], y_obs.dropna()), "La imputación alteró valores observados"
print(f"Entradas imputadas: {imputado.shape[1]} columnas; NaN solo en huecos > 72 h.")
print(f"y_obs: {y_obs.notna().mean():.1%} de las horas con nivel observado real.")
display(reporte_imputacion)
'''),
    ("code", r'''
# Ejemplo visual: un hueco imputable del objetivo (24-72 h) y cómo se rellenó la entrada (y_obs queda vacío).
largo_obj, _, _, _ = rachas_nan(y_obs)
racha = pd.Series(largo_obj, index=y_obs.index)
candidatos = racha[(racha >= 24) & (racha <= MAX_HUECO_IMPUTABLE_H) & es_train]
if len(candidatos):
    t0 = candidatos.index[0]
    ventana = slice(t0 - pd.Timedelta(days=3), t0 + pd.Timedelta(hours=int(candidatos.iloc[0])) + pd.Timedelta(days=3))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(imputado.loc[ventana, TARGET], color="tab:orange", lw=1.2, label="Entrada imputada (TARGET)")
    ax.plot(y_obs.loc[ventana], color="tab:blue", lw=1.6, label="y_obs (observado; NaN en el hueco)")
    ax.set_title("Ejemplo de hueco imputable (≤ 72 h) en el nivel del @@nombre@@")
    ax.set_ylabel("Nivel @@nombre@@ (cm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    guardar_figura(fig, "fig_ejemplo_imputacion_objetivo_@@slug@@.png")
'''),
    ("md", r"""
## Paso 9 — Variables derivadas y columna `valido`

Se calculan con el módulo **`src/data_prep/variables_rio.py`** (función `construir_derivadas`), el mismo
que usa la aplicación de tiempo real (`calcular_variables`), sobre las series imputadas (los huecos > 72 h siguen
como NaN y se propagan a las variables que los usan):
- **Lluvia por pluviómetro (conjunto base):** acumulados móviles de 3, 6, 24 y 72 h.
- **Lluvia extendida (fase A):** por pluviómetro, acumulados de 2 y 12 h, máximo horario en 3 y 6 h e índice de
  precipitación antecedente API(t) = Σ₀⁷¹ 0,9ʲ·P(t−j). Además, la **lluvia areal** (media horaria de los
  pluviómetros de entrada) con sus rezagos de 1 a 6 h, acumulados de 2, 3, 6, 12, 24 y 72 h, máximos y API.
  El acumulado de 1 h es la propia lluvia horaria y no se repite.
- **Nivel:** rezagos de 1, 3, 6, 12 y 24 h, y cambios (deltas) en 1 h y 3 h.
- **Tiempo:** seno y coseno de la hora y del mes.

El catálogo marca con `conjunto` = `base` las variables de la corrida preliminar y con `lluvia_extendida` las
nuevas, para poder repetir la corrida base en la ablación.

Los rezagos y acumulados solo miran hacia atrás. Se descartan las primeras 72 h (calentamiento de
la ventana más larga).

**`valido` (int8) = 1** solo si en la hora t:
1. ninguna entrada usada (incluida toda su ventana de rezagos o acumulados, hasta 72 h) cae en un
   hueco de más de 72 h;
2. ninguna variable usa un valor **interpolado** cuya ancla derecha es posterior a t (sin
   información futura). Los métodos de donantes (CCW, IDW, regresión con vecina con k ≥ 0,
   missForest con valores simultáneos) y la climatología son causales y no invalidan la fila;
3. no queda ningún NaN en las entradas.
"""),
    ("code", r'''
CFG_VAR = vy.ConfigVariables(entradas=tuple(ENTRADAS), target=TARGET)
# La ventana de dependencia de la regla de 72 h (Paso 5) debe cubrir todas las variables del módulo
for c in ENTRADAS:
    assert vy.ventana_dependencia(c, CFG_VAR) == ventana_dependencia(c), c
derivadas, catalogo_features = vy.construir_derivadas(imputado, CFG_VAR)
tiempo = vy.codificacion_temporal(imputado.index)
# Comprobación: el preprocesamiento y la función de tiempo real producen exactamente las mismas columnas
_muestra = imputado.iloc[-500:].join(indicadores.iloc[-500:])
_rt = vy.calcular_variables(_muestra, CFG_VAR)
_pp = pd.concat([imputado, derivadas, tiempo, indicadores], axis=1).iloc[-500:]
assert list(_rt.columns) == list(_pp.columns), "Orden de columnas distinto entre preprocesamiento y tiempo real"
pd.testing.assert_frame_equal(_rt.iloc[VENTANA_MAX_H:], _pp.iloc[VENTANA_MAX_H:], check_dtype=False)
del _muestra, _rt, _pp

# Causas de invalidez
toca_hueco_largo = pd.Series(False, index=imputado.index)
for c in ENTRADAS:
    toca_hueco_largo |= INVALIDA[c]
usa_futuro = pd.Series(False, index=imputado.index)
for c in ENTRADAS:
    usa_futuro |= interp_futuro[c]

tabla = pd.concat([imputado, derivadas, tiempo, indicadores, y_obs], axis=1)
calentamiento = VENTANA_MAX_H
tabla = tabla.iloc[calentamiento:]
columnas_entrada = [c for c in tabla.columns if c != "y_obs"]
completo = tabla[columnas_entrada].notna().all(axis=1)
toca_hueco_largo = toca_hueco_largo.reindex(tabla.index)
usa_futuro = usa_futuro.reindex(tabla.index)
valido = (~toca_hueco_largo & ~usa_futuro & completo)
tabla["valido"] = valido.astype(np.int8)
assert not (completo.eq(False) & ~toca_hueco_largo).any(), "Hay NaN en entradas fuera de los huecos largos"

catalogo_base = vy.catalogo_base(CFG_VAR, list(indicadores.columns))
catalogo_control = pd.DataFrame([
    {"feature": "y_obs", "columna_base": TARGET, "dominio": "nivel", "tipo": "objetivo_observado",
     "ventana_horas": 0, "conjunto": "control"},
    {"feature": "valido", "columna_base": "-", "dominio": "control", "tipo": "fila_valida_regla_72h",
     "ventana_horas": VENTANA_MAX_H, "conjunto": "control"}])
catalogo_features = pd.concat([catalogo_base, catalogo_control, catalogo_features], ignore_index=True)

print(f"Tabla final: {tabla.shape[0]:,} horas x {tabla.shape[1]} columnas")
print(f"Filas válidas: {int(valido.sum()):,} ({valido.mean():.1%})")
print(f"  invalidadas por huecos > 72 h en la ventana: {int(toca_hueco_largo.sum()):,}")
print(f"  invalidadas por interpolación con ancla futura (y no por huecos largos): "
      f"{int((usa_futuro & ~toca_hueco_largo).sum()):,}")
display(catalogo_features.groupby(["conjunto", "dominio", "tipo"]).size().rename("n_columnas"))
'''),
    ("md", r"""
## Paso 10 — Partición cronológica con margen y verificación de eventos

- Cortes base 70/15/15 (los mismos del pipeline general).
- **Margen** = ventana máxima de rezagos (72 h) + τ₂ entre train/val y entre val/test. Las horas del
  margen se **eliminan** de las salidas (no pertenecen a ningún conjunto), de modo que ninguna
  variable derivada ni ninguna etiqueta `y_obs(t+τ₂)` de un conjunto use información del otro. El
  margen se coloca al inicio de val y de test; train conserva su corte.
- **Eventos de prealerta** (cruces de @@q_pre_g@@ m³/s por el nivel observado transformado a caudal, ≈ @@cm_pre_1@@ cm,
  agrupados si están a < 6 h):
  se exigen al menos 3 en val y en test. Si no se cumple, los cortes se desplazan hacia atrás una
  sola vez, conservando la proporción entre validación y prueba, y se documenta.
"""),
    ("code", r'''
def particionar(fin_train, fin_val):
    tr = tabla.loc[tabla.index <= fin_train].copy()
    va = tabla.loc[(tabla.index > fin_train + pd.Timedelta(hours=MARGEN_H)) & (tabla.index <= fin_val)].copy()
    te = tabla.loc[tabla.index > fin_val + pd.Timedelta(hours=MARGEN_H)].copy()
    return tr, va, te


def contar_eventos(d):
    return len(agrupar_eventos(d["y_obs"], UMBRAL_PREALERTA_CM))


train, val, test = particionar(FIN_TRAIN, FIN_VAL)
n_ev = {n: contar_eventos(d) for n, d in (("train", train), ("val", val), ("test", test))}
print("Eventos de prealerta por conjunto (cortes base):", n_ev)
cortes_desplazados = False
if min(n_ev["val"], n_ev["test"]) < MIN_EVENTOS_PARTICION:
    # Desplazamiento hacia atrás (una sola vez), conservando la proporción val/test.
    dur_val, dur_test = FIN_VAL - FIN_TRAIN, tabla.index.max() - FIN_VAL
    razon = dur_val / dur_test
    for dias in range(7, 3650, 7):
        nuevo_fin_val = FIN_VAL - pd.Timedelta(days=dias)
        nuevo_fin_train = nuevo_fin_val - (tabla.index.max() - nuevo_fin_val) * razon
        nuevo_fin_train = nuevo_fin_train.floor("h")
        tr, va, te = particionar(nuevo_fin_train, nuevo_fin_val)
        if min(contar_eventos(va), contar_eventos(te)) >= MIN_EVENTOS_PARTICION:
            FIN_TRAIN, FIN_VAL = nuevo_fin_train, nuevo_fin_val
            train, val, test = tr, va, te
            cortes_desplazados = True
            break
    n_ev = {n: contar_eventos(d) for n, d in (("train", train), ("val", val), ("test", test))}
    print(f"Cortes desplazados -> FIN_TRAIN = {FIN_TRAIN}, FIN_VAL = {FIN_VAL}; eventos: {n_ev}")
    # Tr se recalcula con el nuevo train para comprobar que τ₂ no cambia.
    ev_nuevo, _ = estimar_tr(FIN_TRAIN, umbral=umbral_eventos)
    print(f"Tr con el nuevo train: {ev_nuevo['Tr_h'].median():.2f} h (τ₂ fijado = {TAU2} h)")
else:
    print(f"Se cumplen >= {MIN_EVENTOS_PARTICION} eventos en val y test: no se desplazan los cortes.")

margen_1 = (val.index.min() - train.index.max()) / pd.Timedelta(hours=1) - 1
margen_2 = (test.index.min() - val.index.max()) / pd.Timedelta(hours=1) - 1
print(f"Horas excluidas como margen: train/val = {margen_1:.0f} h, val/test = {margen_2:.0f} h (requerido {MARGEN_H} h)")
'''),
    ("md", r"""
## Paso 11 — Escalado Min-Max, verificaciones y guardado

- **Escalado Min-Max:** mínimo y máximo calculados solo con las **filas válidas de train**; para el
  objetivo se usa la unión con todo el nivel observado de train (así `y_obs` de train queda en [0, 1]).
- **Sin escalar:** los indicadores `__imp` (0/1), la codificación cíclica ([-1, 1]) y `valido`.
- **`y_obs`:** se escala con el **mismo** mínimo y máximo que la columna objetivo, así un solo
  escalador sirve para volver a centímetros.
"""),
    ("code", r'''
sin_escalar = set(indicadores.columns) | set(tiempo.columns) | {"y_obs", "valido"}
train_valido = train["valido"] == 1
escaladores = []
for col in [c for c in tabla.columns if c not in sin_escalar]:
    mn, mx = train.loc[train_valido, col].min(), train.loc[train_valido, col].max()
    if col == TARGET:
        # El objetivo comparte escalador con y_obs: se usa la unión del nivel de las filas válidas y de
        # TODO el nivel observado de train, para que y_obs de train quede dentro de [0, 1].
        mn, mx = min(mn, train["y_obs"].min()), max(mx, train["y_obs"].max())
    rango = mx - mn
    escaladores.append({"columna": col, "metodo": "minmax", "min": mn, "max": mx})
    for d in (train, val, test):
        d[col] = 0.0 if (pd.isna(rango) or rango == 0) else (d[col] - mn) / rango

tgt = next(e for e in escaladores if e["columna"] == TARGET)
for d in (train, val, test):
    d["y_obs"] = (d["y_obs"] - tgt["min"]) / (tgt["max"] - tgt["min"])
escaladores.append({"columna": "y_obs", "metodo": "minmax", "min": tgt["min"], "max": tgt["max"]})
escaladores = pd.DataFrame(escaladores)

reporte_particion = pd.DataFrame([
    {"particion": n, "n_horas": len(d), "desde": d.index.min(), "hasta": d.index.max(),
     "pct": round(100 * len(d) / len(tabla), 1), "horas_con_y_obs": int(d["y_obs"].notna().sum()),
     "filas_validas": int(d["valido"].sum()), "pct_validas": round(100 * d["valido"].mean(), 1),
     "filas_validas_con_y_obs": int(((d["valido"] == 1) & d["y_obs"].notna()).sum()),
     "eventos_prealerta": n_ev[n]}
    for n, d in (("train", train), ("val", val), ("test", test))
])
display(reporte_particion)

filas = []
for n, d in (("train", train), ("val", val), ("test", test)):
    idx = d.index
    filas.append({"particion": n, "n_horas": len(d),
                  "invalidas_hueco_largo": int(toca_hueco_largo.reindex(idx).sum()),
                  "invalidas_solo_interp_futura": int((usa_futuro.reindex(idx) & ~toca_hueco_largo.reindex(idx)).sum()),
                  "validas": int(d["valido"].sum()), "pct_validas": round(100 * d["valido"].mean(), 2)})
reporte_validos = pd.DataFrame(filas)
display(reporte_validos)
'''),
    ("code", r'''
columnas_modelo = [c for c in tabla.columns if c not in ("y_obs", "valido")]
# 1) Sin solape, mismas columnas, margen suficiente.
assert train.index.max() < val.index.min() < test.index.min(), "Hay solape temporal entre particiones"
assert list(train.columns) == list(val.columns) == list(test.columns), "Columnas distintas entre particiones"
assert margen_1 >= MARGEN_H and margen_2 >= MARGEN_H, "Margen insuficiente entre particiones"
for nombre, d in (("train", train), ("val", val), ("test", test)):
    v = d["valido"] == 1
    # 2) Sin NaN en entradas de filas válidas.
    assert d.loc[v, columnas_modelo].notna().all().all(), f"NaN en entradas válidas de {nombre}"
    # 3) y_obs desescalado = nivel observado (tras control de calidad); NaN donde no hay observación.
    y_cm = d["y_obs"] * (tgt["max"] - tgt["min"]) + tgt["min"]
    obs = ancho_limpio.loc[d.index, TARGET]
    assert (y_cm.notna() == obs.notna()).all(), f"y_obs de {nombre} no coincide con lo observado"
    assert np.allclose(y_cm.dropna(), obs.dropna(), atol=1e-6), f"y_obs de {nombre} no reproduce el nivel en cm"
    # 4) Ninguna fila válida toca un hueco > 72 h (comprobación independiente con convolución).
    for c in ENTRADAS:
        g = mascara_hueco_largo(ancho_limpio[c]).astype(int).values
        w = ventana_dependencia(c)
        toca = np.convolve(g, np.ones(w + 1, dtype=int))[:len(g)] > 0
        toca = pd.Series(toca, index=ancho_limpio.index).reindex(d.index)
        assert not (toca & v).any(), f"Fila válida de {nombre} toca un hueco > 72 h de {c}"
    # 5) Eventos suficientes en val y test.
    if nombre != "train":
        assert n_ev[nombre] >= MIN_EVENTOS_PARTICION, f"Menos de {MIN_EVENTOS_PARTICION} eventos en {nombre}"
assert train.loc[train["valido"] == 1, [c for c in escaladores["columna"] if c != "y_obs"]].min().min() >= -1e-9
assert train["y_obs"].min() >= -1e-9 and train["y_obs"].max() <= 1 + 1e-9, "y_obs de train fuera de [0, 1]"
# 6) nRMSE de la validación normalizado por el rango de train.
assert np.allclose(validacion["nRMSE"], validacion["RMSE"] / validacion["rango_train"])
print("OK -- sin solape, margen >= ventana máxima + τ₂, sin NaN en entradas válidas, y_obs = nivel observado,"
      " ninguna fila válida toca un hueco > 72 h, >= 3 eventos en val y test, nRMSE por rango.")
'''),
    ("code", r'''
horizontes = {
    "Tr_h": round(TR_H, 3),
    "Tr_q25_h": round(float(q1_tr), 3),
    "Tr_q75_h": round(float(q3_tr), 3),
    "tau1": int(TAU1),
    "tau2": int(TAU2),
    "margen_h": int(MARGEN_H),
    "umbral_prealerta_m3s": Q_PREALERTA,
    "umbral_alerta_m3s": Q_ALERTA,
    "curva_descarga": CURVA,
    "umbral_prealerta_cm": round(UMBRAL_PREALERTA_CM, 2),
    "umbral_alerta_cm": round(UMBRAL_ALERTA_CM, 2),
    "ventana_max_h": int(VENTANA_MAX_H),
    "eventos": {"n_train": int(n_ev["train"]), "n_val": int(n_ev["val"]), "n_test": int(n_ev["test"]),
                "n_train_con_tr": int(len(tr_validos))},
    "cortes": {"train_fin": str(train.index.max()), "val_inicio": str(val.index.min()),
               "val_fin": str(val.index.max()), "test_inicio": str(test.index.min()),
               "test_fin": str(test.index.max()), "desplazados": bool(cortes_desplazados)},
    "metodo_tr": f"centro de masa de la lluvia areal de la subcuenca {SUBCUENCA_OBJETIVO} -> pico; "
                 f"ventana {VENTANA_TR_H} h, separación de tormentas {MIT_TR_H} h, lluvia mínima {MIN_LLUVIA_EVENTO_MM} mm",
    "estaciones_lluvia_areal": LLUVIA_CUENCA,
    "entradas": ENTRADAS,
    # Parámetros de src/data_prep/variables_rio.py (ConfigVariables.desde_horizontes_json los lee)
    "config_variables": CFG_VAR.a_dict(),
}
with open(OUT_DIR / "horizontes_@@slug@@.json", "w", encoding="utf-8") as f:
    json.dump(horizontes, f, ensure_ascii=False, indent=2)

train.to_parquet(OUT_DIR / "train_@@slug@@.parquet")
val.to_parquet(OUT_DIR / "val_@@slug@@.parquet")
test.to_parquet(OUT_DIR / "test_@@slug@@.parquet")
escaladores.to_csv(OUT_DIR / "escaladores_@@slug@@.csv", index=False)
catalogo_features.to_csv(OUT_DIR / "catalogo_features_@@slug@@.csv", index=False)
reporte_particion.to_csv(OUT_DIR / "reporte_particion_@@slug@@.csv", index=False)
reporte_validos.to_csv(OUT_DIR / "reporte_validos_@@slug@@.csv", index=False)

print(json.dumps({k: v for k, v in horizontes.items() if k not in ("entradas", "estaciones_lluvia_areal")},
                 ensure_ascii=False, indent=2))
print(f"\nArchivos guardados en {OUT_DIR}:")
generados = {*OUT_DIR.glob("*@@slug@@*"), *OUT_DIR.glob("auditoria_*")}
for f in sorted(generados):
    if f.is_file():
        print(f"  - {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
'''),
    ("md", r"""
## Cómo usar estos archivos en el entrenamiento

```python
import json
import pandas as pd

hz = json.load(open("Docs/resources/horizontes_@@slug@@.json", encoding="utf-8"))
train = pd.read_parquet("Docs/resources/train_@@slug@@.parquet")   # índice: TIMESTAMP (UTC), horario continuo
esc = pd.read_csv("Docs/resources/escaladores_@@slug@@.csv").set_index("columna")
TARGET = "@@target@@"
features = [c for c in train.columns if c not in ("y_obs", "valido")]   # lectura dinámica
for tau in sorted({hz["tau1"], hz["tau2"]}):
    d = train.copy()
    d["y"] = d["y_obs"].shift(-tau)                 # etiqueta SOLO con niveles observados
    d = d[(d["valido"] == 1) & d["y"].notna()]      # regla de 72 h + etiqueta observada
```

- Cada archivo es **horario y continuo** dentro de su conjunto (el `shift` es correcto); las filas
  con `valido = 0` pueden tener NaN en las entradas y deben descartarse.
- Entre conjuntos se eliminó un margen de `margen_h` horas (ventana máxima + τ₂).
- Para volver a centímetros, usar el `min`/`max` de `TARGET` (o de `y_obs`) en `escaladores_@@slug@@.csv`.
- **Importante:** `y_obs` y `valido` nunca deben entrar como variables de entrada.
- **Pendiente:** confirmar con ETAPA EP la correspondencia entre archivo y estación de los `.dat`
  de `preprocesados_base` (ver la auditoría del Paso 1).
"""),
]
