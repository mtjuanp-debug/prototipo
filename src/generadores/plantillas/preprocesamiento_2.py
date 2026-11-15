# -*- coding: utf-8 -*-
"""Plantilla del notebook de preprocesamiento (src/data_prep/preprocesamiento_<slug>.ipynb): parte 2 de 4.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("code", r'''
def limpiar_rangos(df):
    df = df.copy()
    conteo = {}
    for c in df.columns:
        variable = c.split("__")[1]
        s = df[c]
        if variable == "Lluvia_Tot":
            malo = s < 0
        elif variable.startswith("HumAire"):
            s = s.where(~((s > 100) & (s <= 105)), 100.0)
            malo = (s < 0) | (s > 105)
        elif variable.startswith("TempAire"):
            malo = (s < -15) | (s > 40)
        elif variable.startswith("RS_"):
            s = s.clip(lower=0)
            malo = s > 1500
        elif variable == "Level_Avg":
            malo = s <= 0
        else:
            malo = pd.Series(False, index=s.index)
        df[c] = s.mask(malo)
        conteo[c] = int(malo.sum())
    return df, conteo


def limites_iqr(s_train, es_lluvia, k=K_IQR):
    base = s_train.dropna()
    if es_lluvia:
        base = base[base > 0]
    if len(base) < 10:
        return None
    q1, q3 = base.quantile(0.25), base.quantile(0.75)
    iqr = q3 - q1
    return (0.0 if es_lluvia else q1 - k * iqr), q3 + k * iqr


def rachas_congeladas(s, min_h=MIN_RACHA_CONGELADA_H):
    """True en las horas que pertenecen a una racha de valores idénticos (no NaN) de >= min_h horas."""
    grupo = (s != s.shift()).cumsum()
    tam = s.groupby(grupo).transform("size")
    return (tam >= min_h) & s.notna()


ancho_limpio, fuera_de_rango = limpiar_rangos(ancho)

# Horas nocturnas: radiación mediana nula en train (se usa el conjunto de estaciones con radiación).
cols_rs = [c for c in ancho_limpio.columns if c.split("__")[1].startswith("RS_") and c.endswith("_Avg")]
mediana_horaria = ancho_limpio.loc[es_train, cols_rs].median(axis=1).groupby(ancho_limpio.index[es_train].hour).median()
HORAS_NOCHE = sorted(int(h) for h in mediana_horaria.index[mediana_horaria <= 1.0])
print("Horas (índice del registro) con radiación mediana nula en train:", HORAS_NOCHE)
es_noche = np.isin(ancho_limpio.index.hour, HORAS_NOCHE)

filas = []
for c in ancho_limpio.columns:
    dom = dominio(c)
    variable = c.split("__")[1]
    s = ancho_limpio[c]
    fila = {"columna": c, "dominio": dom, "fuera_de_rango": fuera_de_rango[c], "n_noche_rs": 0,
            "n_atipicos_iqr": 0, "lim_inf": np.nan, "lim_sup": np.nan, "n_congelado": 0, "accion": ""}
    if variable.startswith("RS_"):
        malo = es_noche & (s > UMBRAL_RS_NOCHE)
        ancho_limpio[c] = s.mask(malo)
        fila.update(n_noche_rs=int(malo.sum()), accion="radiación: límite físico + chequeo nocturno (sin IQR)")
    else:
        lim = limites_iqr(ancho_limpio.loc[es_train, c], dom == "lluvia")
        if lim is not None:
            atipico = (s > lim[1]) if dom == "lluvia" else ((s < lim[0]) | (s > lim[1]))
            fila.update(n_atipicos_iqr=int(atipico.sum()), lim_inf=round(lim[0], 3), lim_sup=round(lim[1], 3))
            if dom == "meteo":
                ancho_limpio[c] = s.mask(atipico)
                fila["accion"] = "IQR: anulado (error de sensor)"
            else:
                fila["accion"] = "IQR: solo marcado (posible evento real)"
    if variable == "Level_Avg":
        congelado = rachas_congeladas(ancho_limpio[c])
        ancho_limpio[c] = ancho_limpio[c].mask(congelado)
        fila["n_congelado"] = int(congelado.sum())
        fila["accion"] += " + niveles congelados -> NaN"
    filas.append(fila)

reporte_limpieza = pd.DataFrame(filas)
reporte_limpieza.to_csv(OUT_DIR / "reporte_limpieza_@@slug@@.csv", index=False)
display(reporte_limpieza[reporte_limpieza["columna"].isin(CANDIDATAS)])
'''),
    ("code", r'''
# Reporte de niveles congelados por partición (cortes base).
filas = []
for c in [c for c in ancho.columns if c.endswith("__Level_Avg")]:
    congelado = ancho[c].notna() & ancho_limpio[c].isna() & ~(ancho[c] <= 0)
    grupo = (ancho[c] != ancho[c].shift()).cumsum()
    for nombre, m in (("train", es_train), ("val", es_val), ("test", es_test)):
        sel = congelado & m
        filas.append({"columna": c, "particion": nombre, "horas_congeladas": int(sel.sum()),
                      "n_rachas": int(grupo[sel].nunique())})
reporte_congelados = pd.DataFrame(filas)
reporte_congelados.to_csv(OUT_DIR / "reporte_congelados_@@slug@@.csv", index=False)
display(reporte_congelados[reporte_congelados["horas_congeladas"] > 0])
congelado_obj = ancho[TARGET].notna() & ancho_limpio[TARGET].isna()
print(f"Horas del objetivo anuladas por congelamiento: {int(congelado_obj.sum())} "
      f"(train {int((congelado_obj & es_train).sum())}, val {int((congelado_obj & es_val).sum())}, "
      f"test {int((congelado_obj & es_test).sum())})")
'''),
    ("md", r"""
## Paso 4 — Tiempo de respuesta y horizontes de pronóstico

La tesis define los horizontes de cada río a partir de su **tiempo de respuesta** (Tr, *lag time*):
el intervalo entre el **centro de masa de la lluvia** y el **pico del hidrograma**, estimado con los
registros del periodo de **entrenamiento** y fijado **antes** de entrenar:
- τ₂ = ⌊Tr⌋ (máxima anticipación alcanzable sin pronóstico de lluvia),
- τ₁ = max(1, round(τ₂ / 2)); si τ₁ = τ₂ se evalúa un único horizonte.

Procedimiento:
1. **Eventos**: cruces del umbral de prealerta (@@q_pre_g@@ m³/s) por el nivel **observado transformado a caudal**
   (equivale a @@cm_pre_1@@ cm, porque la curva de descarga es creciente) en
   train, agrupando cruces separados por menos de 6 h. Si hubiera menos de 10 eventos se usaría el
   P95 de train.
2. **Lluvia areal**: media aritmética de las estaciones pluviométricas de la **subcuenca @@subcuenca@@**
   disponibles en cada hora (no hay polígonos de Thiessen en el catálogo). Se exige al menos una
   estación con dato en toda la ventana.
3. **Lluvia causante**: dentro de las 24 h previas al pico, la tormenta empieza después del último
   periodo seco de ≥ 6 h (criterio usual de separación de tormentas). Si la lluvia areal de la
   tormenta es < 1 mm, el evento no es interpretable (lluvia muy localizada o no registrada) y se
   descarta.
4. **Tr** = hora del pico − centro de masa de la lluvia de la tormenta.

Se reporta la mediana y el rango intercuartílico, una prueba de sensibilidad a la ventana y al
tiempo de separación, y la correlación cruzada lluvia–nivel como contraste independiente.
"""),
    ("code", r'''
LLUVIA_CUENCA = [c for c in ancho_limpio.columns
                 if c.endswith("__Lluvia_Tot") and subcuenca_de.get(estacion_de(c)) == SUBCUENCA_OBJETIVO
                 and ancho_limpio.loc[es_train, c].notna().any()]
lluvia_areal = ancho_limpio[LLUVIA_CUENCA].mean(axis=1)
n_estaciones_areal = ancho_limpio[LLUVIA_CUENCA].notna().sum(axis=1)
display(pd.DataFrame({
    "estacion_lluvia": LLUVIA_CUENCA,
    "pct_dato_train": [round(100 * ancho_limpio.loc[es_train, c].notna().mean(), 1) for c in LLUVIA_CUENCA],
}))


def estimar_tr(fin_train, ventana_h=VENTANA_TR_H, mit_h=MIT_TR_H, min_lluvia=MIN_LLUVIA_EVENTO_MM,
               umbral=None, detalle=True):
    y_tr = ancho_limpio.loc[:fin_train, TARGET]
    if umbral is None:
        umbral = UMBRAL_PREALERTA_CM
    eventos = agrupar_eventos(y_tr, umbral)
    filas = []
    for ev in eventos.itertuples():
        tp = ev.t_pico
        ventana = lluvia_areal.loc[tp - pd.Timedelta(hours=ventana_h):tp]
        fila = {"inicio": ev.inicio, "t_pico": tp, "pico_cm": ev.pico, "Tr_h": np.nan,
                "lluvia_mm": np.nan, "inicio_lluvia": pd.NaT, "centro_masa": pd.NaT, "motivo": "OK"}
        if n_estaciones_areal.loc[ventana.index].min() < 1:
            fila["motivo"] = "sin lluvia areal en la ventana"
            filas.append(fila)
            continue
        v = ventana.values
        humedo = v >= UMBRAL_LLUVIA_MM
        inicio_tormenta, seco = 0, 0
        for j in range(len(v) - 1, -1, -1):
            if humedo[j]:
                seco = 0
            else:
                seco += 1
                if seco >= mit_h and humedo[j:].any():
                    inicio_tormenta = j + seco
                    break
        tormenta = ventana.iloc[inicio_tormenta:]
        total = float(tormenta.sum())
        fila["lluvia_mm"] = total
        if total < min_lluvia:
            fila["motivo"] = f"lluvia areal < {min_lluvia} mm"
            filas.append(fila)
            continue
        horas = np.arange(len(tormenta))
        cm = float((tormenta.values * horas).sum() / total)
        fila["Tr_h"] = (len(tormenta) - 1) - cm
        fila["inicio_lluvia"] = tormenta.index[0]
        fila["centro_masa"] = tormenta.index[0] + pd.Timedelta(hours=cm)
        filas.append(fila)
    return pd.DataFrame(filas), umbral


eventos_tr, umbral_eventos = estimar_tr(FIN_TRAIN)
if len(eventos_tr) < MIN_EVENTOS_UMBRAL:
    p95 = float(ancho_limpio.loc[es_train, TARGET].quantile(0.95))
    print(f"Solo {len(eventos_tr)} eventos de prealerta en train: se usa el P95 ({p95:.1f} cm).")
    eventos_tr, umbral_eventos = estimar_tr(FIN_TRAIN, umbral=p95)

tr_validos = eventos_tr["Tr_h"].dropna()
TR_H = float(tr_validos.median())
q1_tr, q3_tr = tr_validos.quantile([0.25, 0.75])
print(f"Eventos de prealerta en train: {len(eventos_tr)} | con Tr estimable: {len(tr_validos)}")
print(eventos_tr["motivo"].value_counts().to_string())
print(f"Tr: mediana = {TR_H:.2f} h | RIC = [{q1_tr:.2f}, {q3_tr:.2f}] h | "
      f"mín = {tr_validos.min():.2f} h | máx = {tr_validos.max():.2f} h")
eventos_tr.to_csv(OUT_DIR / "eventos_tiempo_respuesta_@@slug@@.csv", index=False)
'''),
    ("code", r'''
# Sensibilidad del Tr a la ventana y al tiempo mínimo entre tormentas (solo informativa).
filas = []
for ventana_h in (12, 24, 48):
    for mit_h in (3, 6):
        ev, _ = estimar_tr(FIN_TRAIN, ventana_h=ventana_h, mit_h=mit_h, umbral=umbral_eventos)
        t = ev["Tr_h"].dropna()
        filas.append({"ventana_h": ventana_h, "separacion_tormentas_h": mit_h, "n_eventos": len(t),
                      "Tr_mediana_h": round(t.median(), 2), "Tr_q25_h": round(t.quantile(0.25), 2),
                      "Tr_q75_h": round(t.quantile(0.75), 2),
                      "principal": (ventana_h == VENTANA_TR_H and mit_h == MIT_TR_H)})
sensibilidad_tr = pd.DataFrame(filas)
sensibilidad_tr.to_csv(OUT_DIR / "sensibilidad_tr_@@slug@@.csv", index=False)
display(sensibilidad_tr)

# Contraste independiente: desfase de máxima correlación entre la lluvia areal y el nivel (train).
y_tr = ancho_limpio.loc[es_train, TARGET]
p_tr = lluvia_areal[es_train]
ccf = pd.Series({k: p_tr.shift(k).corr(y_tr) for k in range(0, 25)})
ccf_subida = pd.Series({k: p_tr.shift(k).corr(y_tr.diff().clip(lower=0)) for k in range(0, 25)})
print(f"Desfase de máxima correlación lluvia -> nivel: {int(ccf.idxmax())} h (r = {ccf.max():.3f})")
print(f"Desfase de máxima correlación lluvia -> subida del nivel: {int(ccf_subida.idxmax())} h "
      f"(r = {ccf_subida.max():.3f}; marca el inicio de la subida, no el pico)")
'''),
    ("code", r'''
TAU2 = int(math.floor(TR_H))
nota_tau = ""
if TAU2 < 1:
    TAU2 = 1
    nota_tau = "Tr < 1 h: se fija τ₂ = 1 h (resolución horaria)."
TAU1 = max(1, int(math.floor(TAU2 / 2 + 0.5)))   # redondeo "half-up", no bancario
MARGEN_H = VENTANA_MAX_H + TAU2
HORIZONTES = sorted({TAU1, TAU2})
print(f"Tr = {TR_H:.2f} h  ->  τ₂ = {TAU2} h, τ₁ = {TAU1} h {nota_tau}")
print(f"Horizontes a evaluar: {HORIZONTES} | τ/Tr = {[round(t / TR_H, 2) for t in HORIZONTES]}")
print(f"Margen entre particiones = ventana máxima ({VENTANA_MAX_H} h) + τ₂ ({TAU2} h) = {MARGEN_H} h")
print("Contraste con la literatura: para el Tomebamba (Matadero-Sayausí) se estima Tr ≈ 1-4 h "
      "(0,6 Tc; Muñoz et al., 2021; Contreras et al., 2021). @@frase_literatura@@")
'''),
    ("code", r'''
# Figura: hidrogramas de los eventos con mayor pico + distribución de Tr.
con_tr = eventos_tr.dropna(subset=["Tr_h"]).sort_values("pico_cm", ascending=False).head(5)
fig, ejes = plt.subplots(2, 3, figsize=(17, 8))
for ax, ev in zip(ejes.ravel()[:5], con_tr.itertuples()):
    ventana = slice(ev.t_pico - pd.Timedelta(hours=30), ev.t_pico + pd.Timedelta(hours=18))
    ax.plot(ancho_limpio.loc[ventana, TARGET], color="tab:blue", lw=1.6, label="Nivel observado")
    ax.axhline(UMBRAL_PREALERTA_CM, color="tab:orange", ls="--", lw=1, label="Prealerta")
    ax.axhline(UMBRAL_ALERTA_CM, color="tab:red", ls="--", lw=1, label="Alerta")
    ax.axvline(ev.t_pico, color="tab:blue", ls=":", lw=1)
    ax.axvline(ev.centro_masa, color="tab:green", ls=":", lw=1.2, label="Centro de masa lluvia")
    ax.set_title(f"{ev.t_pico:%Y-%m-%d %H:%M} | Tr = {ev.Tr_h:.1f} h", fontsize=10)
    ax.set_ylabel("Nivel (cm)")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax2 = ax.twinx()
    ax2.bar(lluvia_areal.loc[ventana].index, lluvia_areal.loc[ventana].values, width=1 / 24,
            color="tab:gray", alpha=0.6)
    ax2.set_ylim(max(1.0, lluvia_areal.loc[ventana].max()) * 3, 0)
    ax2.set_ylabel("Lluvia areal (mm/h)")
    ax.grid(alpha=0.3)
ejes.ravel()[0].legend(fontsize=7, loc="center left")
ax = ejes.ravel()[5]
ax.hist(tr_validos, bins=np.arange(0, tr_validos.max() + 2, 1), color="tab:blue", alpha=0.7)
ax.axvline(TR_H, color="k", lw=1.5, label=f"Mediana = {TR_H:.2f} h")
ax.axvspan(q1_tr, q3_tr, color="tab:blue", alpha=0.15, label="RIC")
ax.set_xlabel("Tr (h)")
ax.set_ylabel("Eventos")
ax.set_title(f"Tiempo de respuesta en train (n = {len(tr_validos)})")
ax.legend(fontsize=8)
fig.suptitle("@@nombre@@ — eventos de prealerta en train y tiempo de respuesta (centro de masa de la lluvia → pico)")
fig.tight_layout()
guardar_figura(fig, "fig_tiempo_respuesta_@@slug@@.png")
'''),
    ("md", r"""
## Paso 5 — Regla de 72 h y selección de estaciones por costo en horas

**Regla de la tesis:** los vacíos de más de 72 h no se imputan y los periodos correspondientes se
excluyen del entrenamiento y la evaluación. Como las variables derivadas miran hacia atrás, un hueco
largo no solo invalida sus propias horas: también todas las horas cuya **ventana de dependencia**
lo toca.

| Dominio | Variables que se derivan | Ventana de dependencia en t |
|---|---|---|
| Lluvia | acumulados de 3, 6, 24 y 72 h | t−71 … t |
| Nivel | rezagos de 1–24 h y deltas de 1 y 3 h | t−24 … t |
| Meteorología | valor en t | t |

Un hueco que toca el inicio o el final del periodo también se trata como largo (no tiene anclas a
ambos lados).

**Selección por costo:** cada entrada candidata reduce las horas útiles. El objetivo ya invalida
sus propias horas (sus huecos largos), así que el costo de una entrada es el **% de horas de train
que siguen siendo útiles con el objetivo pero que esa entrada invalidaría**. Procedimiento voraz:
se ordenan las candidatas por costo individual y se agregan una a una; una entrada se conserva si
su **costo marginal** (respecto del conjunto ya aceptado) es ≤ 5 %. Solo se usa la disponibilidad
del dato en train, nunca sus valores. El costo en val y test se reporta como información.

**Regla de cuenca:** si ningún pluviómetro de la propia cuenca supera el criterio, se conserva el de
menor costo individual. La lluvia de la cuenca es la señal que anticipa la crecida y la tesis asocia
a cada río las estaciones de su cuenca (§1.3, P45); sin ella el modelo solo vería lluvia de cuencas
vecinas.
"""),
    ("code", r'''
def mascara_hueco_largo(s, max_h=MAX_HUECO_IMPUTABLE_H):
    largo, _, _, borde = rachas_nan(s)
    return pd.Series((largo > max_h) | ((largo > 0) & borde), index=s.index)


def ventana_dependencia(col):
    dom = dominio(col)
    if dom == "lluvia":
        return max(VENTANAS_LLUVIA_HORAS) - 1
    if dom == "nivel":
        return max(max(LAGS_NIVEL_HORAS), max(DELTAS_NIVEL_HORAS))
    return 0


def invalida_por(col):
    g = mascara_hueco_largo(ancho_limpio[col]).astype(int)
    w = ventana_dependencia(col)
    return g.rolling(w + 1, min_periods=1).max().astype(bool)


INVALIDA = {c: invalida_por(c) for c in CANDIDATAS}
base_invalida = INVALIDA[TARGET]
utiles_train = (~base_invalida & es_train)
print(f"Horas de train invalidadas solo por los huecos largos del objetivo: "
      f"{int((base_invalida & es_train).sum()):,} de {int(es_train.sum()):,} "
      f"({100 * (base_invalida & es_train).mean() / es_train.mean():.1f} %)")

costos = []
for c in [c for c in CANDIDATAS if c != TARGET]:
    extra = INVALIDA[c] & ~base_invalida
    fila = {"columna": c, "estacion": estacion_de(c), "dominio": dominio(c),
            "pct_nan_train": round(100 * ancho_limpio.loc[es_train, c].isna().mean(), 1),
            "pct_hueco_largo_train": round(100 * mascara_hueco_largo(ancho_limpio[c])[es_train].mean(), 1),
            "ventana_dependencia_h": ventana_dependencia(c),
            "costo_individual_pct": 100 * (extra & es_train).sum() / utiles_train.sum()}
    for nombre, m in (("val", es_val), ("test", es_test)):
        utiles = ~base_invalida & m
        fila[f"costo_individual_{nombre}_pct"] = round(100 * (extra & m).sum() / max(utiles.sum(), 1), 2)
    costos.append(fila)
costos = pd.DataFrame(costos).sort_values(["costo_individual_pct", "columna"]).reset_index(drop=True)

aceptadas, invalida_actual = [], base_invalida.copy()
marginal, decision, acumulado = [], [], []
for c in costos["columna"]:
    extra = INVALIDA[c] & ~invalida_actual
    costo_m = 100 * (extra & es_train).sum() / utiles_train.sum()
    marginal.append(costo_m)
    if costo_m <= 100 * MAX_COSTO_EXTRA:
        aceptadas.append(c)
        invalida_actual |= INVALIDA[c]
        decision.append("CONSERVADA")
    else:
        decision.append(f"EXCLUIDA (costo marginal > {MAX_COSTO_EXTRA:.0%})")
    acumulado.append(100 * ((invalida_actual & ~base_invalida) & es_train).sum() / utiles_train.sum())

# Regla de cuenca: se conserva siempre al menos un pluviómetro de la propia cuenca (el de menor costo
# individual), porque la lluvia de la cuenca es la que anticipa la crecida (Tesis §1.3, P45).
def es_lluvia_cuenca(c):
    return dominio(c) == "lluvia" and subcuenca_de.get(estacion_de(c)) == SUBCUENCA_OBJETIVO

if not any(es_lluvia_cuenca(c) for c in aceptadas):
    cand_cuenca = [c for c in costos["columna"] if es_lluvia_cuenca(c)]
    if cand_cuenca:
        c = cand_cuenca[0]                      # costos ya está ordenado por costo individual
        i = costos.index[costos["columna"] == c][0]
        aceptadas.append(c)
        invalida_actual |= INVALIDA[c]
        decision[i] = "CONSERVADA (regla de cuenca: pluviómetro propio)"
        print(f"Regla de cuenca: se conserva {c} (costo marginal {marginal[i]:.2f} %)")
costo_total = 100 * ((invalida_actual & ~base_invalida) & es_train).sum() / utiles_train.sum()
print(f"Costo acumulado final en train: {costo_total:.2f} %")
costos["costo_marginal_pct"] = marginal
costos["costo_acumulado_pct"] = acumulado
costos["decision"] = decision
costos[["costo_individual_pct", "costo_marginal_pct", "costo_acumulado_pct"]] = \
    costos[["costo_individual_pct", "costo_marginal_pct", "costo_acumulado_pct"]].round(2)
costos.to_csv(OUT_DIR / "costo_estaciones_@@slug@@.csv", index=False)

ENTRADAS = [TARGET] + aceptadas
seleccion["motivo_final"] = seleccion["motivo"]
for fila in costos.itertuples():
    seleccion.loc[seleccion["columna"] == fila.columna, "motivo_final"] = (
        "OK" if fila.decision.startswith("CONSERVADA") else "excluida por costo en horas (regla 72 h)")
seleccion.to_csv(OUT_DIR / "seleccion_estaciones_@@slug@@.csv", index=False)

display(costos)
print(f"Entradas finales ({len(ENTRADAS)}):")
for c in ENTRADAS:
    print("  -", c)
'''),
    ("md", r"""
## Paso 6 — Métodos de imputación candidatos (parámetros ajustados solo con train)

| Tipo de variable | Métodos | Detalle |
|---|---|---|
| Nivel | Interpolación lineal, PCHIP, regresión con vecina, climatología | La vecina se elige por correlación cruzada con desfase k ≥ 0 (la vecina en t−k explica el nivel en t). La climatología hora × mes solo se admite en huecos de **24–72 h fuera de crecidas** (ambas anclas del hueco por debajo del P95 de train). |
| Lluvia | **CCW** (principal), **IDW** (alternativa), lineal, cero, climatología (referencias) | CCW: pesos = r² con hasta 4 vecinas; IDW: pesos = 1/d². |
| Meteorología | **missForest iterativo**, interpolación lineal, climatología | `IterativeImputer` + `ExtraTreesRegressor` (50 árboles, 10 iteraciones) con la temperatura, humedad y radiación de las estaciones de las subcuencas de entrada con ≤ 50 % de NaN en train (incluidas las excluidas como entrada) + codificación cíclica de hora y mes. |

Todos los parámetros (climatologías, correlaciones, desfases, regresiones, bosques) se ajustan
**solo con train** y luego se aplican sin cambios a val y test. Ningún método se usa en huecos de
más de 72 h.
"""),
]
