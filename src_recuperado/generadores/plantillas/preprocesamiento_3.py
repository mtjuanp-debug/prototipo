# -*- coding: utf-8 -*-
"""Plantilla del notebook de preprocesamiento (src/data_prep/preprocesamiento_<slug>.ipynb): parte 3 de 4.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("code", r'''
COLS_LLUVIA = [c for c in ancho_limpio.columns if dominio(c) == "lluvia"]
COLS_NIVEL = [c for c in ancho_limpio.columns if c.endswith("__Level_Avg")]
# Predictores del missForest: meteorología (promedios) de las subcuencas de entrada con <= 50 % NaN en train.
COLS_MF = [c for c in ancho_limpio.columns
           if dominio(c) == "meteo" and c.split("__")[1] in VARIABLES_METEO_ENTRADA
           and subcuenca_de.get(estacion_de(c)) in SUBCUENCAS_ENTRADA
           and ancho_limpio.loc[es_train, c].isna().mean() <= MAX_PCT_NAN_MF]
P95_TRAIN = {c: float(ancho_limpio.loc[es_train, c].quantile(0.95)) for c in COLS_NIVEL}
RANGO_TRAIN = {c: float(ancho_limpio.loc[es_train, c].max() - ancho_limpio.loc[es_train, c].min())
               for c in ancho_limpio.columns}
print("Predictores del missForest:", COLS_MF)

METODOS = {
    "nivel": ["lineal", "pchip", "regresion_vecina", "climatologia"],
    "lluvia": ["ccw", "idw", "lineal", "cero", "climatologia"],
    "meteo": ["missforest", "lineal", "climatologia"],
}
INTERPOLADORES = {"lineal": "linear", "pchip": "pchip"}


def ajustar_climatologia(s):
    obs = s.dropna()
    tabla = obs.groupby([obs.index.month, obs.index.hour]).mean()
    por_hora = obs.groupby(obs.index.hour).mean()
    return tabla, por_hora, obs.mean()


def climatologia(idx, params):
    tabla, por_hora, media = params
    v = tabla.reindex(pd.MultiIndex.from_arrays([idx.month, idx.hour])).values
    v = np.where(np.isnan(v), por_hora.reindex(idx.hour).values, v)
    return pd.Series(np.where(np.isnan(v), media, v), index=idx)


def traslape(a, b):
    return int((a.notna() & b.notna()).sum())


def donantes_ccw(col, s, df_tr):
    res = []
    for c in COLS_LLUVIA:
        if c == col or traslape(s, df_tr[c]) < MIN_TRASLAPE_H:
            continue
        r = s.corr(df_tr[c])
        if np.isfinite(r) and r > 0:
            res.append((c, r))
    res = sorted(res, key=lambda t: -t[1])[:N_DONANTES]
    return [c for c, _ in res], [r ** 2 for _, r in res]


def donantes_idw(col, df_tr):
    x0, y0 = COORDS.get(_norm(estacion_de(col)), (np.nan, np.nan))
    if np.isnan(x0):
        return [], []
    res = []
    for c in COLS_LLUVIA:
        xy = COORDS.get(_norm(estacion_de(c)))
        if c == col or xy is None or df_tr[c].notna().sum() < MIN_TRASLAPE_H:
            continue
        res.append((c, max(np.hypot(xy[0] - x0, xy[1] - y0), 1.0)))
    res = sorted(res, key=lambda t: t[1])[:N_DONANTES]
    return [c for c, _ in res], [1.0 / d ** 2 for _, d in res]


def ajustar_regresion_vecina(col, s, df_tr):
    """Para cada nivel vecino: desfase k in [0, MAX_LAG_VECINA_H] que maximiza corr(s(t), x(t-k));
    regresión lineal s(t) = a + b x(t-k). Se devuelven ordenadas por correlación."""
    res = []
    for c in COLS_NIVEL:
        if c == col or traslape(s, df_tr[c]) < MIN_TRASLAPE_H:
            continue
        corrs = {k: s.corr(df_tr[c].shift(k)) for k in range(0, MAX_LAG_VECINA_H + 1)}
        corrs = {k: r for k, r in corrs.items() if np.isfinite(r)}
        if not corrs:
            continue
        k = max(corrs, key=corrs.get)
        x = df_tr[c].shift(k)
        comun = s.notna() & x.notna()
        if comun.sum() < MIN_TRASLAPE_H or corrs[k] < MIN_CORR_VECINA:
            continue
        b, a = np.polyfit(x[comun].values, s[comun].values, 1)
        res.append((c, k, a, b, corrs[k]))
    return sorted(res, key=lambda t: -t[4])


class MissForest:
    """missForest (Stekhoven y Bühlmann, 2012) con IterativeImputer + ExtraTrees.
    Se ajusta con train y transforma cualquier periodo con los modelos ya ajustados."""

    def __init__(self, columnas):
        self.columnas = list(columnas)

    def _matriz(self, df):
        return pd.concat([df[self.columnas], codificacion_temporal(df.index)], axis=1)

    def fit_transform(self, df_tr):
        self.imp = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=MF_N_ESTIMADORES, min_samples_leaf=5, max_features=0.5,
                                          n_jobs=-1, random_state=SEED),
            max_iter=MF_MAX_ITER, random_state=SEED)
        # Nota: no usar keep_empty_features=True; en sklearn 1.5 hace que fit_transform devuelva
        # la imputación inicial (media) en lugar de la iterativa. Ninguna columna de COLS_MF está vacía en train.
        X = self._matriz(df_tr)
        salida = self.imp.fit_transform(X.values)
        return pd.DataFrame(salida[:, :len(self.columnas)], index=df_tr.index, columns=self.columnas)

    def transform(self, df):
        salida = self.imp.transform(self._matriz(df).values)
        return pd.DataFrame(salida[:, :len(self.columnas)], index=df.index, columns=self.columnas)


def ajustar_parametros(col, s, df_tr):
    # s: serie de la columna en train (puede venir con huecos artificiales ya aplicados)
    dom = dominio(col)
    p = {"climatologia": ajustar_climatologia(s)}
    if dom == "lluvia":
        p["ccw"] = donantes_ccw(col, s, df_tr)
        p["idw"] = donantes_idw(col, df_tr)
    elif dom == "nivel":
        p["regresion_vecina"] = ajustar_regresion_vecina(col, s, df_tr)
    return p


def interpolar(serie, como):
    v = pd.Series(serie.values).interpolate(method=como, limit_area="inside")
    return pd.Series(v.values, index=serie.index)


def ponderado(df, donantes, pesos):
    if not donantes:
        return pd.Series(np.nan, index=df.index)
    V, W = df[donantes].values, np.asarray(pesos)
    disponible = ~np.isnan(V)
    num = np.nansum(V * W, axis=1)
    den = (disponible * W).sum(axis=1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=df.index)


def estimar(metodo, serie, df, p, mf_salida=None, col=None):
    # Devuelve la estimación del método para todas las horas de df (NaN donde no puede estimar).
    if metodo in INTERPOLADORES:
        return interpolar(serie, INTERPOLADORES[metodo])
    if metodo == "climatologia":
        return climatologia(df.index, p["climatologia"])
    if metodo == "cero":
        return pd.Series(0.0, index=df.index)
    if metodo in ("ccw", "idw"):
        return ponderado(df, *p[metodo])
    if metodo == "regresion_vecina":
        est = pd.Series(np.nan, index=df.index)
        for donante, k, a, b, _ in p["regresion_vecina"]:
            est = est.fillna(a + b * df[donante].shift(k))
        return est
    if metodo == "missforest":
        return mf_salida[col].reindex(df.index)
    raise ValueError(metodo)
'''),
    ("md", r"""
## Paso 7 — Validación con huecos artificiales (solo train)

1. Para cada entrada se eligen al azar bloques de horas **observadas** de 1, 6, 24 y 72 h
   (60 bloques por longitud, sin solaparse, con una hora observada a cada lado).
   - **Lluvia:** la mitad de los bloques se toma de periodos con lluvia acumulada ≥ 0,5 mm; si no,
     casi todos caerían en horas secas y el método "cero" ganaría de forma engañosa.
   - **Meteorología:** los bloques se eligen **por estación** y se borran **a la vez todas sus
     variables** (temperatura, humedad, radiación), que es lo que ocurre cuando una estación deja de
     transmitir. Así el missForest no puede apoyarse en las otras variables de la misma estación.
2. Se borran esos valores y los parámetros de cada método se ajustan **sin ellos**.
3. Se imputa con cada método y se compara contra el valor real: MAE, RMSE, sesgo, **nMAE y nRMSE
   normalizados por el rango** (máx − mín de los valores observados en train), como exige la
   tesis. En la lluvia se evalúa además la detección de horas con/sin lluvia (≥ 0,1 mm):
   acierto, POD y FAR.
4. La climatología del nivel solo se evalúa en huecos de 24 y 72 h cuyas anclas están por debajo
   del P95 (fuera de crecidas), que es el único caso en que se permite usarla.
5. Si un método no puede estimar una hora (sin donantes) se usa climatología y se reporta la
   **cobertura**.

El ranking por **tipo de variable × banda de hueco** (corto ≤ 6 h, largo 7–72 h) se hace con el
nRMSE, que penaliza más los errores grandes (los picos). En la lluvia, como la tesis **prioriza CCW**,
otro método solo lo reemplaza si su nRMSE es más de 5 % menor; en ese caso CCW pasa a ser el respaldo.
"""),
    ("code", r'''
def muestrear_huecos(s, L, n, exigir_lluvia=False):
    ok = s.notna().values.astype(int)
    validos = np.convolve(ok, np.ones(L + 2, dtype=int), "valid") == L + 2
    inicios = np.flatnonzero(validos) + 1
    if exigir_lluvia:
        acumulado = np.convolve(np.nan_to_num(s.values), np.ones(L), "valid")
        inicios = inicios[acumulado[inicios] >= 0.5]
    rng.shuffle(inicios)
    elegidos, ocupado = [], np.zeros(len(s), dtype=bool)
    for i in inicios:
        if len(elegidos) >= n:
            break
        if not ocupado[max(0, i - 1):i + L + 1].any():
            ocupado[i:i + L] = True
            elegidos.append(np.arange(i, i + L))
    return elegidos


def metricas(est, verdad, rango):
    err = est - verdad
    mae, rmse = np.abs(err).mean(), np.sqrt((err ** 2).mean())
    return {"MAE": mae, "RMSE": rmse, "sesgo": err.mean(), "nMAE": mae / rango, "nRMSE": rmse / rango}


def metricas_lluvia(est, verdad):
    e, o = est >= UMBRAL_LLUVIA_MM, verdad >= UMBRAL_LLUVIA_MM
    aciertos, fallos, falsas = (e & o).sum(), (~e & o).sum(), (e & ~o).sum()
    return {"acierto_lluvia_no_lluvia": float((e == o).mean()),
            "POD": aciertos / (aciertos + fallos) if aciertos + fallos else np.nan,
            "FAR": falsas / (aciertos + falsas) if aciertos + falsas else np.nan}


df_train = ancho_limpio.loc[es_train]
ENTRADAS_METEO = [c for c in ENTRADAS if dominio(c) == "meteo"]
ESTACIONES_METEO = sorted({estacion_de(c) for c in ENTRADAS_METEO})

# --- Huecos artificiales -------------------------------------------------------
huecos = {}            # (columna, L) -> lista de bloques (arrays de posiciones)
for col in [c for c in ENTRADAS if dominio(c) != "meteo"]:
    s = df_train[col]
    for L in LONGITUDES_HUECO:
        if dominio(col) == "lluvia":
            mitad = N_HUECOS_POR_LONGITUD // 2
            huecos[(col, L)] = (muestrear_huecos(s, L, mitad, exigir_lluvia=True)
                                + muestrear_huecos(s, L, N_HUECOS_POR_LONGITUD - mitad))
        else:
            huecos[(col, L)] = muestrear_huecos(s, L, N_HUECOS_POR_LONGITUD)
for est in ESTACIONES_METEO:
    cols_est = [c for c in ENTRADAS_METEO if estacion_de(c) == est]
    todas = df_train[cols_est].notna().all(axis=1)
    s_est = pd.Series(np.where(todas, 1.0, np.nan), index=df_train.index)
    for L in LONGITUDES_HUECO:
        bloques = muestrear_huecos(s_est, L, N_HUECOS_POR_LONGITUD)
        for c in cols_est:
            huecos[(c, L)] = bloques

# --- missForest con todos los bloques meteorológicos borrados a la vez ---------
df_train_mf = df_train[COLS_MF].copy()
for c in ENTRADAS_METEO:
    pos = np.concatenate([np.concatenate(huecos[(c, L)]) for L in LONGITUDES_HUECO if huecos[(c, L)]])
    df_train_mf.iloc[pos, df_train_mf.columns.get_loc(c)] = np.nan
mf_valid = MissForest(COLS_MF)
mf_valid_salida = mf_valid.fit_transform(df_train_mf)
print("missForest de validación ajustado.")

filas = []
for col in ENTRADAS:
    dom = dominio(col)
    s = df_train[col]
    rango = RANGO_TRAIN[col]
    todas_pos = [np.concatenate(huecos[(col, L)]) for L in LONGITUDES_HUECO if huecos[(col, L)]]
    union = np.unique(np.concatenate(todas_pos))
    s_ajuste = s.copy()
    s_ajuste.iloc[union] = np.nan
    p = ajustar_parametros(col, s_ajuste, df_train)
    fijos = {m: estimar(m, None, df_train, p, mf_valid_salida, col)
             for m in METODOS[dom] if m not in INTERPOLADORES}
    respaldo = fijos.get("climatologia", climatologia(df_train.index, p["climatologia"]))
    for L in LONGITUDES_HUECO:
        bloques = huecos[(col, L)]
        if not bloques:
            continue
        pos = np.concatenate(bloques)
        s_hueco = s.copy()
        s_hueco.iloc[pos] = np.nan
        for m in METODOS[dom]:
            pos_m = pos
            if dom == "nivel" and m == "climatologia":
                if L < 24:
                    continue
                # solo bloques fuera de crecidas: ambas anclas < P95 de train
                pos_m = np.concatenate([b for b in bloques
                                        if max(s.iloc[b[0] - 1], s.iloc[b[-1] + 1]) < P95_TRAIN[col]] or [np.array([], int)])
                if len(pos_m) == 0:
                    continue
            verdad = s.iloc[pos_m].values
            est = (interpolar(s_hueco, INTERPOLADORES[m]) if m in INTERPOLADORES else fijos[m]).iloc[pos_m].values
            cobertura = float(np.isfinite(est).mean())
            est = np.where(np.isfinite(est), est, respaldo.iloc[pos_m].values)
            if dom == "lluvia":
                est = np.clip(est, 0, None)
            fila = {"columna": col, "dominio": dom, "longitud_h": L, "metodo": m, "n_horas": len(pos_m),
                    "rango_train": rango, **metricas(est, verdad, rango), "cobertura": cobertura}
            if dom == "lluvia":
                fila.update(metricas_lluvia(est, verdad))
            filas.append(fila)

validacion = pd.DataFrame(filas)
validacion["banda"] = np.where(validacion["longitud_h"] <= HUECO_CORTO_H, "corto", "largo")
validacion.to_csv(OUT_DIR / "reporte_validacion_imputacion_@@slug@@.csv", index=False)
print(f"{len(validacion):,} evaluaciones (columna x longitud x método); nMAE y nRMSE normalizados por el rango de train.")
'''),
    ("code", r'''
resumen_validacion = (validacion.groupby(["dominio", "banda", "metodo"])
                      .agg(nRMSE=("nRMSE", "mean"), nMAE=("nMAE", "mean"), sesgo=("sesgo", "mean"),
                           cobertura=("cobertura", "mean"), n_columnas=("columna", "nunique"))
                      .reset_index().sort_values(["dominio", "banda", "nRMSE"]))
lluvia_det = (validacion[validacion["dominio"] == "lluvia"]
              .groupby(["dominio", "banda", "metodo"])[["acierto_lluvia_no_lluvia", "POD", "FAR"]].mean())
resumen_validacion = resumen_validacion.merge(lluvia_det.reset_index(), on=["dominio", "banda", "metodo"], how="left")
RANKING = {(d, b): g.sort_values("nRMSE")["metodo"].tolist()
           for (d, b), g in resumen_validacion.groupby(["dominio", "banda"])}
# Métodos que no aparecen en una banda (p. ej., climatología del nivel en huecos cortos) van al final como respaldo.
for (d, b), orden in RANKING.items():
    orden += [m for m in METODOS[d] if m not in orden and not (d == "nivel" and m == "climatologia")]
# Prioridad de CCW en la lluvia (tesis): solo se desplaza si otro método mejora el nRMSE en más de TOLERANCIA_CCW.
for b in ("corto", "largo"):
    if ("lluvia", b) in RANKING:
        g = resumen_validacion[(resumen_validacion["dominio"] == "lluvia") & (resumen_validacion["banda"] == b)]
        n = g.set_index("metodo")["nRMSE"]
        orden = RANKING[("lluvia", b)]
        if orden[0] != "ccw" and n[orden[0]] >= (1 - TOLERANCIA_CCW) * n["ccw"]:
            orden.remove("ccw")
            orden.insert(0, "ccw")
            print(f"lluvia {b}: {orden[1]} mejora a CCW en menos de {TOLERANCIA_CCW:.0%} -> se mantiene CCW como principal")
resumen_validacion.to_csv(OUT_DIR / "resumen_validacion_imputacion_@@slug@@.csv", index=False)

display(resumen_validacion.round(4))
print("\nMétodo elegido (y orden de respaldo) por tipo de variable y banda de hueco:")
for (d, b), orden in RANKING.items():
    print(f"  {d:7s} {b:6s} -> {' > '.join(orden)}")

por_longitud = (validacion.groupby(["dominio", "longitud_h", "metodo"])["nRMSE"].mean().reset_index())
display(por_longitud.pivot_table(index=["dominio", "metodo"], columns="longitud_h", values="nRMSE").round(4))
display(validacion[validacion["dominio"] == "lluvia"]
        .pivot_table(index="metodo", columns="longitud_h", values=["acierto_lluvia_no_lluvia", "POD", "FAR"]).round(3))
'''),
    ("code", r'''
dominios = ["nivel", "lluvia", "meteo"]
fig, ejes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
for ax, d in zip(ejes, dominios):
    tabla = por_longitud[por_longitud["dominio"] == d].pivot(index="longitud_h", columns="metodo", values="nRMSE")
    tabla.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(f"{d.capitalize()}")
    ax.set_xlabel("Longitud del hueco artificial (h)")
    ax.set_ylabel("nRMSE (RMSE / rango observado en train)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Validación con huecos artificiales (periodo de entrenamiento) — menor es mejor")
fig.tight_layout()
guardar_figura(fig, "fig_validacion_imputacion_@@slug@@.png")
'''),
    ("md", r"""
## Paso 8 — Imputación final (solo huecos ≤ 72 h) con columna indicadora

Para cada entrada:
1. Se mide cada racha de NaN. Las de **más de 72 h** (o que tocan un extremo de la serie) **no se
   imputan**: quedan como NaN y sus horas dependientes tendrán `valido = 0` (Paso 9).
2. Las demás se clasifican como cortas (≤ 6 h) o largas (7–72 h) y se rellenan con el método ganador
   de su banda; si no puede estimar una hora se pasa al siguiente del ranking. La climatología del
   nivel solo se usa en huecos de 24–72 h con ambas anclas por debajo del P95 de train.
3. Último recurso (solo si nada pudo estimar): lluvia = 0, nivel = interpolación lineal,
   meteorología = climatología de train.
4. **`<columna>__imp`** = 1 si el dato original faltaba (imputado, o NaN en un hueco largo) y 0 si
   fue observado.
5. Se registra qué horas se rellenaron por **interpolación** (lineal o PCHIP): esos valores usan la
   observación posterior al hueco (ancla derecha), que en tiempo real aún no existe.

**Variable objetivo:**
- `@@target@@` se imputa como **entrada** (en huecos ≤ 72 h), porque los
  rezagos del propio nivel necesitan una serie continua.
- **`y_obs`** es el nivel **observado** después del control de calidad, sin imputar (NaN en los
  huecos y en los tramos congelados). El entrenamiento construye `y = y_obs.shift(-τ)`.
"""),
]
