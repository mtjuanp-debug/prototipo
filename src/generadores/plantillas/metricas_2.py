# -*- coding: utf-8 -*-
"""Plantilla del notebook de métricas (src/metrics/<slug>_models_metric.ipynb): parte 2 de 3.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("md", r"""
## 2. Walk-forward y selección del modelo (P55–P56)

Validación progresiva con ventana expansiva en 5 bloques de val. Para cada familia se toma la
configuración con mayor SS medio entre folds **en cada horizonte** (igual que el entrenamiento). El SS de la
familia es la media de los SS por fold y horizonte (5 folds × 2 horizontes). Si la diferencia con la mejor
familia es menor de 0,01, se prefiere la de menor RMSE en niveles altos (> P95 de train): RMSE de las
predicciones walk-forward (folds unidos) de cada horizonte, promediado entre horizontes. Es el mismo cálculo
de `seleccion_modelo.csv` del entrenamiento; ambas tablas lo documentan en las columnas `definicion_*`.
"""),
    ("code", r'''
def ss_por_modelo_en(df):
    # SS de cada modelo y horizonte sobre un conjunto de predicciones con persistencia en las mismas horas
    rmse = df.groupby(["modelo", "horizonte"]).apply(lambda g: np.sqrt(np.mean((g["y_pred_cm"] - g["y_real_cm"]) ** 2)))
    rmse = rmse.rename("RMSE").reset_index()
    pers = rmse[rmse["modelo"] == "Persistencia"].set_index("horizonte")["RMSE"]
    rmse["SS"] = 1 - rmse["RMSE"] / rmse["horizonte"].map(pers)
    return rmse


def rmse_alto(df, modelo):
    g = df[df["modelo"] == modelo]
    vals = [metricas_niveles_altos(x["y_real_cm"].to_numpy(), x["y_pred_cm"].to_numpy())["RMSE_alto"]
            for _, x in g.groupby("horizonte")]
    return np.nanmean(vals) if len(vals) and not np.all(np.isnan(vals)) else np.nan


ss_fold, seleccion, MEJOR_CONFIG = None, None, None
if busqueda is not None and len(busqueda):
    b = busqueda.copy()
    if "modelo" not in b.columns:
        b["modelo"] = b["familia"]
    # El entrenamiento elige una configuración por familia y horizonte (aunque los candidatos sean comunes)
    compartida = False
    claves = ["familia", "horizonte", "config_id"]
    agg = b.groupby(claves).agg(SS_medio=("SS", "mean"), RMSE_medio=("RMSE", "mean"), n_folds=("fold", "nunique"),
                                config_json=("config_json", "first"), modelo=("modelo", "first")).reset_index()
    MEJOR_CONFIG = agg.loc[agg.groupby(claves[:-1])["SS_medio"].idxmax()].reset_index(drop=True)
    sel_b = b.merge(MEJOR_CONFIG[claves], on=claves)
    ss_fold = sel_b.pivot_table(index=["familia", "horizonte"], columns="fold", values="SS").reset_index()
    ss_fold.columns = [f"SS_fold{c}" if isinstance(c, (int, np.integer)) else c for c in ss_fold.columns]
    cols_f = [c for c in ss_fold.columns if str(c).startswith("SS_fold")]
    ss_fold["SS_medio"] = ss_fold[cols_f].mean(axis=1)
    ss_fold["SS_desv"] = ss_fold[cols_f].std(axis=1)
    seleccion = ss_fold.pivot(index="familia", columns="horizonte", values="SS_medio")
    seleccion.columns = [f"SS_medio_h{h}" for h in seleccion.columns]
    seleccion["SS_medio"] = seleccion.mean(axis=1)
    seleccion["modelo_busqueda"] = sel_b.groupby("familia")["modelo"].first()
    print(f"Configuraciones {'comunes a ambos horizontes' if compartida else 'elegidas por horizonte'}; "
          f"{b['config_id'].nunique()} ids, {b['fold'].nunique()} folds.")
elif len(pred_wf):
    print("[aviso] Sin busqueda_resultados.csv: la selección usa el SS de las predicciones walk-forward agregadas.")
    s = ss_por_modelo_en(pred_wf)
    s = s[s["modelo"] != "Persistencia"]
    s["familia"] = s["modelo"].map(FAMILIA_DE)
    s = s.loc[s.groupby(["familia", "horizonte"])["SS"].idxmax()]
    seleccion = s.pivot(index="familia", columns="horizonte", values="SS")
    seleccion.columns = [f"SS_medio_h{h}" for h in seleccion.columns]
    seleccion["SS_medio"] = seleccion.mean(axis=1)
    seleccion["modelo_busqueda"] = s.groupby("familia")["modelo"].first()
else:
    print("[aviso] Sin búsqueda ni walk-forward: la selección usa el SS en val del modelo final (dentro de muestra si se reentrenó con val).")
    s = ss_por_modelo_en(pred_final[pred_final["particion"] == "val"])
    s = s[s["modelo"] != "Persistencia"]
    s["familia"] = s["modelo"].map(FAMILIA_DE)
    s = s.loc[s.groupby(["familia", "horizonte"])["SS"].idxmax()]
    seleccion = s.pivot(index="familia", columns="horizonte", values="SS")
    seleccion.columns = [f"SS_medio_h{h}" for h in seleccion.columns]
    seleccion["SS_medio"] = seleccion.mean(axis=1)
    seleccion["modelo_busqueda"] = s.groupby("familia")["modelo"].first()


def modelo_final(fam, h):
    cands = [m for m in MODELOS if FAMILIA_DE.get(m) == fam and (m, "test", h) in DATOS]
    if not cands:
        return None
    pref = seleccion["modelo_busqueda"].get(fam) if seleccion is not None else None
    if pref in cands or len(cands) == 1:
        return pref if pref in cands else cands[0]
    base = pred_wf if len(pred_wf) else pred_final[pred_final["particion"] == "val"]
    s = ss_por_modelo_en(base[base["horizonte"] == h])
    s = s[s["modelo"].isin(cands)]
    return s.loc[s["SS"].idxmax(), "modelo"] if len(s) else cands[0]


MODELO_FINAL = {(f, h): modelo_final(f, h) for f in FAMILIAS for h in HORIZONTES}
base_alto = pred_wf if len(pred_wf) else pred_final[pred_final["particion"] == "val"]
seleccion = seleccion.reindex([f for f in FAMILIAS if f in seleccion.index])
seleccion["RMSE_alto_cm"] = [np.nanmean([rmse_alto(base_alto[base_alto["horizonte"] == h], MODELO_FINAL[(f, h)])
                                         for h in HORIZONTES]) for f in seleccion.index]
mejor_ss = seleccion["SS_medio"].max()
seleccion["dif_vs_mejor"] = mejor_ss - seleccion["SS_medio"]
seleccion["en_empate"] = seleccion["dif_vs_mejor"] < TOL_SS
empatados = seleccion[seleccion["en_empate"]]
FAMILIA_SEL = empatados["RMSE_alto_cm"].idxmin() if empatados["RMSE_alto_cm"].notna().any() else seleccion["SS_medio"].idxmax()
seleccion["seleccionado"] = seleccion.index == FAMILIA_SEL
seleccion["regla"] = np.where(seleccion["seleccionado"],
                              "desempate por RMSE en niveles altos" if len(empatados) > 1 else "mayor SS medio", "")
MODELO_SEL = {h: MODELO_FINAL[(FAMILIA_SEL, h)] for h in HORIZONTES}
MEJORES_POR_H = {h: ["Persistencia"] + [MODELO_FINAL[(f, h)] for f in FAMILIAS if MODELO_FINAL[(f, h)]] for h in HORIZONTES}
MEJORES = MEJORES_POR_H[TAU1 if TAU1 in HORIZONTES else HORIZONTES[0]]

seleccion["definicion_SS"] = "media de SS por fold y horizonte (mejor configuración de cada horizonte)"
seleccion["definicion_RMSE_alto"] = (f"RMSE walk-forward en y_obs > P95 ({P95:.1f} cm; {DEF_P95}), folds unidos, "
                                     "media de horizontes")
seleccion.round(4).to_csv(TAB_DIR / "seleccion_walkforward.csv")
if ss_fold is not None:
    ss_fold.round(4).to_csv(TAB_DIR / "ss_por_fold.csv", index=False)
    MEJOR_CONFIG.to_csv(TAB_DIR / "mejor_configuracion_por_familia.csv", index=False)
    display(ss_fold.round(4))
display(seleccion.round(4))
print(f"Familia seleccionada para el @@nombre@@: {FAMILIA_SEL} -> modelos finales {MODELO_SEL}")
for clave in ("seleccion_global", "modelo_seleccionado"):
    if clave in CONFIG:
        print(f"Selección registrada por el notebook de entrenamiento ({clave}):", CONFIG[clave])
if (MOD_DIR / "seleccion_modelo.csv").exists():
    print("seleccion_modelo.csv del entrenamiento (para contrastar con la tabla anterior):")
    display(pd.read_csv(MOD_DIR / "seleccion_modelo.csv").round(4))
'''),
    ("code", r'''
# G14 — SS por fold del walk-forward (mejor configuración de cada familia)
if ss_fold is not None:
    cols_f = [c for c in ss_fold.columns if str(c).startswith("SS_fold")]
    fig, ejes = plt.subplots(1, len(HORIZONTES), figsize=(7 * len(HORIZONTES), 4.5), sharey=True)
    ejes = np.atleast_1d(ejes)
    for ax, h in zip(ejes, HORIZONTES):
        sub = ss_fold[ss_fold["horizonte"] == h].set_index("familia")
        for fam in sub.index:
            estilo = {"lw": 2.6} if fam == FAMILIA_SEL else {"lw": 1.2}
            ax.plot(range(1, len(cols_f) + 1), sub.loc[fam, cols_f], "o-", label=f"{fam} (media {sub.loc[fam, 'SS_medio']:.3f})", **estilo)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        if np.nanmin(sub[cols_f].to_numpy(dtype=float)) < -1:
            ax.set_ylim(bottom=-1)
            ax.text(0.01, 0.02, "eje recortado en SS = −1", transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_xticks(range(1, len(cols_f) + 1))
        ax.set_xlabel("Fold (bloque de validación)")
        ax.set_title(f"Horizonte {etiqueta_h(h)}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ejes[0].set_ylabel("SS = 1 − RMSE/RMSE persistencia")
    fig.suptitle("G14 — Índice de habilidad por fold del walk-forward (0 = persistencia)", fontsize=13)
    fig.tight_layout()
    guardar_figura(fig, "G14_ss_por_fold.png")
else:
    print("Sin resultados de búsqueda: se omite G14.")
'''),
    ("md", r"""
## 3. Tabla completa de métricas (modelo × horizonte × partición)

`val_fold` agrupa las predicciones fuera de muestra del walk-forward (los 5 bloques). Si el modelo
final se reentrenó con train + val, sus métricas en `val` son dentro de muestra; la evaluación fuera de
muestra es `val_fold` (selección) y `test` (contraste final).
"""),
    ("code", r'''
grupos = list(pred_final.groupby(["modelo", "familia", "horizonte", "particion"], sort=False))
if len(pred_wf):
    grupos += list(pred_wf.groupby(["modelo", "familia", "horizonte", "particion"], sort=False))
filas, EVENTOS_PICO = [], []
for (modelo, familia, h, part), g in grupos:
    g = g.sort_values("t_obj")
    o, s = g["y_real_cm"].to_numpy(), g["y_pred_cm"].to_numpy()
    fila = {"modelo": modelo, "familia": familia, "horizonte": h, "particion": part, "n_horas": len(g)}
    fila.update(metricas_regresion(o, s))
    fila.update(metricas_niveles_altos(o, s))
    ev = pd.concat([eventos_pico(x["t_obj"], x["y_real_cm"].to_numpy(), x["y_pred_cm"].to_numpy())
                    for _, x in g.groupby("fold", dropna=False)], ignore_index=True) if part == "val_fold" else \
        eventos_pico(g["t_obj"], o, s)
    fila.update(resumen_picos(ev))
    if len(ev):
        EVENTOS_PICO.append(ev.assign(modelo=modelo, familia=familia, horizonte=h, particion=part))
    filas.append(fila)
metricas = pd.DataFrame(filas)
rmse_pers = metricas[metricas["modelo"] == "Persistencia"].set_index(["horizonte", "particion"])["RMSE"]
metricas["SS"] = 1 - metricas["RMSE"] / metricas.set_index(["horizonte", "particion"]).index.map(rmse_pers)
EVENTOS_PICO = pd.concat(EVENTOS_PICO, ignore_index=True) if EVENTOS_PICO else pd.DataFrame()
metricas.to_csv(TAB_DIR / "metricas_completas.csv", index=False)
EVENTOS_PICO.to_csv(TAB_DIR / "eventos_crecida_por_modelo.csv", index=False)
print(f"Tabla de métricas: {metricas.shape[0]} filas x {metricas.shape[1]} columnas -> {TAB_DIR / 'metricas_completas.csv'}")
'''),
    ("code", r'''
COLS_PRINCIPALES = ["SS", "RMSE", "MAE", "sesgo", "NSE", "KGE", "KGE_r", "KGE_alfa", "KGE_beta", "PBIAS", "R2"]
COLS_ALTOS = ["n_alto", "MAE_alto", "RMSE_alto", "sesgo_alto"]
for h in HORIZONTES:
    for part in ["val_fold", "test"]:
        t = metricas[(metricas["horizonte"] == h) & (metricas["particion"] == part)].set_index("modelo")
        if len(t):
            print(f"\n=== Horizonte {etiqueta_h(h)} — {part.upper()} ===")
            display(t[COLS_PRINCIPALES + COLS_ALTOS].sort_values("SS", ascending=False).round(3))
'''),
    ("md", r"""
## 4. Contraste con la persistencia en test: prueba de Diebold–Mariano (P58, H1 de P70)

- Pérdida cuadrática: dₜ = e²_modelo,t − e²_persistencia,t sobre las mismas horas de test.
- Estadístico DM = d̄ / √(V̂/n), con V̂ = γ̂₀ + 2 Σₖ₌₁ᴸ (1 − k/(L + 1)) γ̂ₖ (Newey–West, núcleo de
  Bartlett). Rezago L = max(τ − 1, ⌈n^{1/3}⌉): τ − 1 es el mínimo teórico para pronósticos a τ pasos,
  pero los errores horarios del nivel siguen autocorrelacionados más allá, por lo que se usa la regla
  ⌈n^{1/3}⌉. Las autocovarianzas se calculan sobre la grilla horaria (los huecos no se unen).
- Corrección de Harvey, Leybourne y Newbold: DM* = DM·√((n + 1 − 2τ + τ(τ − 1)/n)/n), comparado con
  una t de Student con n − 1 grados de libertad.
- H1 unilateral: el modelo tiene menor error cuadrático que la persistencia (d̄ < 0).
- Corrección de Holm sobre las comparaciones río-horizonte del modelo seleccionado. Aquí son las 2 del
  @@nombre@@; en la tesis el equipo la aplica sobre hasta 8 (cuatro ríos × dos horizontes), por lo que los
  p-valores sin ajustar de esta tabla deben reunirse con los de los otros ríos para la decisión final.
- Como información complementaria se contrasta también el modelo final de cada familia, con Holm sobre
  todas esas comparaciones (no forma parte de la hipótesis).
"""),
    ("code", r'''
def dm_modelo(modelo, h, part="test"):
    gm, gp = DATOS[(modelo, part, h)], DATOS[("Persistencia", part, h)]
    comunes = gm.index.intersection(gp.index)
    em = (gm.loc[comunes, "y_pred_cm"] - gm.loc[comunes, "y_real_cm"]).to_numpy()
    ep = (gp.loc[comunes, "y_pred_cm"] - gp.loc[comunes, "y_real_cm"]).to_numpy()
    r = diebold_mariano(em, ep, comunes, h)
    r["SS"] = 1 - np.sqrt(np.mean(em ** 2)) / np.sqrt(np.mean(ep ** 2))
    return r


filas = []
for h in HORIZONTES:
    for f in FAMILIAS:
        m = MODELO_FINAL[(f, h)]
        if m is not None:
            filas.append({"familia": f, "modelo": m, "horizonte": h, "seleccionado": f == FAMILIA_SEL, **dm_modelo(m, h)})
dm_todos = pd.DataFrame(filas)
dm_todos["p_holm_familias"] = holm(dm_todos["p_unilateral"])
dm_sel = dm_todos[dm_todos["seleccionado"]].copy()
dm_sel["p_holm"] = holm(dm_sel["p_unilateral"])
dm_sel["rechaza_H0_5pct"] = (dm_sel["SS"] > 0) & (dm_sel["p_holm"] < ALFA_DM)
dm_sel.to_csv(TAB_DIR / "diebold_mariano.csv", index=False)
dm_todos.to_csv(TAB_DIR / "diebold_mariano_familias.csv", index=False)
print("Modelo seleccionado frente a la persistencia (test):")
display(dm_sel.drop(columns="seleccionado").round(5))
print("Complementario: modelo final de cada familia frente a la persistencia (test):")
display(dm_todos.round(5))
'''),
    ("code", r'''
# G16 — Diebold–Mariano (estadístico con corrección HLN) por familia y horizonte
fig, ejes = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = ejes[0]
x = np.arange(len(FAMILIAS))
ancho = 0.8 / len(HORIZONTES)
crit = stats.t.ppf(ALFA_DM, df=max(dm_todos["n"].min() - 1, 1))
for i, h in enumerate(HORIZONTES):
    sub = dm_todos[dm_todos["horizonte"] == h].set_index("familia").reindex(FAMILIAS)
    barras = ax.bar(x + (i - (len(HORIZONTES) - 1) / 2) * ancho, sub["DM_HLN"], ancho, label=etiqueta_h(h))
    for bx, (fam, r) in zip(barras, sub.iterrows()):
        if pd.notna(r["DM_HLN"]):
            ax.annotate(f"p={r['p_holm_familias']:.3f}", (bx.get_x() + bx.get_width() / 2, r["DM_HLN"]),
                        xytext=(0, -10 if r["DM_HLN"] < 0 else 3), textcoords="offset points", ha="center", fontsize=6)
ax.axhline(crit, color="crimson", ls="--", lw=0.9, label=f"valor crítico unilateral 5 % ({crit:.2f})")
ax.axhline(0, color="black", lw=0.8)
lim = np.nanmax(np.abs(dm_todos["DM_HLN"].to_numpy(dtype=float)))
ax.set_ylim(min(-3, -1.15 * lim), max(3, 1.15 * lim))
ax.set_xticks(x, [f"{f}{' ★' if f == FAMILIA_SEL else ''}" for f in FAMILIAS])
ax.set_ylabel("DM* (negativo = mejor que persistencia)")
ax.set_title("Estadístico DM* en test (p con Holm entre familias)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax = ejes[1]
ax.axis("off")
tabla = dm_sel[["modelo", "horizonte", "SS", "DM_HLN", "p_unilateral", "p_holm", "rechaza_H0_5pct"]].copy()
tabla["horizonte"] = tabla["horizonte"].map(etiqueta_h)
celdas_txt = [[r["modelo"], r["horizonte"], f"{r['SS']:.3f}", f"{r['DM_HLN']:.2f}", f"{r['p_unilateral']:.2e}",
               f"{r['p_holm']:.2e}", "sí" if r["rechaza_H0_5pct"] else "no"] for _, r in tabla.iterrows()]
tb = ax.table(cellText=celdas_txt, colLabels=["Modelo", "Horizonte", "SS", "DM*", "p", "p Holm", "SS>0 y p<0,05"],
              loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(8)
tb.scale(1, 1.6)
ax.set_title(f"Modelo seleccionado ({FAMILIA_SEL}) frente a la persistencia")
fig.suptitle("G16 — Prueba de Diebold–Mariano (HAC Newey–West + HLN), conjunto de prueba", fontsize=13)
fig.tight_layout()
guardar_figura(fig, "G16_diebold_mariano.png")
'''),
    ("md", r"""
## 5. Crecidas y niveles altos (test)

Eventos de crecida = episodios con nivel observado ≥ umbral de prealerta (agrupación < 6 h). Los
niveles altos son las horas con nivel observado > P95 de train.
"""),
    ("code", r'''
cols_picos = ["n_eventos", "error_pico_medio_cm", "error_pico_abs_medio_cm", "error_pico_rel_medio_pct",
              "pct_picos_subestimados", "desfase_medio_h", "desfase_abs_medio_h"]
for h in HORIZONTES:
    t = metricas[(metricas["horizonte"] == h) & (metricas["particion"] == "test")].set_index("modelo").loc[MEJORES_POR_H[h]]
    print(f"\n=== Crecidas y niveles altos en test — {etiqueta_h(h)} (umbral de evento {Q_PRE:.0f} m³/s ≈ {U_PRE:.1f} cm; P95 = {P95:.1f} cm) ===")
    display(t[cols_picos + COLS_ALTOS].round(2))
if len(EVENTOS_PICO):
    det = EVENTOS_PICO[(EVENTOS_PICO["particion"] == "test") & EVENTOS_PICO["modelo"].isin(set(sum(MEJORES_POR_H.values(), [])))]
    det.to_csv(TAB_DIR / "eventos_crecida_test.csv", index=False)
    print("Detalle por evento (modelo seleccionado):")
    display(det[det["modelo"].isin(MODELO_SEL.values())].round(2))
'''),
    ("md", r"""
## 6. Estados de alerta (P50–P53, P196–P211)

- **Por instante y por horizonte:** estado pronosticado para el instante objetivo frente al estado
  observado en ese instante.
- **Estado emitido (combinado):** en cada instante de emisión t, el más severo de los estados
  pronosticados a τ₁ y τ₂; se compara con el más severo de los observados en t + τ₁ y t + τ₂ (solo
  instantes con ambos horizontes disponibles).
- Reducciones 2 × 2: **aviso** = prealerta o alerta frente a normal; **alerta** = alerta frente al resto.
  La falsa alarma se define solo en estas tablas (P211).
- **Errores ordinales:** % exacto, subestimación de 1 o 2 niveles y sobreestimación de 1 o 2 niveles.
- **Estados en caudal:** el nivel pronosticado y el observado se convierten a caudal con la curva de
  descarga y se comparan con @@q_pre_g@@ m³/s (prealerta) y @@q_ale_g@@ m³/s (alerta) (P103, P247). Como la curva es
  creciente, equivale a comparar el nivel con @@cm_pre_2@@ y @@cm_ale_2@@ cm, que se muestran solo como referencia.
- **Evaluación por eventos (simulación secuencial):** evento observado = cruce del umbral de prealerta
  (o de alerta) agrupando cruces a menos de 6 h. Se detecta si antes del cruce se emitió un estado igual o
  más severo, con anticipación no mayor que el horizonte, y se mantuvo hasta el cruce. La anticipación
  efectiva es la duración de esa racha (acotada por τ). Se reportan eventos omitidos y episodios de falsa
  alarma (alerta emitida sin evento observado en la ventana del horizonte).
"""),
    ("code", r'''
filas, MATRICES = [], {}
for part in ["val", "test"]:
    for m in MODELOS:
        if not all((m, part, h) in DATOS for h in HORIZONTES):
            continue
        for h in HORIZONTES + ["combinado"]:
            if h == "combinado":
                e = estados_combinados(m, part)
                e = e[e["completo"]]
            else:
                e = estados_horizonte(m, part, h).dropna()
            res, cm = metricas_estados(e["est_obs"].to_numpy(int), e["est_pred"].to_numpy(int))
            MATRICES[(m, part, h)] = cm
            filas.append({"modelo": m, "familia": FAMILIA_DE[m], "particion": part, "horizonte": h, **res})
alertas = pd.DataFrame(filas)
alertas.to_csv(TAB_DIR / "alertas_por_instante.csv", index=False)
pd.DataFrame([{"modelo": m, "particion": p, "horizonte": h, "observado": ESTADOS[i], "pronosticado": ESTADOS[j], "n": int(cm[i, j])}
              for (m, p, h), cm in MATRICES.items() for i in range(3) for j in range(3)]).to_csv(
    TAB_DIR / "matrices_confusion_3x3.csv", index=False)

COLS_AL = ["n", "n_obs_prealerta", "n_obs_alerta", "F1_macro", "POD_aviso", "precision_aviso", "F1_aviso", "FAR_aviso",
           "POFD_aviso", "POD_alerta", "precision_alerta", "F1_alerta", "FAR_alerta", "POFD_alerta"]
COLS_ORD = ["pct_exacto", "pct_sub_1", "pct_sub_2", "pct_sobre_1", "pct_sobre_2"]
for h in HORIZONTES + ["combinado"]:
    t = alertas[(alertas["particion"] == "test") & (alertas["horizonte"] == h)].set_index("modelo")
    print(f"\n=== Alertas por instante en test — {etiqueta_h(h)} ===")
    display(t[COLS_AL].round(3))
    display(t[COLS_ORD + [f"{a}_{e}" for e in ESTADOS for a in ("POD", "precision", "F1")]].round(3))
'''),
    ("code", r'''
filas, detalle = [], []
for part in ["val", "test"]:
    obs = SERIE_OBS[part]
    for m in MODELOS:
        if not all((m, part, h) in DATOS for h in HORIZONTES):
            continue
        series = {h: estados_horizonte(m, part, h)["est_pred"] for h in HORIZONTES}
        series["combinado"] = estados_combinados(m, part)["est_pred"]
        for h, est in series.items():
            ant_max = max(HORIZONTES) if h == "combinado" else h
            for k in (1, 2):
                ev, falsas = simular_eventos(est, obs, k, ant_max)
                evaluables = ev[ev["evaluable"]]
                filas.append({"modelo": m, "familia": FAMILIA_DE[m], "particion": part, "horizonte": h,
                              "estado": ESTADOS[k], "n_eventos": len(ev), "n_evaluables": len(evaluables),
                              "detectados": int(evaluables["detectado"].sum()),
                              "omitidos": int((~evaluables["detectado"]).sum()),
                              "pct_detectados": 100 * evaluables["detectado"].mean() if len(evaluables) else np.nan,
                              "anticipacion_media_h": evaluables["anticipacion_h"].mean(),
                              "anticipacion_mediana_h": evaluables["anticipacion_h"].median(),
                              "anticipacion_max_posible_h": ant_max, "falsas_alarmas_episodios": falsas})
                detalle.append(ev.assign(modelo=m, particion=part, horizonte=str(h), estado=ESTADOS[k]))
alertas_eventos = pd.DataFrame(filas)
alertas_eventos.to_csv(TAB_DIR / "alertas_por_evento.csv", index=False)
pd.concat(detalle, ignore_index=True).to_csv(TAB_DIR / "alertas_eventos_detalle.csv", index=False)
t = alertas_eventos[(alertas_eventos["particion"] == "test")]
display(t[t["horizonte"].astype(str) == "combinado"].set_index(["modelo", "estado"]).drop(columns=["familia", "particion", "horizonte"]).round(2))
display(t[t["horizonte"].astype(str) != "combinado"].set_index(["modelo", "horizonte", "estado"]).drop(columns=["familia", "particion"]).round(2))
'''),
    ("md", r"""
## 7. Brecha de generalización relativa a la persistencia (P58, P142)

Razón de error fuera de muestra / dentro de muestra: RMSE_val / RMSE_train, RMSE_val_fold / RMSE_train y
RMSE_test / RMSE_train. La persistencia no tiene parámetros ajustados, así que su razón solo refleja el
cambio de régimen y de variabilidad entre periodos. La **brecha relativa** = razón del modelo / razón de
la persistencia aísla el sobreajuste: valores mayores que 1 indican que el modelo se degrada fuera de
muestra más de lo que explica el cambio de periodo. La referencia de 1,3 de las figuras es solo
orientativa (la tesis no fija un umbral). Si el modelo final se reentrenó con train + val, la razón
val/train no es fuera de muestra; se usan val_fold y test.
"""),
    ("code", r'''
rmse_ancho = metricas.pivot_table(index=["modelo", "horizonte"], columns="particion", values="RMSE")
brechas = pd.DataFrame(index=rmse_ancho.index)
for part in ["val", "val_fold", "test"]:
    if part in rmse_ancho.columns:
        brechas[f"razon_{part}_train"] = rmse_ancho[part] / rmse_ancho["train"]
brechas = brechas.reset_index()
for c in [c for c in brechas.columns if c.startswith("razon_")]:
    ref = brechas[brechas["modelo"] == "Persistencia"].set_index("horizonte")[c]
    brechas[c.replace("razon_", "brecha_rel_")] = brechas[c] / brechas["horizonte"].map(ref)
col_diag = "brecha_rel_val_fold_train" if "brecha_rel_val_fold_train" in brechas else "brecha_rel_test_train"
brechas["diagnostico"] = np.where(brechas[col_diag] > 1.3, "posible sobreajuste (se degrada más que la persistencia)",
                                  np.where(brechas[col_diag] > 1.0, "degradación algo mayor que la persistencia",
                                           "generaliza como o mejor que la persistencia"))
brechas.to_csv(TAB_DIR / "brechas_generalizacion.csv", index=False)
for h in HORIZONTES:
    print(f"\n=== Brecha de generalización — {etiqueta_h(h)} ===")
    display(brechas[brechas["horizonte"] == h].set_index("modelo").drop(columns="horizonte").round(3))
'''),
    ("md", r"""
## 8. Gráficas del dataset (G1, G2)
"""),
]
