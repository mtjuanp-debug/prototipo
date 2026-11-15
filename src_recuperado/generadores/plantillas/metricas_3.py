# -*- coding: utf-8 -*-
"""Plantilla del notebook de métricas (src/metrics/<slug>_models_metric.ipynb): parte 3 de 3.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("code", r'''
colores = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green", "val_fold": "tab:purple"}
fig, ax = plt.subplots(figsize=(16, 4.5))
for part in PARTICIONES:
    ax.plot(SERIE_OBS[part].index, SERIE_OBS[part].values, lw=0.5, color=colores[part], label=part)
ax.axhline(U_PRE, ls="--", lw=0.9, color="darkorange", label=f"prealerta = {U_PRE:.0f} cm")
ax.axhline(U_ALE, ls="--", lw=0.9, color="crimson", label=f"alerta = {U_ALE:.0f} cm")
ax.axhline(P95, ls=":", lw=0.9, color="gray", label=f"P95 train = {P95:.0f} cm")
ax.set_title("G1 — Nivel observado del @@nombre@@ (y_obs) por partición y umbrales oficiales")
ax.set_ylabel("Nivel (cm)")
ax.legend(ncol=6, fontsize=8)
ax.grid(alpha=0.3)
guardar_figura(fig, "G01_serie_particiones.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.boxplot([SERIE_OBS[p].dropna().values for p in PARTICIONES], tick_labels=PARTICIONES, showfliers=True,
           flierprops={"markersize": 2, "alpha": 0.3})
ax.axhline(U_PRE, ls="--", lw=0.8, color="darkorange")
ax.axhline(U_ALE, ls="--", lw=0.8, color="crimson")
ax.set_title("G2 — Distribución del nivel observado por partición")
ax.set_ylabel("Nivel (cm)")
ax.grid(axis="y", alpha=0.3)
guardar_figura(fig, "G02_boxplot_particiones.png")
resumen_obs = pd.DataFrame({p: SERIE_OBS[p].describe(percentiles=[0.5, 0.95, 0.99]) for p in PARTICIONES})
for p in PARTICIONES:
    o = SERIE_OBS[p]
    resumen_obs.loc["eventos_prealerta", p] = len(episodios(a_horas(o.asfreq("h").index), a_caudal(o.asfreq("h").to_numpy()), Q_PRE))
    resumen_obs.loc["eventos_alerta", p] = len(episodios(a_horas(o.asfreq("h").index), a_caudal(o.asfreq("h").to_numpy()), Q_ALE))
display(resumen_obs.round(1))
'''),
    ("md", r"""
## 9. Curvas de sobreajuste (G3)

Cada panel muestra el RMSE (cm) en el ajuste y en la validación interna a lo largo del entrenamiento
del modelo final. La línea punteada marca la mejor iteración. Si la validación sube mientras el ajuste
sigue bajando, hay sobreajuste. En las redes, la pérdida de ajuste se mide durante la época, con dropout
activo, por lo que puede quedar por encima de la validación.
"""),
    ("code", r'''
ETIQ_EJE = {"epocas": "Época", "rondas": "Ronda de boosting", "estimadores": "N.º de estimadores", "n_arboles": "N.º de árboles"}


def graficar_curvas(h):
    claves = [m for m in MODELOS if isinstance(HIST.get(f"{m}|h{h}"), dict) and "train" in HIST[f"{m}|h{h}"]]
    extra = rf_prof is not None and (rf_prof["horizonte"] == h).any()
    if not claves and not extra:
        print(f"Sin historiales para h = {h}.")
        return
    n_col = 4
    n_fil = int(np.ceil((len(claves) + extra) / n_col))
    fig, ejes = plt.subplots(n_fil, n_col, figsize=(19, 3.7 * n_fil), squeeze=False)
    ejes = ejes.ravel()
    for ax, m in zip(ejes, claves):
        hst = HIST[f"{m}|h{h}"]
        x = np.arange(1, len(hst["train"]) + 1)
        ax.plot(x, hst["train"], label="ajuste", color="tab:blue")
        if hst.get("val") is not None:
            ax.plot(np.arange(1, len(hst["val"]) + 1), hst["val"], label=hst.get("origen") or "validación / control", color="tab:orange")
        if hst.get("mejor_iteracion") is not None:
            ax.axvline(hst["mejor_iteracion"], color="gray", ls="--", lw=0.9)
        if hst.get("tipo") == "epocas" and "lr" in hst:
            ax2 = ax.twinx()
            ax2.plot(np.arange(1, len(hst["lr"]) + 1), hst["lr"], color="green", lw=0.8, alpha=0.6)
            ax2.set_yscale("log")
            ax2.tick_params(axis="y", labelsize=7, colors="green")
        ax.set_title(f"{m} (mejor: {hst.get('mejor_iteracion')})", fontsize=10)
        ax.set_xlabel(ETIQ_EJE.get(hst.get("tipo"), "Iteración"))
        ax.set_ylabel("RMSE (cm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    if extra:
        ax = ejes[len(claves)]
        d = rf_prof[rf_prof["horizonte"] == h]
        ax.plot(d["max_depth"].astype(str), d["RMSE_train"], "o-", label="train")
        ax.plot(d["max_depth"].astype(str), d["RMSE_val"], "o-", label="val")
        ax.set_title("RF — curva de validación de max_depth", fontsize=10)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("RMSE (cm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in ejes[len(claves) + extra:]:
        ax.axis("off")
    fig.suptitle(f"G3 — Curvas de sobreajuste, horizonte {etiqueta_h(h)} (verde: learning rate)", fontsize=13)
    fig.tight_layout()
    guardar_figura(fig, f"G03_curvas_sobreajuste_h{h}.png")


for hh in HORIZONTES:
    graficar_curvas(hh)
'''),
    ("md", r"""
## 10. Gráficas por modelo (G4–G9, G15)

Se muestran el modelo final de cada familia y la persistencia, en ambos horizontes. El modelo
seleccionado para el río se marca con ★.
"""),
    ("code", r'''
def datos_modelo(modelo, part, h):
    return DATOS[(modelo, part, h)]


def nombre(m):
    return f"{m} ★" if m in MODELO_SEL.values() else m


def grafico_real_vs_predicho(h):
    lista = MEJORES_POR_H[h]
    base = datos_modelo("Persistencia", "test", h)
    o = base["y_real_cm"].to_numpy()
    horas = a_horas(base.index)
    eventos = episodios(horas, a_caudal(o), Q_PRE) or episodios(horas, o, P95)
    eventos = sorted(eventos, key=lambda ev: -o[ev].max())[:3]
    fig = plt.figure(figsize=(18, 10))
    ax0 = fig.add_subplot(2, 1, 1)
    grilla = base.asfreq("h")   # los huecos sin observación quedan como cortes en la línea
    ax0.plot(grilla.index, grilla["y_real_cm"], color="black", lw=0.9, label="Observado")
    for m in lista[1:]:
        ax0.plot(grilla.index, datos_modelo(m, "test", h)["y_pred_cm"].reindex(grilla.index), lw=0.6, alpha=0.85, label=nombre(m))
    ax0.axhline(U_PRE, ls="--", lw=0.7, color="darkorange")
    ax0.axhline(U_ALE, ls="--", lw=0.7, color="crimson")
    ax0.set_title(f"G4 — Nivel observado vs predicho en test ({etiqueta_h(h)})")
    ax0.set_ylabel("Nivel (cm)")
    ax0.legend(ncol=7, fontsize=8)
    ax0.grid(alpha=0.3)
    for k, ev in enumerate(eventos):
        ax = fig.add_subplot(2, 3, 4 + k)
        t_pico = base.index[ev[np.argmax(o[ev])]]
        ventana = slice(t_pico - pd.Timedelta(hours=36), t_pico + pd.Timedelta(hours=36))
        ax.plot(base.loc[ventana].index, base.loc[ventana, "y_real_cm"], color="black", lw=1.6, label="Observado")
        for m in lista:
            ax.plot(datos_modelo(m, "test", h).loc[ventana, "y_pred_cm"], lw=1.0, label=nombre(m))
        ax.axhline(U_PRE, ls="--", lw=0.7, color="darkorange")
        ax.axhline(U_ALE, ls="--", lw=0.7, color="crimson")
        ax.set_title(f"Evento {k + 1}: pico {base.loc[t_pico, 'y_real_cm']:.0f} cm ({t_pico:%Y-%m-%d %H:%M})", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    guardar_figura(fig, f"G04_real_vs_predicho_h{h}.png")


for hh in HORIZONTES:
    grafico_real_vs_predicho(hh)
'''),
    ("code", r'''
# G5 — dispersión observado vs predicho (línea 1:1) en train, val y test
for h in HORIZONTES:
    lista = MEJORES_POR_H[h]
    fig, ejes = plt.subplots(len(lista), 3, figsize=(14, 3.9 * len(lista)), squeeze=False)
    for i, m in enumerate(lista):
        for j, part in enumerate(PARTICIONES):
            g = datos_modelo(m, part, h)
            ax = ejes[i, j]
            PredictionErrorDisplay.from_predictions(g["y_real_cm"], g["y_pred_cm"], kind="actual_vs_predicted",
                                                    subsample=4000, random_state=SEED, ax=ax,
                                                    scatter_kwargs={"s": 4, "alpha": 0.35})
            met = metricas_regresion(g["y_real_cm"].to_numpy(), g["y_pred_cm"].to_numpy())
            ax.set_title(f"{nombre(m)} — {part} | R² {met['R2']:.3f} | RMSE {met['RMSE']:.2f} cm", fontsize=9)
            ax.set_xlabel("Predicho (cm)")
            ax.set_ylabel("Observado (cm)")
    fig.suptitle(f"G5 — Observado vs predicho ({etiqueta_h(h)})", fontsize=13)
    fig.tight_layout()
    guardar_figura(fig, f"G05_dispersion_h{h}.png")
'''),
    ("code", r'''
# G6 — residuos (predicho − observado, signo del sesgo medio) vs predicho + histograma (test)
for h in HORIZONTES:
    lista = MEJORES_POR_H[h]
    fig, ejes = plt.subplots(len(lista), 2, figsize=(13, 3.3 * len(lista)), squeeze=False)
    for i, m in enumerate(lista):
        g = datos_modelo(m, "test", h)
        residuo = g["y_pred_cm"] - g["y_real_cm"]
        muestra = g.sample(min(5000, len(g)), random_state=SEED).index
        ejes[i, 0].scatter(g.loc[muestra, "y_pred_cm"], residuo.loc[muestra], s=4, alpha=0.35)
        ejes[i, 0].axhline(0, color="black", lw=0.8)
        ejes[i, 0].set_title(f"{nombre(m)} — residuo vs predicho (test)", fontsize=9)
        ejes[i, 0].set_xlabel("Predicho (cm)")
        ejes[i, 0].set_ylabel("Predicho − observado (cm)")
        ejes[i, 1].hist(residuo, bins=80, color="tab:purple", alpha=0.8)
        ejes[i, 1].axvline(residuo.mean(), color="black", ls="--", lw=0.8, label=f"sesgo medio {residuo.mean():.2f} cm")
        ejes[i, 1].set_yscale("log")
        ejes[i, 1].set_title(f"{nombre(m)} — histograma de residuos (escala log)", fontsize=9)
        ejes[i, 1].legend(fontsize=8)
    for ax in ejes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle(f"G6 — Residuos ({etiqueta_h(h)})", fontsize=13)
    fig.tight_layout()
    guardar_figura(fig, f"G06_residuos_h{h}.png")
'''),
    ("code", r'''
# G7 — MAE y sesgo por rango de nivel observado (test); rangos por percentiles de train y umbrales oficiales
cortes = sorted({("P50", SERIE_OBS["train"].quantile(0.5)), ("P90", SERIE_OBS["train"].quantile(0.9)),
                 ("P95", P95), ("prealerta", U_PRE), ("alerta", U_ALE)}, key=lambda x: x[1])
bordes = [-np.inf] + [v for _, v in cortes] + [np.inf]
nombres_c = [n for n, _ in cortes]
etiquetas = [f"< {nombres_c[0]}"] + [f"{a}–{b}" for a, b in zip(nombres_c[:-1], nombres_c[1:])] + [f"≥ {nombres_c[-1]}"]
for h in HORIZONTES:
    lista = MEJORES_POR_H[h]
    filas = []
    for m in lista:
        g = datos_modelo(m, "test", h)
        rango = pd.cut(g["y_real_cm"], bordes, labels=etiquetas, right=False)
        e = g["y_pred_cm"] - g["y_real_cm"]
        for et, v in e.groupby(rango, observed=False):
            filas.append({"modelo": m, "rango": et, "MAE": v.abs().mean(), "sesgo": v.mean(), "n": len(v)})
    error_rango = pd.DataFrame(filas)
    error_rango.to_csv(TAB_DIR / f"error_por_rango_h{h}.csv", index=False)
    fig, ejes = plt.subplots(1, 2, figsize=(18, 5))
    n_por_rango = error_rango.groupby("rango", observed=False)["n"].first().loc[etiquetas]
    for ax, met in zip(ejes, ["MAE", "sesgo"]):
        tabla_er = error_rango.pivot(index="rango", columns="modelo", values=met).loc[etiquetas, lista]
        tabla_er.plot(kind="bar", ax=ax, width=0.85)
        ax.set_xticklabels([f"{et}\n(n={n})" for et, n in n_por_rango.items()], rotation=0, fontsize=8)
        ax.set_ylabel(f"{met} (cm)")
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(met)
    fig.suptitle(f"G7 — Error por rango de nivel observado en test ({etiqueta_h(h)})", fontsize=13)
    fig.tight_layout()
    guardar_figura(fig, f"G07_error_por_rango_h{h}.png")
    display(error_rango.pivot(index="rango", columns="modelo", values="MAE").loc[etiquetas, lista].round(2))
'''),
    ("code", r'''
# G8 — matrices de confusión 3x3 de los estados de alerta (test): τ₁, τ₂ y estado emitido combinado
cols_g8 = HORIZONTES + ["combinado"]
lista = [m for m in MEJORES_POR_H[HORIZONTES[0]]]
fig, ejes = plt.subplots(len(lista), len(cols_g8), figsize=(4.4 * len(cols_g8), 3.9 * len(lista)), squeeze=False)
for i, m in enumerate(lista):
    for j, h in enumerate(cols_g8):
        ax = ejes[i, j]
        m_h = m if h == "combinado" or m == "Persistencia" else MODELO_FINAL.get((FAMILIA_DE[m], h), m)
        cm = MATRICES[(m_h, "test", h)]
        ConfusionMatrixDisplay(cm, display_labels=ESTADOS).plot(ax=ax, colorbar=False, values_format="d", cmap="Blues")
        r = alertas[(alertas["modelo"] == m_h) & (alertas["particion"] == "test") & (alertas["horizonte"] == h)].iloc[0]
        ax.set_title(f"{nombre(m_h)} — {etiqueta_h(h)}\nF1 macro {r['F1_macro']:.2f} | POD aviso {r['POD_aviso']:.2f} | "
                     f"FAR aviso {r['FAR_aviso']:.2f}", fontsize=8)
        ax.set_xlabel("Pronosticado")
        ax.set_ylabel("Observado")
fig.suptitle("G8 — Matrices de confusión de los estados de alerta en test", fontsize=13, y=1.005)
fig.tight_layout()
guardar_figura(fig, "G08_matrices_confusion_3x3.png")
'''),
    ("code", r'''
# G9 — importancia por permutación y G15 — SHAP (si existen los archivos)
def columna_valor(df, preferidas):
    for c in preferidas:
        if c in df.columns:
            return c
    num = [c for c in df.select_dtypes("number").columns if c not in ("horizonte", "fold")]
    return num[0] if num else None


def graficar_importancia(df, col, col_err, titulo, archivo):
    for h in HORIZONTES:
        sub = df[df["horizonte"] == h] if "horizonte" in df.columns else df
        modelos_imp = list(dict.fromkeys(sub["modelo"]))
        if not modelos_imp:
            continue
        n_col = min(3, len(modelos_imp))
        n_fil = int(np.ceil(len(modelos_imp) / n_col))
        fig, ejes = plt.subplots(n_fil, n_col, figsize=(7 * n_col, 6 * n_fil), squeeze=False)
        ejes = ejes.ravel()
        for ax, m in zip(ejes, modelos_imp):
            top = sub[sub["modelo"] == m].nlargest(15, col).iloc[::-1]
            xerr = top[col_err] if col_err and col_err in top.columns else None
            ax.barh(top["variable"].astype(str).str.replace("__", " · "), top[col], xerr=xerr, color="teal")
            ax.set_title(nombre(m), fontsize=10)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="x", alpha=0.3)
        for ax in ejes[len(modelos_imp):]:
            ax.axis("off")
        fig.suptitle(f"{titulo} ({etiqueta_h(h)})", fontsize=13)
        fig.tight_layout()
        guardar_figura(fig, f"{archivo}_h{h}.png")


if importancias is not None and len(importancias):
    c = columna_valor(importancias, ["importancia_cm", "importancia"])
    graficar_importancia(importancias, c, "desv_cm", "G9 — Importancia por permutación: aumento del RMSE al permutar (cm)",
                         "G09_importancia_permutacion")
else:
    print("Sin importancia por permutación: se omite G9.")
if shap_imp is not None and len(shap_imp):
    c = columna_valor(shap_imp, ["shap_abs_medio_cm", "shap_abs_medio", "media_abs_shap", "importancia"])
    graficar_importancia(shap_imp, c, None, "G15 — SHAP: media del |valor SHAP| (cm)", "G15_shap")
else:
    print("Sin shap_importancia.csv: se omite G15.")
'''),
    ("md", r"""
## 11. Gráficas comparativas (G10–G13)
"""),
    ("code", r'''
# G10 — barras agrupadas por modelo y partición
parts_g10 = [p for p in ["train", "val", "val_fold", "test"] if p in set(metricas["particion"])]
for h in HORIZONTES:
    t = metricas[metricas["horizonte"] == h]
    fig, ejes = plt.subplots(2, 3, figsize=(20, 9))
    for ax, met in zip(ejes.ravel(), ["SS", "RMSE", "MAE", "NSE", "KGE", "sesgo"]):
        tabla_m = t.pivot(index="modelo", columns="particion", values=met).reindex(index=MODELOS, columns=parts_g10)
        tabla_m.plot(kind="bar", ax=ax, width=0.8, color=[colores[p] for p in parts_g10])
        ax.set_title(met)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        if met in ("NSE", "KGE"):
            ax.set_ylim(max(0, np.nanmin(tabla_m.to_numpy()) - 0.05), 1.0)
    fig.suptitle(f"G10 — Métricas por modelo y partición ({etiqueta_h(h)})", fontsize=13)
    fig.tight_layout()
    guardar_figura(fig, f"G10_metricas_por_particion_h{h}.png")

# G11 — brechas de generalización relativas a la persistencia
cols_b = [c for c in brechas.columns if c.startswith("brecha_rel_")]
fig, ejes = plt.subplots(1, len(HORIZONTES), figsize=(8 * len(HORIZONTES), 4.8), squeeze=False)
for ax, h in zip(ejes.ravel(), HORIZONTES):
    b = brechas[brechas["horizonte"] == h].set_index("modelo").reindex(MODELOS)
    x = np.arange(len(b))
    ancho = 0.8 / len(cols_b)
    for i, c in enumerate(cols_b):
        ax.bar(x + (i - (len(cols_b) - 1) / 2) * ancho, b[c], ancho, label=c.replace("brecha_rel_", "").replace("_", "/"))
    ax.axhline(1.0, color="black", lw=0.8)
    ax.axhline(1.3, color="crimson", ls="--", lw=0.8, label="referencia orientativa 1,3")
    ax.set_xticks(x, b.index, rotation=45, fontsize=8)
    ax.set_title(etiqueta_h(h))
    ax.set_ylabel("razón del modelo / razón de la persistencia")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("G11 — Brecha de generalización relativa a la persistencia (1 = igual que la persistencia)", fontsize=13)
fig.tight_layout()
guardar_figura(fig, "G11_brechas_vs_persistencia.png")
'''),
    ("code", r'''
# G12 — métricas de test vs horizonte (modelo final de cada familia)
filas = []
for h in HORIZONTES:
    for m in MEJORES_POR_H[h]:
        r = metricas[(metricas["modelo"] == m) & (metricas["horizonte"] == h) & (metricas["particion"] == "test")].iloc[0]
        a = alertas[(alertas["modelo"] == m) & (alertas["horizonte"] == h) & (alertas["particion"] == "test")].iloc[0]
        filas.append({"familia": FAMILIA_DE[m], "horizonte": h, "modelo": m, "SS": r["SS"], "RMSE": r["RMSE"],
                      "NSE": r["NSE"], "KGE": r["KGE"], "RMSE_alto": r["RMSE_alto"], "F1_macro": a["F1_macro"]})
por_h = pd.DataFrame(filas)
por_h.to_csv(TAB_DIR / "metricas_test_por_horizonte.csv", index=False)
mets = ["SS", "RMSE", "NSE", "KGE", "RMSE_alto", "F1_macro"]
fig, ejes = plt.subplots(1, len(mets), figsize=(4 * len(mets), 4.3))
for ax, met in zip(ejes, mets):
    for fam, g in por_h.groupby("familia", sort=False):
        ax.plot(g["horizonte"], g[met], "o-", label=fam, lw=2.4 if fam == FAMILIA_SEL else 1.2)
    ax.set_xticks(HORIZONTES, [etiqueta_h(h) for h in HORIZONTES], fontsize=8)
    ax.set_title(f"{met} en test")
    ax.grid(alpha=0.3)
ejes[0].legend(fontsize=8)
fig.suptitle("G12 — Degradación con el horizonte de pronóstico", fontsize=13)
fig.tight_layout()
guardar_figura(fig, "G12_metricas_vs_horizonte.png")
display(por_h.round(3))
'''),
    ("code", r'''
# G13 — mapa de calor de alertas en test, estado emitido combinado (color = calidad; texto = valor real)
t = alertas[(alertas["particion"] == "test") & (alertas["horizonte"] == "combinado")].set_index("modelo").reindex(MODELOS)
ev = alertas_eventos[(alertas_eventos["particion"] == "test") & (alertas_eventos["horizonte"].astype(str) == "combinado")]
for k in ESTADOS[1:]:
    e = ev[ev["estado"] == k].set_index("modelo").reindex(MODELOS)
    t[f"pct_omitidos_{k}"] = 100 - e["pct_detectados"]
    t[f"anticipacion_{k}_h"] = e["anticipacion_media_h"]
columnas = ["F1_macro", "POD_aviso", "precision_aviso", "F1_aviso", "FAR_aviso", "POFD_aviso", "POD_alerta",
            "precision_alerta", "F1_alerta", "FAR_alerta", "pct_omitidos_prealerta", "pct_omitidos_alerta",
            "anticipacion_prealerta_h", "anticipacion_alerta_h", "pct_exacto"]
valores = t[columnas].astype(float)
calidad = valores.copy()
for c in columnas:
    if c.startswith(("FAR", "POFD")):
        calidad[c] = 1 - valores[c]
    elif c.startswith("pct_omitidos"):
        calidad[c] = 1 - valores[c] / 100
    elif c.startswith("anticipacion"):
        calidad[c] = valores[c] / max(HORIZONTES)
    elif c == "pct_exacto":
        calidad[c] = valores[c] / 100
fig, ax = plt.subplots(figsize=(16, 0.5 * len(t) + 2.2))
im = ax.imshow(calidad.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(columnas)), [c.replace("_", " ") for c in columnas], rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(t)), [nombre(m) for m in t.index], fontsize=8)
for i in range(len(t)):
    for j in range(len(columnas)):
        v = valores.iloc[i, j]
        ax.text(j, i, "—" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="calidad (1 = ideal)")
ax.set_title("G13 — Verificación de alertas en test: estado emitido combinado (aviso = prealerta o alerta)")
guardar_figura(fig, "G13_alertas_heatmap.png")
t[columnas].to_csv(TAB_DIR / "alertas_resumen_test_combinado.csv")
'''),
    ("md", r"""
## 12. Resumen automático
"""),
    ("code", r'''
print(f"Horizontes: τ₁ = {TAU1} h, τ₂ = {TAU2} h | umbrales: prealerta {Q_PRE:.0f} m³/s (≈{U_PRE:.1f} cm), alerta {Q_ALE:.0f} m³/s (≈{U_ALE:.1f} cm) | P95 train {P95:.1f} cm")
print(f"Familia seleccionada (SS medio de selección): {FAMILIA_SEL} | SS medio {seleccion.loc[FAMILIA_SEL, 'SS_medio']:.4f}\n")
for h in HORIZONTES:
    print(f"--- Horizonte {etiqueta_h(h)} ---")
    for m in MEJORES_POR_H[h]:
        r = metricas[(metricas["modelo"] == m) & (metricas["horizonte"] == h)].set_index("particion")
        if "test" not in r.index:
            continue
        rt = r.loc["test"]
        print(f"  {nombre(m):14s} | SS test {rt['SS']:+.3f} | RMSE {rt['RMSE']:.2f} cm | MAE {rt['MAE']:.2f} | sesgo {rt['sesgo']:+.2f} "
              f"| NSE {rt['NSE']:.3f} | KGE {rt['KGE']:.3f} | RMSE alto {rt['RMSE_alto']:.2f} | picos subestimados "
              f"{rt['pct_picos_subestimados']:.0f} % | desfase {rt['desfase_medio_h']:+.1f} h")
print("\nHipótesis (@@nombre@@; Holm sobre los 2 horizontes, el equipo la extiende a los 8 contrastes):")
for _, r in dm_sel.iterrows():
    print(f"  {r['modelo']} {etiqueta_h(r['horizonte'])}: SS {r['SS']:+.3f}, DM* {r['DM_HLN']:.2f}, p Holm {r['p_holm']:.2e} -> "
          f"{'se rechaza H0' if r['rechaza_H0_5pct'] else 'no se rechaza H0'}")
ev = alertas_eventos[(alertas_eventos["particion"] == "test") & (alertas_eventos["horizonte"].astype(str) == "combinado")
                     & alertas_eventos["modelo"].isin(MODELO_SEL.values())]
print("\nAlertas del modelo seleccionado en test (estado emitido combinado):")
for _, r in ev.iterrows():
    print(f"  {r['estado']}: {r['detectados']}/{r['n_evaluables']} eventos detectados, {r['omitidos']} omitidos, "
          f"anticipación media {r['anticipacion_media_h']:.1f} h, {r['falsas_alarmas_episodios']} episodios de falsa alarma")
print("\nTablas en:", TAB_DIR)
print("Figuras en:", FIG_DIR)
'''),
]
