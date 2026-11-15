# -*- coding: utf-8 -*-
"""Plantilla del notebook de modelos (src/models/modelos_<slug>.ipynb): parte 3 de 3.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("code", r'''
def entrenar_final(familia, config_id, cfg, D):
    tau, pos = D["tau"], D["pos"]
    part = PART[pos]
    pos_tv = pos[np.isin(part, ["train", "val"])]
    t0 = time.time()
    extra = {}
    if familia == "RandomForest":
        modelo = ajustar_rf(cfg, pos_tv, D)
        predecir = lambda p: modelo.predict(X_TAB[p])
        if GUARDAR_RF:
            joblib.dump(modelo, MODELOS_DIR / f"RF_h{tau}.joblib", compress=3)
            extra["archivo"] = f"RF_h{tau}.joblib"
    elif familia == "AdaBoost":
        modelo = ajustar_ada(cfg, pos_tv, D)
        predecir = lambda p: modelo.predict(X_TAB[p])
        joblib.dump(modelo, MODELOS_DIR / f"ADA_h{tau}.joblib", compress=3)
        extra["archivo"] = f"ADA_h{tau}.joblib"
    elif familia == "XGBoost":
        pos_fit, pos_ctrl = division_es(pos_tv)
        modelo = ajustar_xgb(cfg, pos_fit, pos_ctrl, D, curva_train=True)
        predecir = lambda p: modelo.predict(X_TAB[p])
        ev = modelo.evals_result()
        extra["mejor_iteracion"] = int(modelo.best_iteration) + 1
        HISTORIALES[f"XGB|h{tau}"] = {"tipo": "rondas", "origen": "final: ajuste vs control (último 10 % de train+val)",
                                      "train": [float(v) * RANGO_CM for v in ev["validation_0"]["rmse"]],
                                      "val": [float(v) * RANGO_CM for v in ev["validation_1"]["rmse"]],
                                      "mejor_iteracion": extra["mejor_iteracion"]}
        modelo.save_model(MODELOS_DIR / f"XGB_h{tau}.json")
        extra["archivo"] = f"XGB_h{tau}.json"
    else:
        pos_fit, pos_ctrl = division_es(pos_tv)
        modelo, hist, mejor_epoca = entrenar_red(cfg, pos_fit, pos_ctrl, D, etiqueta=f"{CORTO[familia]} h={tau}",
                                                 verboso=True)
        predecir = lambda p: predecir_red(modelo, p, D)
        extra.update(mejor_iteracion=mejor_epoca, epocas_corridas=len(hist["train"]),
                     s_por_epoca=round(float(np.mean(hist["s_epoca"])), 2))
        HISTORIALES[f"{CORTO[familia]}|h{tau}"] = {"tipo": "epocas",
                                                   "origen": "final: ajuste vs control (último 10 % de train+val)",
                                                   **hist, "mejor_iteracion": mejor_epoca}
        torch.save(modelo.state_dict(), MODELOS_DIR / f"{CORTO[familia]}_h{tau}.pt")
        extra["archivo"] = f"{CORTO[familia]}_h{tau}.pt"
        # TorchScript: la aplicación puede cargarlo sin la clase RegresorRecurrente (entrada: lote × 24 × variables)
        try:
            # Se traza una copia en CPU: el estado oculto inicial queda fijado al dispositivo del trazado
            modelo_cpu = RegresorRecurrente(len(FEAT_SEQ), cfg["hidden_size"], cfg["num_layers"], cfg["celda"],
                                            cfg["dropout"], 0.0)
            modelo_cpu.load_state_dict({k: v.detach().cpu() for k, v in modelo.state_dict().items()})
            modelo_cpu.eval()
            ejemplo = torch.zeros(2, LOOKBACK_H, len(FEAT_SEQ))
            torch.jit.trace(modelo_cpu, ejemplo).save(str(MODELOS_DIR / f"{CORTO[familia]}_h{tau}_ts.pt"))
            del modelo_cpu
            extra["archivo_torchscript"] = f"{CORTO[familia]}_h{tau}_ts.pt"
        except Exception as error:
            print(f"  Aviso: no se pudo exportar TorchScript de {CORTO[familia]} h={tau}: {error}")
    tiempo = time.time() - t0
    fila = {"familia": familia, "modelo": CORTO[familia], "horizonte": tau, "config_id": config_id,
            "parametros": json.dumps(cfg), "tiempo_final_s": round(tiempo, 1)}
    for nombre_part in ("train", "val", "test"):
        p = pos[part == nombre_part]
        pred = predecir(p) + D["base"][p]          # test: única predicción del modelo final (en nivel)
        registrar_predicciones(CORTO[familia], familia, tau, nombre_part, p, D["y"][p], pred)
        if nombre_part != "test":
            fila[f"RMSE_{nombre_part}_ajuste_final"] = rmse_cm(D["y"][p], pred)
    fila.update({k: v for k, v in SELECCION[(familia, tau)].items() if k != "cfg"})
    fila.update(extra)
    RESUMEN.append(fila)
    MODELOS_FINALES[f"{CORTO[familia]}|h{tau}"] = {
        "familia": familia, "horizonte": tau, "config_id": config_id, "parametros": cfg,
        **{k: extra[k] for k in ("archivo", "archivo_torchscript", "mejor_iteracion") if k in extra}}
    print(f"  Final {CORTO[familia]:5s} h={tau} ({config_id}) | RMSE train {fila['RMSE_train_ajuste_final']:.2f} cm "
          f"| val (en ajuste) {fila['RMSE_val_ajuste_final']:.2f} cm | {tiempo:.0f} s")
    del modelo
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def diagnostico_arbol(familia, cfg, D, con_shap):
    pos = D["pos"]
    pos_tr, pos_va = pos[PART[pos] == "train"], pos[PART[pos] == "val"]
    tau = D["tau"]
    if familia == "RandomForest":
        modelo = ajustar_rf(cfg, pos_tr, D)
        sub = submuestra(pos_tr, N_CURVA_TRAIN)
        curva = {"train": curva_arboles_rf(modelo, sub, D), "val": curva_arboles_rf(modelo, pos_va, D)}
        HISTORIALES[f"RF|h{tau}"] = {"tipo": "n_arboles", "origen": "diagnóstico: ajuste en train, evaluación en val",
                                     **curva, "mejor_iteracion": int(np.argmin(curva["val"])) + 1}
        CURVAS_PROFUNDIDAD.append(curva_profundidad_rf(cfg, pos_tr, pos_va, D))
    elif familia == "AdaBoost":
        modelo = ajustar_ada(cfg, pos_tr, D)
        sub = submuestra(pos_tr, N_CURVA_TRAIN)
        curva = {"train": [rmse_cm(D["yf"][sub], p) for p in modelo.staged_predict(X_TAB[sub])],
                 "val": [rmse_cm(D["yf"][pos_va], p) for p in modelo.staged_predict(X_TAB[pos_va])]}
        HISTORIALES[f"ADA|h{tau}"] = {"tipo": "estimadores", "origen": "diagnóstico: ajuste en train, evaluación en val",
                                      **curva, "mejor_iteracion": int(np.argmin(curva["val"])) + 1}
    else:
        pos_fit, pos_es = division_es(pos_tr)
        modelo = ajustar_xgb(cfg, pos_fit, pos_es, D)
    importancia_permutacion(familia, modelo, pos_va, D)
    if con_shap:
        importancia_shap(familia, modelo, pos_va, D)
    del modelo
    gc.collect()
'''),
    ("md", r"""
## 7. Ejecución por horizonte
"""),
    ("code", r'''
t_global = time.time()
for tau in HORIZONTES:
    print(f"\n{'=' * 90}\nHORIZONTE τ = {tau} h\n{'=' * 90}")
    D = preparar_horizonte(tau)
    folds = {k: particion_fold(D["pos"], k) for k in FOLDS}
    rmse_pers = {}
    for k, (pos_ent, pos_eva) in folds.items():
        assert len(pos_eva) > 0, f"El bloque {k} no tiene filas evaluables"
        assert TS[pos_ent].max() < BORDES[k - 1] - pd.Timedelta(hours=MARGEN_H) or k == 1
        assert not np.isin(PART[np.concatenate([pos_ent, pos_eva])], ["test"]).any()
        rmse_pers[k] = rmse_cm(D["y"][pos_eva], PERS[pos_eva])
        registrar_predicciones("Persistencia", "Persistencia", tau, "val_fold", pos_eva, D["y"][pos_eva],
                               PERS[pos_eva], fold=k)
        print(f"  fold {k}: entrena {len(pos_ent):,} filas (hasta {TS[pos_ent].max()}) | evalúa {len(pos_eva):,} "
              f"| RMSE persistencia {rmse_pers[k]:.2f} cm")
    for p in ("train", "val", "test"):
        sel = D["pos"][PART[D["pos"]] == p]
        registrar_predicciones("Persistencia", "Persistencia", tau, p, sel, D["y"][sel], PERS[sel])
    RESUMEN.append({"familia": "Persistencia", "modelo": "Persistencia", "horizonte": tau, "config_id": "-",
                    "SS_medio_folds": 0.0, "RMSE_medio_folds": float(np.mean(list(rmse_pers.values())))})

    for familia in FAMILIAS:
        print(f"\n{familia} — búsqueda ({len(CANDIDATOS[familia])} configuraciones × {len(folds)} folds)")
        buscar(familia, D, folds, rmse_pers)
        config_id = elegir_mejor(familia, D, folds)
        if config_id is not None and not SOLO_BUSQUEDA:
            entrenar_final(familia, config_id, SELECCION[(familia, tau)]["cfg"], D)

    # Diagnóstico de árboles (fuera de muestra en val) y SHAP del mejor árbol compatible con TreeExplainer
    compatibles = [f for f in ("RandomForest", "XGBoost") if (f, tau) in SELECCION]
    familia_shap = max(compatibles, key=lambda f: SELECCION[(f, tau)]["SS_medio_folds"]) if compatibles else None
    for familia in ([] if SOLO_BUSQUEDA else FAMILIAS_ARBOL):
        if (familia, tau) in SELECCION:
            print(f"Diagnóstico {familia} h={tau}")
            diagnostico_arbol(familia, SELECCION[(familia, tau)]["cfg"], D, con_shap=familia == familia_shap)
    del D
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print(f"\nTiempo total: {(time.time() - t_global) / 60:.1f} min")
'''),
    ("md", r"""
## 8. Selección del modelo (tesis, P57)

Mayor SS promedio entre folds y entre horizontes, con la mejor configuración de cada familia en cada horizonte.
Si la diferencia con el líder es menor de 0,01, gana el de menor RMSE en niveles mayores al P95 de `y_obs` de
train (promedio de los horizontes, sobre las predicciones del walk-forward).
"""),
    ("code", r'''
busqueda = pd.DataFrame(BUSQUEDA)
filas_sel = []
for familia in FAMILIAS:
    if not all((familia, tau) in SELECCION for tau in HORIZONTES):
        continue
    ss = [s for tau in HORIZONTES for s in busqueda.loc[(busqueda["familia"] == familia) & (busqueda["horizonte"] == tau)
                                                         & (busqueda["config_id"] == SELECCION[(familia, tau)]["config_id"]), "SS"]]
    filas_sel.append({"familia": familia, "modelo": CORTO[familia], "SS_global": float(np.mean(ss)),
                      "RMSE_alto_global": float(np.nanmean([SELECCION[(familia, tau)]["RMSE_alto_folds"] for tau in HORIZONTES])),
                      **{f"config_h{tau}": SELECCION[(familia, tau)]["config_id"] for tau in HORIZONTES}})
tabla_sel = pd.DataFrame(filas_sel).sort_values("SS_global", ascending=False).reset_index(drop=True)
tabla_sel["definicion_SS"] = "media de SS por fold y horizonte (mejor configuración de cada horizonte)"
tabla_sel["definicion_RMSE_alto"] = (f"RMSE walk-forward en y_obs > P95 ({P95_TRAIN_CM:.1f} cm; P95 de y_obs en las "
                                     "filas válidas de train), folds unidos, media de horizontes")
lider = tabla_sel.iloc[0]
empatados = tabla_sel[tabla_sel["SS_global"] >= lider["SS_global"] - EMPATE_SS]
if len(empatados) > 1 and empatados["RMSE_alto_global"].notna().any():
    ganador = empatados.loc[empatados["RMSE_alto_global"].idxmin()]
    criterio = f"empate (< {EMPATE_SS}) resuelto con RMSE en niveles > P95 ({P95_TRAIN_CM:.1f} cm)"
else:
    ganador, criterio = lider, "mayor SS promedio entre folds y horizontes"
tabla_sel["seleccionado"] = tabla_sel["familia"] == ganador["familia"]
SELECCION_GLOBAL = {"familia": ganador["familia"], "modelo": ganador["modelo"], "criterio": criterio,
                    "SS_global": float(ganador["SS_global"]), "P95_train_cm": P95_TRAIN_CM,
                    "configuraciones": {int(tau): SELECCION[(ganador["familia"], tau)]["config_id"] for tau in HORIZONTES}}
display(tabla_sel.round(4))
print(f"Modelo seleccionado: {ganador['familia']} ({criterio})")
'''),
    ("md", r"""
## 9. Guardado de resultados
"""),
    ("code", r'''
predicciones = pd.concat(PREDICCIONES, ignore_index=True)
predicciones["fold"] = predicciones["fold"].astype("Int64")
# Test se predice una sola vez por modelo y horizonte
_test = predicciones[predicciones["particion"] == "test"]
assert not _test.duplicated(["modelo", "horizonte", "timestamp"]).any()
assert predicciones.loc[predicciones["particion"] == "val_fold", "fold"].notna().all()
assert predicciones.loc[predicciones["particion"] != "val_fold", "fold"].isna().all()

resumen = pd.DataFrame(RESUMEN)
resumen["SS_global"] = resumen["familia"].map(tabla_sel.set_index("familia")["SS_global"])
resumen["seleccionado"] = resumen["familia"] == SELECCION_GLOBAL["familia"]

predicciones.to_parquet(OUT_DIR / "predicciones_@@slug@@.parquet", index=False)
resumen.to_csv(OUT_DIR / "resumen_entrenamiento.csv", index=False)
busqueda.to_csv(RUTA_BUSQUEDA, index=False)
tabla_sel.to_csv(OUT_DIR / "seleccion_modelo.csv", index=False)
if CURVAS_PROFUNDIDAD:
    pd.concat(CURVAS_PROFUNDIDAD, ignore_index=True).to_csv(OUT_DIR / "rf_curva_profundidad.csv", index=False)
if IMPORTANCIAS:
    pd.concat(IMPORTANCIAS, ignore_index=True).to_csv(OUT_DIR / "importancia_permutacion.csv", index=False)
if SHAP_FILAS:
    pd.concat(SHAP_FILAS, ignore_index=True).to_csv(OUT_DIR / "shap_importancia.csv", index=False)
with open(OUT_DIR / "historiales.json", "w", encoding="utf-8") as f:
    json.dump(HISTORIALES, f)

configuracion = {
    "target": TARGET, "horizontes": HORIZONTES, "horizontes_json": CFG_HORIZONTES, "margen_h": MARGEN_H,
    "lookback_h": LOOKBACK_H, "seed": SEED, "rapido": RAPIDO,
    "walk_forward": {"n_bloques": N_BLOQUES, "folds_evaluados": FOLDS,
                     "bordes_bloques": [str(b) for b in BORDES], "fraccion_early_stopping": FRACCION_ES},
    "presupuesto": {"n_iter_arboles": N_ITER_ARBOLES, "n_iter_redes": N_ITER_REDES},
    "candidatos": {f: {cid: cfg for cid, cfg in lista} for f, lista in CANDIDATOS.items()},
    "xgb_comun": XGB_COMUN, "paciencia_lr": PACIENCIA_LR, "empate_ss": EMPATE_SS, "p95_train_cm": P95_TRAIN_CM,
    "seleccion_por_familia": {f"{f}|h{tau}": {k: v for k, v in s.items()} for (f, tau), s in SELECCION.items()},
    "seleccion_global": SELECCION_GLOBAL,
    "features_tabulares": FEAT_TAB, "features_secuencia": FEAT_SEQ,
    "dispositivo_redes": str(DEVICE), "dispositivo_xgb": XGB_DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "versiones": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "sklearn": sklearn.__version__, "xgboost": xgb.__version__, "torch": torch.__version__,
                  "shap": None if shap is None else shap.__version__},
}
configuracion["formulacion"] = {"objetivo": OBJETIVO, "variables": VARIABLES, "peso_alfa": PESO_ALFA,
                                 "solo_busqueda": SOLO_BUSQUEDA, "rf_max_arboles": RF_MAX_ARBOLES or None}
with open(OUT_DIR / "configuracion.json", "w", encoding="utf-8") as f:
    json.dump(configuracion, f, indent=2, ensure_ascii=False, default=str)


def _ruta_rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(p)


# Contrato de carga para la aplicación de tiempo real (src/apps): qué espera cada modelo y cómo volver a nivel
formulacion = {
    "descripcion": "Modelos finales del nivel del @@nombre@@ (estrategia directa, un modelo por horizonte).",
    "target": TARGET, "horizontes_h": HORIZONTES,
    "objetivo": OBJETIVO,
    "reconstruccion_nivel": ("nivel_cm(t+τ) = (x[TARGET](t) + salida) * (max - min) + min" if OBJETIVO == "delta"
                             else "nivel_cm(t+τ) = salida * (max - min) + min"),
    "escala_objetivo_cm": {"min": TGT_MIN, "max": TGT_MAX},
    "pesos": {"alfa": PESO_ALFA, "regla": "w = 1 + alfa * 1[y_obs(t+tau) >= P95 train o lluvia areal(t) >= 0.1 mm]",
              "p95_train_cm": P95_TRAIN_CM, "umbral_lluvia_mm": UMBRAL_LLUVIA_PESO_MM,
              "pluviometros_areal": COLS_LLUVIA},
    "variables": VARIABLES,
    "features_tabulares": FEAT_TAB,
    "features_secuencia": FEAT_SEQ,
    "lookback_h": LOOKBACK_H,
    "entrada_arboles": "matriz (n, len(features_tabulares)) escalada, columnas en este orden, fila = instante t",
    "entrada_redes": "tensor (n, lookback_h, len(features_secuencia)) escalado; último paso = instante t",
    "escaladores": _ruta_rel(DATA_DIR / "escaladores_@@slug@@.csv"),
    "horizontes_json": _ruta_rel(HORIZONTES_JSON),
    "catalogo": _ruta_rel(DATA_DIR / "catalogo_features_@@slug@@.csv"),
    "modulo_variables": "src/data_prep/variables_rio.py (calcular_variables + escalar)",
    "config_variables": CFG_HORIZONTES.get("config_variables"),
    "arquitectura_redes": {"clase": "RegresorRecurrente: nn.LSTM|nn.RNN(batch_first) -> último paso -> "
                                    "Linear(hidden,32) -> ReLU -> Linear(32,1)",
                           "n_variables": len(FEAT_SEQ)},
    "modelos": MODELOS_FINALES,
    "seleccion_global": SELECCION_GLOBAL,
    "versiones": configuracion["versiones"],
}
if not SOLO_BUSQUEDA:
    with open(MODELOS_DIR / "formulacion_@@slug@@.json", "w", encoding="utf-8") as f:
        json.dump(formulacion, f, indent=2, ensure_ascii=False, default=str)
print(f"Predicciones: {len(predicciones):,} filas | modelos: {sorted(predicciones['modelo'].unique())} | horizontes: {HORIZONTES}")
print("Archivos en", OUT_DIR)
'''),
    ("md", r"""
## 10. Revisión en la etapa de entrenamiento

Vista rápida del walk-forward y de las curvas de sobreajuste. La evaluación completa, con test, picos, alertas y
pruebas de Diebold–Mariano, está en `src/metrics/@@slug@@_models_metric.ipynb`.
"""),
    ("code", r'''
columnas = ["familia", "horizonte", "config_id", "SS_medio_folds", "SS_std_folds", "RMSE_medio_folds",
            "RMSE_alto_folds", "n_configs", "mejor_iteracion", "RMSE_train_ajuste_final", "tiempo_final_s"]
display(resumen[[c for c in columnas if c in resumen]].sort_values(["horizonte", "SS_medio_folds"],
                                                                     ascending=[True, False]).round(4))
'''),
    ("code", r'''
mejores = pd.DataFrame([{"familia": f, "horizonte": tau, "config_id": s["config_id"]} for (f, tau), s in SELECCION.items()])
ss_folds = busqueda.merge(mejores, on=["familia", "horizonte", "config_id"])
fig, ejes = plt.subplots(1, len(HORIZONTES), figsize=(7 * len(HORIZONTES), 4), squeeze=False)
for ax, tau in zip(ejes.ravel(), HORIZONTES):
    tabla = ss_folds[ss_folds["horizonte"] == tau].pivot(index="fold", columns="familia", values="SS")
    tabla.plot(ax=ax, marker="o")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"SS por fold — mejor configuración, τ = {tau} h")
    ax.set_ylabel("SS (vs persistencia)")
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
'''),
    ("code", r'''
h_ver = HORIZONTES[0]
claves = [k for k in HISTORIALES if k.endswith(f"|h{h_ver}")]
if claves:
    n_col = 3
    n_fil = int(np.ceil(len(claves) / n_col))
    fig, ejes = plt.subplots(n_fil, n_col, figsize=(16, 3.8 * n_fil), squeeze=False)
    for ax, clave in zip(ejes.ravel(), claves):
        hst = HISTORIALES[clave]
        x = np.arange(1, len(hst["train"]) + 1)
        ax.plot(x, hst["train"], label="train / ajuste")
        ax.plot(x, hst["val"], label="val / control")
        ax.axvline(hst["mejor_iteracion"], color="gray", ls="--", lw=0.8)
        ax.set_title(f"{clave.split('|')[0]} — {hst['origen'].split(':')[0]}", fontsize=10)
        ax.set_xlabel(hst["tipo"])
        ax.set_ylabel("RMSE (cm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in ejes.ravel()[len(claves):]:
        ax.axis("off")
    fig.suptitle(f"Curvas de sobreajuste — τ = {h_ver} h (línea punteada: mejor iteración)")
    fig.tight_layout()
    plt.show()
'''),
]
