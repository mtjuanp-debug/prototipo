# -*- coding: utf-8 -*-
"""Plantilla del notebook de modelos (src/models/modelos_<slug>.ipynb): parte 2 de 3.

Cada celda es (tipo, texto). Los marcadores ``@@clave@@`` los sustituye ``generadores.comun.renderizar`` con
los valores del río en ``config/rios.json``. No editar los notebooks generados: editar aquí y regenerar.
"""

CELDAS = [
    ("code", r'''
def a_tensor(arr):
    # torch.from_numpy falla si PyTorch se compiló con NumPy 1.x y hay NumPy 2.x instalado.
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    try:
        return torch.from_numpy(arr)
    except RuntimeError:
        return torch.tensor(arr.tolist(), dtype=torch.float32)


def a_indices(pos):
    pos = np.ascontiguousarray(pos, dtype=np.int64)
    try:
        return torch.from_numpy(pos).to(DEVICE)
    except RuntimeError:
        return torch.tensor(pos.tolist(), dtype=torch.long, device=DEVICE)


def a_numpy(t):
    t = t.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t.tolist(), dtype=np.float32)


# Variables por paso de tiempo de las redes, en GPU si está disponible (las horas no válidas valen 0,
# pero nunca entran en una ventana porque VENTANA_OK lo impide)
X_SEQ = a_tensor(np.nan_to_num(G[FEAT_SEQ].to_numpy(np.float32))).to(DEVICE)


def preparar_horizonte(tau):
    """Etiqueta y(t) = y_obs(t+τ) sobre la malla y posiciones utilizables para el horizonte τ.

    - ``y``: nivel observado en t+τ (escalado); con él se evalúa todo (RMSE, SS, predicciones en cm).
    - ``yf``: lo que aprende el modelo: Δ = y − nivel(t) (objetivo delta) o y (objetivo nivel).
    - ``base``: lo que se suma a la salida del modelo para volver a nivel (nivel(t) o 0).
    - ``w``: peso de entrenamiento 1 + α·𝟙[y ≥ P95 de train o lluvia areal(t) ≥ 0,1 mm].
    """
    y = np.full(N_MALLA, np.nan, dtype=np.float32)
    y[:-tau] = Y_OBS[tau:]
    misma = np.zeros(N_MALLA, dtype=bool)
    misma[:-tau] = PART[tau:] == PART[:-tau]
    pos = np.flatnonzero(VENTANA_OK & misma & ~np.isnan(y))
    yf = (y - BASE_OBJ).astype(np.float32)
    crecida = (np.nan_to_num(y, nan=-1.0) >= P95_TRAIN_ESC) | (LLUVIA_AREAL_MM >= UMBRAL_LLUVIA_PESO_MM)
    w = (1.0 + PESO_ALFA * crecida).astype(np.float32)
    return {"tau": tau, "y": y, "yf": yf, "base": BASE_OBJ, "w": w, "pos": pos,
            "y_t": a_tensor(np.nan_to_num(yf)).to(DEVICE), "w_t": a_tensor(w).to(DEVICE)}


def pesos(pos, D):
    """sample_weight de las filas pos (None si no se ponderan)."""
    return D["w"][pos] if PESO_ALFA > 0 else None


def particion_fold(pos, k):
    """Fold k (1..5): entrena con train + val anterior al bloque k menos el margen; evalúa en el bloque k."""
    ini, fin = BORDES[k - 1], BORDES[k]
    t, part = TS[pos], PART[pos]
    evaluacion = (part == "val") & (t >= ini) & (t < fin)
    entrenamiento = (part == "train") | ((part == "val") & (t < ini - pd.Timedelta(hours=MARGEN_H)))
    return pos[entrenamiento], pos[evaluacion]


def division_es(pos):
    """Último 10 % cronológico para early stopping (las posiciones ya están en orden temporal)."""
    n_es = max(1, int(round(FRACCION_ES * len(pos))))
    return pos[:-n_es], pos[-n_es:]


def submuestra(pos, n):
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(pos, min(n, len(pos)), replace=False))


def rmse_cm(y_esc, pred_esc):
    return float(np.sqrt(np.mean((np.asarray(pred_esc, float) - np.asarray(y_esc, float)) ** 2)) * RANGO_CM)
'''),
    ("md", r"""
## 2. Registro de resultados

- `PREDICCIONES`: predicciones en cm. `particion = "val_fold"` con su `fold` para las predicciones del
  walk-forward (mejor configuración de cada familia) y `train`/`val`/`test` con `fold` nulo para el modelo final.
- `BUSQUEDA`: una fila por familia × horizonte × configuración × fold, con RMSE (cm) y SS.
- `HISTORIALES`: curvas de sobreajuste en RMSE (cm).
"""),
    ("code", r'''
PREDICCIONES, HISTORIALES, RESUMEN, BUSQUEDA, MODELOS_FINALES = [], {}, [], [], {}
PRED_FOLDS, SELECCION, IMPORTANCIAS, SHAP_FILAS, CURVAS_PROFUNDIDAD = {}, {}, [], [], []
RUTA_BUSQUEDA = OUT_DIR / "busqueda_resultados.csv"


def registrar_predicciones(modelo, familia, tau, nombre_part, pos, y_esc, pred_esc, fold=None):
    """pred_esc siempre en NIVEL escalado (las salidas Δ ya llegan sumadas a nivel(t))."""
    PREDICCIONES.append(pd.DataFrame({
        "timestamp": TS[pos], "particion": nombre_part, "horizonte": tau, "modelo": modelo, "familia": familia,
        "fold": pd.array([fold] * len(pos), dtype="Int64"),
        "y_real_cm": a_cm(y_esc), "y_pred_cm": a_cm(pred_esc)}))
'''),
    ("md", r"""
## 3. Árboles: ajuste e importancias
"""),
    ("code", r'''
def ajustar_rf(cfg, pos_fit, D):
    return RandomForestRegressor(**cfg, n_jobs=-1, random_state=SEED).fit(X_TAB[pos_fit], D["yf"][pos_fit],
                                                                         sample_weight=pesos(pos_fit, D))


def ajustar_ada(cfg, pos_fit, D):
    cfg_modelo = {k: v for k, v in cfg.items() if k != "max_depth"}
    modelo = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=cfg["max_depth"]),
                               random_state=SEED, **cfg_modelo)
    return modelo.fit(X_TAB[pos_fit], D["yf"][pos_fit], sample_weight=pesos(pos_fit, D))


def ajustar_xgb(cfg, pos_fit, pos_es, D, curva_train=False):
    """Early stopping con el último elemento de eval_set (pos_es)."""
    eval_set = [(X_TAB[pos_es], D["yf"][pos_es])]
    if curva_train:
        sub = submuestra(pos_fit, N_CURVA_TRAIN)
        eval_set = [(X_TAB[sub], D["yf"][sub])] + eval_set
    modelo = xgb.XGBRegressor(**XGB_COMUN, **cfg)
    # El early stopping se evalúa sin pesos (RMSE simple, el mismo criterio de la selección)
    modelo.fit(X_TAB[pos_fit], D["yf"][pos_fit], sample_weight=pesos(pos_fit, D), eval_set=eval_set, verbose=False)
    return modelo


def curva_arboles_rf(modelo, pos, D):
    acumulado, errores = np.zeros(len(pos)), []
    X = X_TAB[pos]
    for k, arbol in enumerate(modelo.estimators_, start=1):
        acumulado += arbol.predict(X)
        errores.append(rmse_cm(D["yf"][pos], acumulado / k))
    return errores


def curva_profundidad_rf(cfg, pos_tr, pos_va, D):
    sub = submuestra(pos_tr, N_CURVA_TRAIN)
    filas = []
    for prof in RF_PROFUNDIDADES:
        m = ajustar_rf({**cfg, "n_estimators": RF_ARBOLES_CURVA, "max_depth": prof}, pos_tr, D)
        filas.append({"horizonte": D["tau"], "max_depth": "None" if prof is None else prof,
                      "RMSE_train": rmse_cm(D["yf"][sub], m.predict(X_TAB[sub])),
                      "RMSE_val": rmse_cm(D["yf"][pos_va], m.predict(X_TAB[pos_va]))})
        del m
    return pd.DataFrame(filas)


def importancia_permutacion(familia, modelo, pos_va, D):
    sel = submuestra(pos_va, N_PERM_FILAS)
    # Con Δ el RMSE en el espacio del modelo es igual al RMSE en nivel (nivel(t) es fijo en cada fila)
    r = permutation_importance(modelo, X_TAB[sel], D["yf"][sel], scoring="neg_root_mean_squared_error",
                               n_repeats=N_PERM_REPETICIONES, random_state=SEED, n_jobs=1)
    IMPORTANCIAS.append(pd.DataFrame({"modelo": CORTO[familia], "familia": familia, "horizonte": D["tau"],
                                      "variable": FEAT_TAB, "importancia_cm": r.importances_mean * RANGO_CM,
                                      "desv_cm": r.importances_std * RANGO_CM}))


def importancia_shap(familia, modelo, pos_va, D):
    """Media de |SHAP| (cm) en una submuestra de val. Para RF se usa la aproximación de Saabas por costo."""
    if shap is None:
        return
    sel = submuestra(pos_va, N_SHAP_FILAS)
    t0 = time.time()
    try:
        explicador = shap.TreeExplainer(modelo)
        if familia == "RandomForest":
            valores = explicador.shap_values(X_TAB[sel], approximate=True, check_additivity=False)
        else:
            valores = explicador.shap_values(X_TAB[sel], check_additivity=False)
    except Exception as error:
        print(f"  Aviso: SHAP falló para {familia} h={D['tau']}: {error}")
        return
    SHAP_FILAS.append(pd.DataFrame({"modelo": CORTO[familia], "familia": familia, "horizonte": D["tau"],
                                    "variable": FEAT_TAB, "shap_abs_medio_cm": np.abs(valores).mean(axis=0) * RANGO_CM,
                                    "n_filas": len(sel), "aproximado": familia == "RandomForest"}))
    print(f"  SHAP {familia} h={D['tau']}: {len(sel)} filas de val en {time.time() - t0:.0f} s")
'''),
    ("md", r"""
## 4. Redes recurrentes (RNN base y LSTM)

- Ventana de 24 h consecutivas válidas; el objetivo es Δ (o el nivel observado en t+τ con `OBJETIVO=nivel`).
- Con pesos (α > 0) la pérdida es la MSE ponderada Σ w·e² / Σ w de cada lote; el control usa la MSE simple.
- Cada lote se arma indexando la ventana directamente sobre el tensor de la malla, sin copiar datos.
- Early stopping sobre el conjunto de control (último 10 % cronológico) y se restaura el mejor estado.
"""),
    ("code", r'''
OFFSETS = torch.arange(-LOOKBACK_H + 1, 1, device=DEVICE)


def lotes(pos_t, y_t, batch_size, barajar, generador=None, w_t=None):
    n = len(pos_t)
    orden = torch.randperm(n, generator=generador).to(DEVICE) if barajar else torch.arange(n, device=DEVICE)
    for ini in range(0, n, batch_size):
        filas = pos_t[orden[ini:ini + batch_size]]
        if w_t is None:
            yield X_SEQ[filas[:, None] + OFFSETS], y_t[filas]
        else:
            yield X_SEQ[filas[:, None] + OFFSETS], y_t[filas], w_t[filas]


class RegresorRecurrente(nn.Module):
    def __init__(self, n_variables, hidden_size, num_layers, celda, dropout, y_media):
        super().__init__()
        clase = nn.LSTM if celda == "lstm" else nn.RNN
        self.rnn = clase(input_size=n_variables, hidden_size=hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.cabeza = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
        with torch.no_grad():
            self.cabeza[-1].bias.fill_(float(y_media))

    def forward(self, x):
        salida, _ = self.rnn(x)
        return self.cabeza(salida[:, -1, :]).squeeze(1)


@torch.no_grad()
def predecir_red(modelo, pos, D, batch_size=4096):
    modelo.eval()
    if len(pos) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([a_numpy(modelo(X)) for X, _ in lotes(a_indices(pos), D["y_t"], batch_size, False)])


def entrenar_red(cfg, pos_fit, pos_ctrl, D, etiqueta="", verboso=False):
    torch.manual_seed(SEED)
    generador = torch.Generator().manual_seed(SEED)
    modelo = RegresorRecurrente(len(FEAT_SEQ), cfg["hidden_size"], cfg["num_layers"], cfg["celda"],
                                cfg["dropout"], float(np.mean(D["yf"][pos_fit]))).to(DEVICE)
    clase_opt = torch.optim.AdamW if cfg["optimizador"] == "adamw" else torch.optim.Adam
    opt = clase_opt(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=PACIENCIA_LR)
    criterio = nn.MSELoss()
    pos_fit_t, y_ctrl = a_indices(pos_fit), D["yf"][pos_ctrl]
    hist = {"train": [], "val": [], "lr": [], "s_epoca": []}
    mejor, mejor_estado, espera, mejor_epoca = np.inf, None, 0, 0
    for epoca in range(1, cfg["epocas"] + 1):
        t0 = time.time()
        modelo.train()
        suma, n = 0.0, 0
        for X, y, w in lotes(pos_fit_t, D["y_t"], cfg["batch_size"], barajar=True, generador=generador,
                             w_t=D["w_t"]):
            opt.zero_grad()
            if PESO_ALFA > 0:
                perdida = torch.sum(w * (modelo(X) - y) ** 2) / torch.sum(w)   # MSE ponderada
            else:
                perdida = criterio(modelo(X), y)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            suma += perdida.item() * len(y)
            n += len(y)
        perdida_ctrl = float(np.mean((predecir_red(modelo, pos_ctrl, D) - y_ctrl) ** 2))
        planificador.step(perdida_ctrl)
        hist["train"].append(float(np.sqrt(suma / n) * RANGO_CM))
        hist["val"].append(float(np.sqrt(perdida_ctrl) * RANGO_CM))
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["s_epoca"].append(time.time() - t0)
        if verboso:
            print(f"    {etiqueta} época {epoca:3d} | RMSE ajuste {hist['train'][-1]:.2f} cm | control "
                  f"{hist['val'][-1]:.2f} cm | lr {hist['lr'][-1]:.1e} | {hist['s_epoca'][-1]:.1f} s")
        if perdida_ctrl < mejor:
            mejor, espera, mejor_epoca = perdida_ctrl, 0, epoca
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= cfg["patience"]:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, hist, mejor_epoca
'''),
    ("md", r"""
## 5. Búsqueda aleatoria con walk-forward

Para cada horizonte, familia y configuración se entrena un modelo por fold. La persistencia
(`ŷ(t+τ) = nivel(t)`) se evalúa en las mismas filas de cada bloque para calcular el SS. Los resultados se
guardan de forma incremental en `busqueda_resultados.csv`.
"""),
    ("code", r'''
def correr_fold(familia, cfg, pos_ent, pos_eva, D):
    info = {}
    if familia == "RandomForest":
        modelo = ajustar_rf(cfg, pos_ent, D)
        pred = modelo.predict(X_TAB[pos_eva])
    elif familia == "AdaBoost":
        modelo = ajustar_ada(cfg, pos_ent, D)
        pred = modelo.predict(X_TAB[pos_eva])
    elif familia == "XGBoost":
        pos_fit, pos_es = division_es(pos_ent)
        modelo = ajustar_xgb(cfg, pos_fit, pos_es, D)
        pred = modelo.predict(X_TAB[pos_eva])
        info["mejor_iteracion"] = int(modelo.best_iteration) + 1
    else:
        pos_fit, pos_es = division_es(pos_ent)
        modelo, hist, mejor_epoca = entrenar_red(cfg, pos_fit, pos_es, D)
        pred = predecir_red(modelo, pos_eva, D)
        info.update(mejor_iteracion=mejor_epoca, epocas=len(hist["train"]),
                    s_por_epoca=round(float(np.mean(hist["s_epoca"])), 2))
    del modelo
    return pred, info


def _ada_fold(cfg, X_ent, y_ent, X_eva, w_ent=None):
    """AdaBoost no paraleliza internamente: sus folds se ejecutan en procesos separados."""
    t0 = time.time()
    cfg_modelo = {k: v for k, v in cfg.items() if k != "max_depth"}
    modelo = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=cfg["max_depth"]),
                               random_state=SEED, **cfg_modelo).fit(X_ent, y_ent, sample_weight=w_ent)
    return modelo.predict(X_eva), time.time() - t0


def ejecutar_folds(familia, cfg, folds, D):
    """Devuelve {k: (pred, info, tiempo_s)} o {k: excepción} si el fold falló."""
    salida = {}
    if familia == "AdaBoost" and len(folds) > 1:
        try:
            res = joblib.Parallel(n_jobs=min(len(folds), N_PROCESOS_ADA))(
                joblib.delayed(_ada_fold)(cfg, X_TAB[pe], D["yf"][pe], X_TAB[pv], pesos(pe, D))
                for pe, pv in folds.values())
            return {k: (pred, {}, t) for k, (pred, t) in zip(folds, res)}
        except Exception as error:
            if RAPIDO:
                raise
            return {k: error for k in folds}
    for k, (pos_ent, pos_eva) in folds.items():
        t0 = time.time()
        try:
            pred, info = correr_fold(familia, cfg, pos_ent, pos_eva, D)
            salida[k] = (pred, info, time.time() - t0)
        except Exception as error:
            if RAPIDO:
                raise
            salida[k] = error
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return salida


def buscar(familia, D, folds, rmse_pers):
    tau = D["tau"]
    for config_id, cfg in CANDIDATOS[familia]:
        preds = {}
        for k, r in ejecutar_folds(familia, cfg, folds, D).items():
            pos_eva = folds[k][1]
            if isinstance(r, Exception):
                print(f"    Falla {config_id} fold {k}: {r}")
                rmse, info, tiempo = np.nan, {}, np.nan
            else:
                pred, info, tiempo = r
                pred = pred + D["base"][pos_eva]          # salida del modelo (Δ o nivel) -> nivel
                rmse = rmse_cm(D["y"][pos_eva], pred)
                preds[k] = pred
            BUSQUEDA.append({"familia": familia, "modelo": CORTO[familia], "horizonte": tau,
                             "config_id": config_id, "config_json": json.dumps(cfg), "fold": k,
                             "RMSE": rmse, "SS": 1 - rmse / rmse_pers[k], "n_eval": len(pos_eva),
                             "tiempo_s": round(tiempo, 1), **info})
        PRED_FOLDS[(familia, tau, config_id)] = preds
        filas = [f for f in BUSQUEDA if f["familia"] == familia and f["horizonte"] == tau and f["config_id"] == config_id]
        print(f"  {config_id:9s} h={tau} | SS medio {np.nanmean([f['SS'] for f in filas]):+.4f} | "
              f"RMSE medio {np.nanmean([f['RMSE'] for f in filas]):.2f} cm | {np.nansum([f['tiempo_s'] for f in filas]):.0f} s")
        pd.DataFrame(BUSQUEDA).to_csv(RUTA_BUSQUEDA, index=False)
        gc.collect()


def elegir_mejor(familia, D, folds):
    """Mejor configuración por SS medio entre folds (solo configuraciones que corrieron en todos los folds)."""
    tau = D["tau"]
    r = pd.DataFrame(BUSQUEDA)
    r = r[(r["familia"] == familia) & (r["horizonte"] == tau)]
    g = r.groupby("config_id").agg(SS_medio=("SS", "mean"), SS_std=("SS", "std"), RMSE_medio=("RMSE", "mean"),
                                   n_ok=("SS", "count"))
    g = g[g["n_ok"] == len(folds)]
    if g.empty:
        print(f"  {familia} h={tau}: ninguna configuración completó todos los folds")
        return None
    config_id = g["SS_medio"].idxmax()
    cfg = dict(CANDIDATOS[familia])[config_id]
    # Predicciones del walk-forward de la mejor configuración y RMSE en niveles altos (> P95 de train)
    y_real, y_pred = [], []
    for k, (_, pos_eva) in folds.items():
        pred = PRED_FOLDS[(familia, tau, config_id)][k]
        registrar_predicciones(CORTO[familia], familia, tau, "val_fold", pos_eva, D["y"][pos_eva], pred, fold=k)
        y_real.append(a_cm(D["y"][pos_eva]))
        y_pred.append(a_cm(pred))
    y_real, y_pred = np.concatenate(y_real), np.concatenate(y_pred)
    alto = y_real > P95_TRAIN_CM
    rmse_alto = float(np.sqrt(np.mean((y_pred[alto] - y_real[alto]) ** 2))) if alto.any() else np.nan
    mejor_it = r.loc[r["config_id"] == config_id, "mejor_iteracion"] if "mejor_iteracion" in r else pd.Series(dtype=float)
    SELECCION[(familia, tau)] = {"config_id": config_id, "cfg": cfg, "SS_medio_folds": float(g.loc[config_id, "SS_medio"]),
                                 "SS_std_folds": float(g.loc[config_id, "SS_std"]) if len(folds) > 1 else np.nan,
                                 "RMSE_medio_folds": float(g.loc[config_id, "RMSE_medio"]),
                                 "RMSE_alto_folds": rmse_alto, "n_alto_folds": int(alto.sum()),
                                 "mejor_iteracion_media_folds": float(mejor_it.mean()) if mejor_it.notna().any() else np.nan,
                                 "n_configs": int(len(g))}
    print(f"  >> {familia} h={tau}: mejor {config_id} | SS medio {g.loc[config_id, 'SS_medio']:+.4f} "
          f"| RMSE > P95: {rmse_alto:.2f} cm")
    return config_id
'''),
    ("md", r"""
## 6. Modelo final y diagnóstico

- **Modelo final:** mejor configuración reentrenada con train + val. XGBoost y redes usan early stopping con el
  último 10 % cronológico de train + val, y sus curvas comparan el ajuste con ese tramo de control. Se predice
  train, val y test (**test una sola vez**).
- **Diagnóstico de los árboles:** la mejor configuración se ajusta solo con train y se evalúa en val, que queda
  fuera de muestra. Con ese modelo se calculan las curvas de RF (n.º de árboles y `max_depth`) y de AdaBoost
  (`staged_predict`), la importancia por permutación en val y SHAP (`TreeExplainer`) del mejor árbol
  compatible (RF o XGBoost) en una submuestra de val.
"""),
]
