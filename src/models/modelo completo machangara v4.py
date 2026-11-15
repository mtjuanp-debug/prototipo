# -*- coding: utf-8 -*-

"""
================================================================================
PREDICCIÓN DE NIVEL DE AGUA — ESTACIÓN MACHÁNGARA-LLANTERA
Horizonte de predicción: +3 horas

MODELOS:
    1. Persistencia
    2. Random Forest
    3. XGBoost
    4. AdaBoost
    5. RNN base
    6. LSTM Escenario 1
    7. LSTM Escenario 2
    8. LSTM Escenario 3

INCLUYE:
    - Carga de datos
    - Variable objetivo t+3h
    - Selección de variables
    - Imputación
    - Entrenamiento
    - Métricas Train / Validation / Test
    - Curvas de entrenamiento
    - Alineación temporal de predicciones
    - Tablas del informe
    - Gráficos del informe
    - Importancia de variables
    - Análisis de niveles altos
    - Exportación CSV
    - Exportación Excel
================================================================================
"""

# ==============================================================================
# 0. IMPORTACIONES
# ==============================================================================

import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)


# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ==============================================================================

DATA_DIR = Path(
    r"C:\Users\jpsanche\OneDrive - ETAPA EP\Documentos\Maestría"
    r"\Inteligencia Artificial Aplicada\UIDE\Documentación\Titulación"
    r"\Preprocesamiento\salida_preprocesamiento"
)

OUT_DIR = Path(
    r"C:\Users\jpsanche\OneDrive - ETAPA EP\Documentos\Maestría"
    r"\Inteligencia Artificial Aplicada\UIDE\Documentación\Titulación"
    r"\Preprocesamiento\Modelo entrenado v3\salida_python"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. CONFIGURACIÓN GENERAL
# ==============================================================================

TARGET_COL = "LIM_MachángaraLlantera__Level_Avg"
HORIZONTE_H = 3
LOOKBACK_H = 24
RANDOM_STATE = 42
LOG = OUT_DIR / "log_entrenamiento_completo.txt"

# ==============================================================================
# 3. FUNCIONES GENERALES
# ==============================================================================

def log(msg):
    """
    Imprime y guarda mensajes en el archivo de log.
    """

    line = f"[{time.strftime('%H:%M:%S')}] {msg}"

    print(line, flush=True)

    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def guardar_csv(df, nombre):
    """
    Guarda DataFrame como CSV.
    """

    ruta = OUT_DIR / nombre

    df.to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig"
    )

    log(f"Archivo generado: {ruta}")

    return ruta


def guardar_y_mostrar(fig, nombre_archivo):
    """
    Guarda una figura y además la muestra en Spyder.
    """

    ruta = OUT_DIR / nombre_archivo

    fig.savefig(
        ruta,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Gráfica guardada: {ruta}")

    plt.show()

    plt.close(fig)


def inverse_target(y):
    """
    Revierte el escalado Min-Max de la variable objetivo.
    """

    return y * (tgt_max - tgt_min) + tgt_min


# ==============================================================================
# 4. CARGA DE DATASETS
# ==============================================================================

log("=" * 80)
log("INICIO DEL ENTRENAMIENTO")
log("=" * 80)

log("Cargando datasets...")


train_path = DATA_DIR / "train.parquet"
val_path = DATA_DIR / "val.parquet"
test_path = DATA_DIR / "test.parquet"
scaler_path = DATA_DIR / "escaladores.csv"


train = pd.read_parquet(train_path)

val = pd.read_parquet(val_path)

test = pd.read_parquet(test_path)

escaladores = pd.read_csv(
    scaler_path
).set_index("columna")


log(f"Train: {train.shape}")

log(f"Validation: {val.shape}")

log(f"Test: {test.shape}")


# ==============================================================================
# 5. ESCALADOR DE LA VARIABLE OBJETIVO
# ==============================================================================

tgt_min, tgt_max = escaladores.loc[
    TARGET_COL,
    ["min", "max"]
]


log(
    f"Objetivo: {TARGET_COL}"
)

log(
    f"Min objetivo: {tgt_min}"
)

log(
    f"Max objetivo: {tgt_max}"
)


# ==============================================================================
# 6. DEFINICIÓN DE VARIABLE OBJETIVO t+3h
# ==============================================================================

def con_target(df):

    df = df.copy()

    df["y"] = df[TARGET_COL].shift(
        -HORIZONTE_H
    )

    return df


train = con_target(train)

val = con_target(val)

test = con_target(test)


# ==============================================================================
# 7. INFORMACIÓN TEMPORAL
# ==============================================================================

def detectar_indice_temporal(df):

    if isinstance(df.index, pd.DatetimeIndex):

        return True

    posibles = [
        "timestamp",
        "fecha",
        "datetime",
        "date",
        "Fecha",
        "Timestamp"
    ]

    for col in posibles:

        if col in df.columns:

            try:

                df[col] = pd.to_datetime(
                    df[col]
                )

                df = df.set_index(col)

                return df

            except Exception:

                pass

    return False


# Intentamos identificar índice temporal

train_temporal = detectar_indice_temporal(train)

val_temporal = detectar_indice_temporal(val)

test_temporal = detectar_indice_temporal(test)


if isinstance(train_temporal, pd.DataFrame):

    train = train_temporal

if isinstance(val_temporal, pd.DataFrame):

    val = val_temporal

if isinstance(test_temporal, pd.DataFrame):

    test = test_temporal


# ==============================================================================
# 8. SELECCIÓN DE FEATURES
# ==============================================================================

all_cols = [
    c for c in train.columns
    if c != "y"
]


# ------------------------------------------------------------------------------
# Variables base
# ------------------------------------------------------------------------------

base_cols = [
    c for c in all_cols
    if not any(
        s in c
        for s in [
            "_lag_",
            "_acum_",
            "_delta_"
        ]
    )
]


nan_pct_base = train[
    base_cols
].isna().mean()


base_cols_ok = nan_pct_base[
    nan_pct_base < 0.05
].index.tolist()


# ------------------------------------------------------------------------------
# Variables tabulares
# ------------------------------------------------------------------------------

nan_pct_all = train[
    all_cols
].isna().mean()


tabular_cols_ok = nan_pct_all[
    nan_pct_all < 0.05
].index.tolist()


log(
    f"Features base RNN/LSTM: {len(base_cols_ok)}"
)

log(
    f"Features tabulares RF/XGBoost/AdaBoost: "
    f"{len(tabular_cols_ok)}"
)


# ==============================================================================
# 9. PREPARACIÓN DE DATOS
# ==============================================================================

def preparar(df, cols):

    d = df[
        cols + ["y"]
    ].copy()

    d[cols] = (
        d[cols]
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    return d


train_tab = (
    preparar(
        train,
        tabular_cols_ok
    )
    .dropna(subset=["y"])
)


val_tab = (
    preparar(
        val,
        tabular_cols_ok
    )
    .dropna(subset=["y"])
)


test_tab = (
    preparar(
        test,
        tabular_cols_ok
    )
    .dropna(subset=["y"])
)


train_seq = preparar(
    train,
    base_cols_ok
)


val_seq = preparar(
    val,
    base_cols_ok
)


test_seq = preparar(
    test,
    base_cols_ok
)


# ==============================================================================
# 10. TABLA 1 — PARTICIONES DEL DATASET
# ==============================================================================

def informacion_particion(df):

    total = len(df)

    valid_target = df["y"].notna().sum()

    secuencias = max(
        0,
        valid_target - LOOKBACK_H + 1
    )

    return total, valid_target, secuencias


part_train = informacion_particion(train)

part_val = informacion_particion(val)

part_test = informacion_particion(test)


tabla_1 = pd.DataFrame({

    "Partición": [
        "Entrenamiento",
        "Validación",
        "Prueba"
    ],

    "Horas": [
        part_train[0],
        part_val[0],
        part_test[0]
    ],

    "%": [
        70.0,
        15.0,
        15.0
    ],

    "Horas con objetivo válido": [
        part_train[1],
        part_val[1],
        part_test[1]
    ],

    "Secuencias LSTM 24h": [
        part_train[2],
        part_val[2],
        part_test[2]
    ]
})


guardar_csv(
    tabla_1,
    "tabla_01_particiones.csv"
)


# ==============================================================================
# 11. TABLA 2 — ESTADÍSTICOS DEL NIVEL
# ==============================================================================

def estadisticos_nivel(df):

    valores = inverse_target(
        df["y"].dropna().values
    )

    return {

        "Media": np.mean(valores),

        "Desv. est.": np.std(valores),

        "Mínimo": np.min(valores),

        "Mediana": np.median(valores),

        "P95": np.percentile(
            valores,
            95
        ),

        "P99": np.percentile(
            valores,
            99
        ),

        "Máximo": np.max(valores)

    }


stats_train = estadisticos_nivel(train)

stats_val = estadisticos_nivel(val)

stats_test = estadisticos_nivel(test)


tabla_2 = pd.DataFrame({

    "Partición": [
        "Entrenamiento",
        "Validación",
        "Prueba"
    ],

    "Media": [
        stats_train["Media"],
        stats_val["Media"],
        stats_test["Media"]
    ],

    "Desv. est.": [
        stats_train["Desv. est."],
        stats_val["Desv. est."],
        stats_test["Desv. est."]
    ],

    "Mínimo": [
        stats_train["Mínimo"],
        stats_val["Mínimo"],
        stats_test["Mínimo"]
    ],

    "Mediana": [
        stats_train["Mediana"],
        stats_val["Mediana"],
        stats_test["Mediana"]
    ],

    "P95": [
        stats_train["P95"],
        stats_val["P95"],
        stats_test["P95"]
    ],

    "P99": [
        stats_train["P99"],
        stats_val["P99"],
        stats_test["P99"]
    ],

    "Máximo": [
        stats_train["Máximo"],
        stats_val["Máximo"],
        stats_test["Máximo"]
    ]
})


guardar_csv(
    tabla_2,
    "tabla_02_estadisticos_nivel.csv"
)


# ==============================================================================
# 12. TABLA 3 — VARIABLES
# ==============================================================================

tabla_3 = pd.DataFrame({

    "Conjunto": [
        "Base",
        "Tabular"
    ],

    "Usado por": [
        "RNN y LSTM",
        "Random Forest, XGBoost, AdaBoost"
    ],

    "Criterio": [
        "Sin _lag_, _acum_, _delta_ y <5% NaN",
        "<5% NaN en entrenamiento"
    ],

    "Columnas totales": [
        len(base_cols),
        len(all_cols)
    ],

    "Columnas retenidas": [
        len(base_cols_ok),
        len(tabular_cols_ok)
    ]
})


guardar_csv(
    tabla_3,
    "tabla_03_variables.csv"
)


# ==============================================================================
# 13. RESULTADOS
# ==============================================================================

RESULTADOS = []

HISTORIALES = {}

PRED_TEST = {}

TIEMPOS = {}

IMPORTANCIAS = {}


# ==============================================================================
# 14. FUNCIÓN PARA MÉTRICAS
# ==============================================================================

def registrar_metricas(
    nombre,
    y_true_tr,
    y_pred_tr,
    y_true_va,
    y_pred_va,
    y_true_te,
    y_pred_te
):

    for part, yt, yp in [

        (
            "train",
            y_true_tr,
            y_pred_tr
        ),

        (
            "val",
            y_true_va,
            y_pred_va
        ),

        (
            "test",
            y_true_te,
            y_pred_te
        )

    ]:

        yt_r = inverse_target(
            np.asarray(yt)
        )

        yp_r = inverse_target(
            np.asarray(yp)
        )

        RESULTADOS.append({

            "modelo": nombre,

            "particion": part,

            "MAE":
                mean_absolute_error(
                    yt_r,
                    yp_r
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        yt_r,
                        yp_r
                    )
                ),

            "R2":
                r2_score(
                    yt_r,
                    yp_r
                )

        })

    ultimo = RESULTADOS[-1]

    log(
        f"{nombre} -> "
        f"test MAE={ultimo['MAE']:.2f} cm | "
        f"RMSE={ultimo['RMSE']:.2f} cm | "
        f"R2={ultimo['R2']:.3f}"
    )


# ==============================================================================
# 15. PERSISTENCIA
# ==============================================================================

inicio = time.time()


pred_persistencia = inverse_target(
    test_tab[TARGET_COL].values
)


PRED_TEST[
    "Persistencia"
] = pd.Series(
    pred_persistencia,
    index=test_tab.index
)


# Métricas de persistencia

ytr_persist = train_tab[TARGET_COL].values

yva_persist = val_tab[TARGET_COL].values

yte_persist = test_tab[TARGET_COL].values


registrar_metricas(
    "Persistencia",

    train_tab["y"].values,
    ytr_persist,

    val_tab["y"].values,
    yva_persist,

    test_tab["y"].values,
    yte_persist
)


TIEMPOS["Persistencia"] = 0.0


# ==============================================================================
# 16. RANDOM FOREST
# ==============================================================================

log("=" * 80)

log("ENTRENANDO RANDOM FOREST")

inicio = time.time()


rf = RandomForestRegressor(

    n_estimators=120,

    max_depth=14,

    min_samples_leaf=5,

    max_features=0.5,

    n_jobs=-1,

    random_state=RANDOM_STATE
)


rf.fit(

    train_tab[
        tabular_cols_ok
    ].values,

    train_tab["y"].values
)


pred_tr = rf.predict(
    train_tab[
        tabular_cols_ok
    ].values
)


pred_va = rf.predict(
    val_tab[
        tabular_cols_ok
    ].values
)


pred_te = rf.predict(
    test_tab[
        tabular_cols_ok
    ].values
)


registrar_metricas(
    "RandomForest",

    train_tab["y"].values,
    pred_tr,

    val_tab["y"].values,
    pred_va,

    test_tab["y"].values,
    pred_te
)


PRED_TEST[
    "RandomForest"
] = pd.Series(
    inverse_target(pred_te),
    index=test_tab.index
)


tiempo_rf = time.time() - inicio

TIEMPOS["RandomForest"] = tiempo_rf


log(
    f"Tiempo Random Forest: "
    f"{tiempo_rf:.2f} segundos"
)


imp_rf = pd.Series(

    rf.feature_importances_,

    index=tabular_cols_ok

).sort_values(
    ascending=False
)


IMPORTANCIAS[
    "RandomForest"
] = imp_rf


imp_rf.head(20).to_csv(

    OUT_DIR /
    "rf_importancia_features.csv",

    encoding="utf-8-sig"
)


# ==============================================================================
# 17. XGBOOST
# ==============================================================================

log("=" * 80)

log("ENTRENANDO XGBOOST")

inicio = time.time()


xgb_model = xgb.XGBRegressor(

    n_estimators=1000,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=1,

    reg_lambda=1,

    objective="reg:squarederror",

    early_stopping_rounds=50,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


xgb_model.fit(

    train_tab[
        tabular_cols_ok
    ].values,

    train_tab["y"].values,

    eval_set=[

        (
            val_tab[
                tabular_cols_ok
            ].values,

            val_tab["y"].values
        )

    ],

    verbose=False
)


pred_tr = xgb_model.predict(

    train_tab[
        tabular_cols_ok
    ].values
)


pred_va = xgb_model.predict(

    val_tab[
        tabular_cols_ok
    ].values
)


pred_te = xgb_model.predict(

    test_tab[
        tabular_cols_ok
    ].values
)


registrar_metricas(

    "XGBoost",

    train_tab["y"].values,
    pred_tr,

    val_tab["y"].values,
    pred_va,

    test_tab["y"].values,
    pred_te
)


PRED_TEST[
    "XGBoost"
] = pd.Series(

    inverse_target(pred_te),

    index=test_tab.index
)


tiempo_xgb = time.time() - inicio

TIEMPOS["XGBoost"] = tiempo_xgb


log(
    f"Tiempo XGBoost: "
    f"{tiempo_xgb:.2f} segundos"
)


imp_xgb = pd.Series(

    xgb_model.feature_importances_,

    index=tabular_cols_ok

).sort_values(
    ascending=False
)


IMPORTANCIAS[
    "XGBoost"
] = imp_xgb


imp_xgb.head(20).to_csv(

    OUT_DIR /
    "xgb_importancia_features.csv",

    encoding="utf-8-sig"
)


# ==============================================================================
# 18. ADABOOST
# ==============================================================================

log("=" * 80)

log("ENTRENANDO ADABOOST")

inicio = time.time()


ada = AdaBoostRegressor(

    estimator=DecisionTreeRegressor(
        max_depth=3
    ),

    n_estimators=200,

    learning_rate=0.1,

    loss="linear",

    random_state=RANDOM_STATE
)


ada.fit(

    train_tab[
        tabular_cols_ok
    ].values,

    train_tab["y"].values
)


pred_tr = ada.predict(

    train_tab[
        tabular_cols_ok
    ].values
)


pred_va = ada.predict(

    val_tab[
        tabular_cols_ok
    ].values
)


pred_te = ada.predict(

    test_tab[
        tabular_cols_ok
    ].values
)


registrar_metricas(

    "AdaBoost",

    train_tab["y"].values,
    pred_tr,

    val_tab["y"].values,
    pred_va,

    test_tab["y"].values,
    pred_te
)


PRED_TEST[
    "AdaBoost"
] = pd.Series(

    inverse_target(pred_te),

    index=test_tab.index
)


tiempo_ada = time.time() - inicio

TIEMPOS["AdaBoost"] = tiempo_ada


# ==============================================================================
# 19. DATASET PARA RNN / LSTM
# ==============================================================================

class SecuenciaDataset(Dataset):

    def __init__(
        self,
        df,
        feature_cols,
        lookback
    ):

        self.X = df[
            feature_cols
        ].values.astype(
            np.float32
        )

        self.y = df[
            "y"
        ].values.astype(
            np.float32
        )

        self.lookback = lookback

        valid = np.where(
            ~np.isnan(self.y)
        )[0]

        self.indices = [

            i

            for i in valid

            if i >= lookback - 1

        ]


    def __len__(self):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        idx
    ):

        i = self.indices[idx]

        seq = self.X[
            i - self.lookback + 1:
            i + 1
        ]

        return (

            torch.from_numpy(seq),

            torch.tensor(
                self.y[i]
            )

        )


# ==============================================================================
# 20. MODELO RECURRENTE
# ==============================================================================

class RecurrentRegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        cell="lstm",
        dropout=0.2,
        y_mean=0.0
    ):

        super().__init__()

        self.cell = cell.lower()

        dropout_real = (
            dropout
            if num_layers > 1
            else 0.0
        )


        if self.cell == "lstm":

            self.rnn = nn.LSTM(

                input_size=input_size,

                hidden_size=hidden_size,

                num_layers=num_layers,

                batch_first=True,

                dropout=dropout_real
            )

        elif self.cell == "rnn":

            self.rnn = nn.RNN(

                input_size=input_size,

                hidden_size=hidden_size,

                num_layers=num_layers,

                batch_first=True,

                nonlinearity="tanh",

                dropout=dropout_real
            )

        else:

            raise ValueError(
                "cell debe ser 'rnn' o 'lstm'"
            )


        self.fc1 = nn.Linear(
            hidden_size,
            32
        )


        self.relu = nn.ReLU()


        self.fc2 = nn.Linear(
            32,
            1
        )


        # Inicialización del sesgo
        # con la media del objetivo
        with torch.no_grad():

            self.fc2.bias.fill_(
                float(y_mean)
            )


    def forward(self, x):

        output, _ = self.rnn(x)

        last = output[:, -1, :]

        h = self.fc1(last)

        h = self.relu(h)

        out = self.fc2(h)

        return out.squeeze(1)


# ==============================================================================
# 21. DATASETS RECURRENTES
# ==============================================================================

ds_train = SecuenciaDataset(

    train_seq,

    base_cols_ok,

    LOOKBACK_H
)


ds_val = SecuenciaDataset(

    val_seq,

    base_cols_ok,

    LOOKBACK_H
)


ds_test = SecuenciaDataset(

    test_seq,

    base_cols_ok,

    LOOKBACK_H
)


log(
    f"Secuencias train: {len(ds_train)}"
)

log(
    f"Secuencias val: {len(ds_val)}"
)

log(
    f"Secuencias test: {len(ds_test)}"
)


# ==============================================================================
# 22. DEVICE
# ==============================================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


log(
    f"Dispositivo utilizado: {device}"
)


# ==============================================================================
# 23. MEDIA DEL TARGET
# ==============================================================================

y_mean_train = float(

    np.nanmean(
        ds_train.y
    )

)


# ==============================================================================
# 24. FUNCIÓN DE EVALUACIÓN
# ==============================================================================

def evaluar(
    model,
    loader
):

    model.eval()

    ys = []

    ps = []


    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            pred = model(X)

            ys.extend(
                y.numpy()
            )

            ps.extend(
                pred.cpu().numpy()
            )


    return (

        np.asarray(ys),

        np.asarray(ps)

    )


# ==============================================================================
# 25. ENTRENAMIENTO RNN/LSTM
# ==============================================================================

def entrenar_recurrente(

    nombre,

    cell,

    hidden_size,

    num_layers,

    dropout,

    batch_size,

    epochs,

    patience,

    lr,

    weight_decay=0.0

):

    log("=" * 80)

    log(
        f"ENTRENANDO {nombre}"
    )


    inicio = time.time()


    dl_train = DataLoader(

        ds_train,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=False
    )


    dl_val = DataLoader(

        ds_val,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=False
    )


    dl_test = DataLoader(

        ds_test,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=False
    )


    model = RecurrentRegressor(

        input_size=len(base_cols_ok),

        hidden_size=hidden_size,

        num_layers=num_layers,

        cell=cell,

        dropout=dropout,

        y_mean=y_mean_train
    ).to(device)


    criterion = nn.MSELoss()


    if weight_decay > 0:

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=lr,

            weight_decay=weight_decay

        )

    else:

        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=lr

        )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=2

    )


    best_val = np.inf

    best_state = None

    wait = 0


    historial = {

        "train_loss": [],

        "val_loss": [],

        "lr": []

    }


    for epoch in range(1, epochs + 1):

        model.train()

        train_losses = []


        for X, y in dl_train:

            X = X.to(device)

            y = y.to(device)


            optimizer.zero_grad()


            pred = model(X)


            loss = criterion(
                pred,
                y
            )


            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            optimizer.step()


            train_losses.append(
                loss.item()
            )


        train_loss = float(
            np.mean(train_losses)
        )


        # ----------------------------------------------------------
        # VALIDACIÓN
        # ----------------------------------------------------------

        model.eval()

        val_losses = []


        with torch.no_grad():

            for X, y in dl_val:

                X = X.to(device)

                y = y.to(device)


                pred = model(X)


                loss = criterion(
                    pred,
                    y
                )


                val_losses.append(
                    loss.item()
                )


        val_loss = float(
            np.mean(val_losses)
        )


        scheduler.step(
            val_loss
        )


        lr_actual = optimizer.param_groups[0]["lr"]


        historial[
            "train_loss"
        ].append(train_loss)


        historial[
            "val_loss"
        ].append(val_loss)


        historial[
            "lr"
        ].append(lr_actual)


        log(

            f"{nombre} | "

            f"Época {epoch:02d}/{epochs} | "

            f"Train={train_loss:.6f} | "

            f"Val={val_loss:.6f} | "

            f"LR={lr_actual:.6g}"

        )


        # ----------------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------------

        if val_loss < best_val:

            best_val = val_loss

            best_state = {

                k: v.detach().cpu().clone()

                for k, v in model.state_dict().items()

            }

            wait = 0

        else:

            wait += 1


        if wait >= patience:

            log(
                f"{nombre}: "
                f"Early stopping en época {epoch}"
            )

            break


    # Restaurar mejor estado

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    # ----------------------------------------------------------
    # MÉTRICAS
    # ----------------------------------------------------------

    yt_tr, yp_tr = evaluar(
        model,
        dl_train
    )


    yt_va, yp_va = evaluar(
        model,
        dl_val
    )


    yt_te, yp_te = evaluar(
        model,
        dl_test
    )


    registrar_metricas(

        nombre,

        yt_tr,
        yp_tr,

        yt_va,
        yp_va,

        yt_te,
        yp_te

    )


    # ----------------------------------------------------------
    # PREDICCIONES ALINEADAS
    # ----------------------------------------------------------

    indices_test = test_seq.index[
        ds_test.indices
    ]


    PRED_TEST[
        nombre
    ] = pd.Series(

        inverse_target(yp_te),

        index=indices_test

    )


    HISTORIALES[
        nombre
    ] = historial


    tiempo_total = (
        time.time() - inicio
    )


    TIEMPOS[
        nombre
    ] = tiempo_total


    log(
        f"{nombre}: "
        f"Tiempo total = "
        f"{tiempo_total:.2f} segundos"
    )


    return model


# ==============================================================================
# 26. ENTRENAMIENTO DE LOS MODELOS RECURRENTES
# ==============================================================================

rnn_base = entrenar_recurrente(

    nombre="RNN base",

    cell="rnn",

    hidden_size=64,

    num_layers=2,

    dropout=0.2,

    batch_size=512,

    epochs=12,

    patience=3,

    lr=1e-3

)


lstm_1 = entrenar_recurrente(

    nombre="LSTM Escenario 1",

    cell="lstm",

    hidden_size=64,

    num_layers=2,

    dropout=0.2,

    batch_size=512,

    epochs=12,

    patience=3,

    lr=1e-3

)


lstm_2 = entrenar_recurrente(

    nombre="LSTM Escenario 2",

    cell="lstm",

    hidden_size=32,

    num_layers=3,

    dropout=0.2,

    batch_size=32,

    epochs=25,

    patience=5,

    lr=1e-3

)


lstm_3 = entrenar_recurrente(

    nombre="LSTM Escenario 3",

    cell="lstm",

    hidden_size=128,

    num_layers=3,

    dropout=0.2,

    batch_size=256,

    epochs=40,

    patience=6,

    lr=1e-4,

    weight_decay=1e-4

)


# ==============================================================================
# 27. TABLA 4 — HIPERPARÁMETROS DE ÁRBOLES
# ==============================================================================

tabla_4 = pd.DataFrame({

    "Modelo": [

        "Random Forest",

        "XGBoost",

        "AdaBoost"

    ],

    "Hiperparámetro": [

        "n_estimators",

        "n_estimators / early stopping",

        "n_estimators"

    ],

    "Aplicado": [

        "120",

        "1000 / 50",

        "200"

    ],

    "Otros parámetros": [

        "max_depth=14; min_samples_leaf=5; max_features=0.5",

        "learning_rate=0.05; max_depth=6; subsample=0.8; colsample_bytree=0.8",

        "DecisionTree(max_depth=3); learning_rate=0.1; loss=linear"

    ]

})


guardar_csv(
    tabla_4,
    "tabla_04_hiperparametros_arboles.csv"
)


# ==============================================================================
# 28. TABLA 5 — HIPERPARÁMETROS RECURRENTES
# ==============================================================================

tabla_5 = pd.DataFrame({

    "Parámetro": [

        "Celda recurrente",

        "hidden_size",

        "num_layers",

        "dropout",

        "batch_size",

        "Épocas máximas",

        "patience",

        "learning_rate",

        "Optimizador"

    ],

    "RNN base": [

        "RNN tanh",

        64,

        2,

        0.2,

        512,

        12,

        3,

        "1e-3",

        "Adam"

    ],

    "LSTM Escenario 1": [

        "LSTM",

        64,

        2,

        0.2,

        512,

        12,

        3,

        "1e-3",

        "Adam"

    ],

    "LSTM Escenario 2": [

        "LSTM",

        32,

        3,

        0.2,

        32,

        25,

        5,

        "1e-3",

        "Adam"

    ],

    "LSTM Escenario 3": [

        "LSTM",

        128,

        3,

        0.2,

        256,

        40,

        6,

        "1e-4",

        "AdamW"

    ]

})


guardar_csv(
    tabla_5,
    "tabla_05_hiperparametros_recurrentes.csv"
)


# ==============================================================================
# 29. TABLA 6 — DESVIACIONES
# ==============================================================================

tabla_6 = pd.DataFrame({

    "Elemento": [

        "Random Forest",

        "LSTM Escenario 2",

        "LSTM Escenario 3",

        "Variables de entrada",

        "Búsqueda hiperparámetros",

        "Semillas"

    ],

    "Aplicado": [

        "120 árboles; max_depth=14; min_samples_leaf=5",

        "25 épocas; patience=5",

        "40 épocas; patience=6",

        f"{len(base_cols_ok)} variables base",

        "No ejecutada",

        "Una semilla: 42"

    ],

    "Motivo": [

        "Reducir tiempo de ejecución",

        "Reducir costo computacional",

        "Reducir costo computacional",

        "Filtro de completitud <5%",

        "Se utilizaron configuraciones iniciales",

        "No se evaluó variabilidad entre semillas"

    ]

})


guardar_csv(
    tabla_6,
    "tabla_06_desviaciones.csv"
)


# ==============================================================================
# 30. TABLA 7 — ITERACIÓN PRELIMINAR
# ==============================================================================

tabla_7 = pd.DataFrame({

    "Modelo": [

        "Persistencia",

        "Random Forest preliminar",

        "LSTM v1",

        "LSTM corregido"

    ],

    "MAE": [

        1.63,

        1.54,

        8.50,

        4.87

    ],

    "RMSE": [

        3.72,

        3.21,

        9.51,

        5.94

    ],

    "R2": [

        0.894,

        0.921,

        0.308,

        0.730

    ],

    "Comentario": [

        "Referencia mínima",

        "Configuración preliminar",

        "Sesgo aditivo aproximado +8 cm",

        "Bias inicial + clipping + scheduler"

    ]

})


guardar_csv(
    tabla_7,
    "tabla_07_iteracion_preliminar.csv"
)


# ==============================================================================
# 31. TABLA 8 — MÉTRICAS COMPLETAS
# ==============================================================================

resultados_df = pd.DataFrame(
    RESULTADOS
)


guardar_csv(
    resultados_df,
    "tabla_08_metricas_completas.csv"
)


guardar_csv(
    resultados_df,
    "resultados_metricas_completo.csv"
)


# ==============================================================================
# 32. TABLA 9 — RANKING DE PRUEBA
# ==============================================================================

test_metrics = resultados_df[
    resultados_df["particion"] == "test"
].copy()


val_metrics = resultados_df[
    resultados_df["particion"] == "val"
].copy()


tabla_9 = test_metrics.merge(

    val_metrics[
        [
            "modelo",
            "MAE",
            "RMSE"
        ]
    ],

    on="modelo",

    suffixes=(
        "_test",
        "_val"
    )

)


tabla_9[
    "RMSE_vs_persistencia_%"
] = np.nan


rmse_persistencia = float(

    tabla_9.loc[
        tabla_9["modelo"]
        == "Persistencia",
        "RMSE_test"
    ].iloc[0]

)


tabla_9[
    "RMSE_vs_persistencia_%"
] = (

    (
        tabla_9["RMSE_test"]
        / rmse_persistencia
    ) - 1
) * 100


tabla_9[
    "MAE_test_val"
] = (

    tabla_9["MAE_test"]
    /
    tabla_9["MAE_val"]
)


tabla_9[
    "Tiempo_s"
] = (

    tabla_9["modelo"]
    .map(TIEMPOS)
    .fillna(0)
)


tabla_9 = tabla_9.sort_values(
    "RMSE_test"
).reset_index(
    drop=True
)


tabla_9.insert(
    0,
    "#",
    np.arange(
        1,
        len(tabla_9) + 1
    )
)


guardar_csv(
    tabla_9,
    "tabla_09_ranking_prueba.csv"
)


# ==============================================================================
# 33. PREDICCIONES ALINEADAS
# ==============================================================================

pred_df = pd.DataFrame(
    index=test_tab.index
)


pred_df[
    "y_real"
] = inverse_target(
    test_tab["y"].values
)


for nombre, serie in PRED_TEST.items():

    pred_df[
        nombre
    ] = serie.reindex(
        pred_df.index
    )


pred_df.index.name = "timestamp"


pred_test_df = pred_df.reset_index()


# ==============================================================================
# TIMESTAMP OBJETIVO
# ==============================================================================

# Convertir timestamp a datetime de forma segura.
# Se utiliza pandas porque permite manejar correctamente
# datetime64[ns, UTC].

pred_test_df["timestamp"] = pd.to_datetime(
    pred_test_df["timestamp"],
    errors="coerce",
    utc=True
)


# Timestamp desde el cual se realiza la predicción
pred_test_df["timestamp_origen"] = (
    pred_test_df["timestamp"]
)


# Timestamp objetivo = timestamp de origen + 3 horas
pred_test_df["timestamp_objetivo"] = (
    pred_test_df["timestamp"]
    +
    pd.Timedelta(
        hours=HORIZONTE_H
    )
)

# ==============================================================================
# 34. TABLA 10 — SESGO Y NIVELES ALTOS
# ==============================================================================

p95_test = stats_test["P95"]


pred_cols = [
    c
    for c in pred_df.columns
    if c != "y_real"
]


filtrado_alto = (
    pred_df["y_real"]
    >= p95_test
)


tabla_10_filas = []


for modelo in pred_cols:

    tmp = pred_df[
        [
            "y_real",
            modelo
        ]
    ].dropna()


    if len(tmp) == 0:

        continue


    error = (
        tmp[modelo]
        -
        tmp["y_real"]
    )


    tmp_alto = tmp[
        tmp["y_real"]
        >= p95_test
    ]


    error_alto = (
        tmp_alto[modelo]
        -
        tmp_alto["y_real"]
    )


    # Máximo real del periodo
    idx_pico = (
        tmp["y_real"]
        .idxmax()
    )


    real_pico = float(
        tmp.loc[
            idx_pico,
            "y_real"
        ]
    )


    pred_pico = float(
        tmp.loc[
            idx_pico,
            modelo
        ]
    )


    tabla_10_filas.append({

        "Modelo": modelo,

        "Sesgo medio prueba":
            error.mean(),

        "MAE >= P95":
            np.mean(
                np.abs(
                    error_alto
                )
            )
            if len(error_alto)
            else np.nan,

        "RMSE >= P95":
            np.sqrt(
                np.mean(
                    error_alto ** 2
                )
            )
            if len(error_alto)
            else np.nan,

        "Sesgo >= P95":
            error_alto.mean()
            if len(error_alto)
            else np.nan,

        "Máximo real":
            real_pico,

        "Máximo predicho en pico":
            pred_pico,

        "Error en pico":
            pred_pico - real_pico

    })


tabla_10 = pd.DataFrame(
    tabla_10_filas
)


guardar_csv(
    tabla_10,
    "tabla_10_sesgo_niveles_altos.csv"
)


# ==============================================================================
# 35. TABLA 11 — IMPORTANCIA DE VARIABLES
# ==============================================================================

top_rf = (

    IMPORTANCIAS[
        "RandomForest"
    ]
    .head(10)
)


top_xgb = (

    IMPORTANCIAS[
        "XGBoost"
    ]
    .head(10)
)


tabla_11 = pd.DataFrame({

    "#": range(
        1,
        11
    ),

    "Random Forest Variable":
        top_rf.index.tolist(),

    "Random Forest Importancia":
        top_rf.values,

    "XGBoost Variable":
        top_xgb.index.tolist(),

    "XGBoost Importancia":
        top_xgb.values

})


guardar_csv(
    tabla_11,
    "tabla_11_importancia_variables.csv"
)


# ==============================================================================
# 36. TABLA 12 — ARCHIVOS GENERADOS
# ==============================================================================

tabla_12 = pd.DataFrame({

    "Archivo": [

        "modelo_completo_machangara.py",

        "resultados_metricas_completo.csv",

        "predicciones_test_completo.csv",

        "historiales_entrenamiento.json",

        "rf_importancia_features.csv",

        "xgb_importancia_features.csv",

        "tabla_01_particiones.csv",

        "tabla_02_estadisticos_nivel.csv",

        "tabla_03_variables.csv",

        "tabla_04_hiperparametros_arboles.csv",

        "tabla_05_hiperparametros_recurrentes.csv",

        "tabla_06_desviaciones.csv",

        "tabla_07_iteracion_preliminar.csv",

        "tabla_08_metricas_completas.csv",

        "tabla_09_ranking_prueba.csv",

        "tabla_10_sesgo_niveles_altos.csv",

        "tabla_11_importancia_variables.csv"

    ],

    "Contenido": [

        "Script completo",

        "Métricas Train/Val/Test",

        "Predicciones de prueba",

        "Historial de entrenamiento",

        "Importancia RF",

        "Importancia XGBoost",

        "Particiones",

        "Estadísticos",

        "Variables",

        "Hiperparámetros árboles",

        "Hiperparámetros recurrentes",

        "Desviaciones",

        "Iteración preliminar",

        "Métricas completas",

        "Ranking",

        "Sesgo y niveles altos",

        "Importancia de variables"

    ]

})


guardar_csv(
    tabla_12,
    "tabla_12_archivos_generados.csv"
)


# ==============================================================================
# 37. HISTORIALES JSON
# ==============================================================================

with open(

    OUT_DIR /
    "historiales_entrenamiento.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        HISTORIALES,
        f,
        indent=2
    )


# ==============================================================================
# 38. EXPORTAR TODAS LAS TABLAS A EXCEL
# ==============================================================================

excel_path = (
    OUT_DIR /
    "Informe_Tablas_Machangara.xlsx"
)


with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    tabla_1.to_excel(
        writer,
        sheet_name="01_Particiones",
        index=False
    )

    tabla_2.to_excel(
        writer,
        sheet_name="02_Estadisticos",
        index=False
    )

    tabla_3.to_excel(
        writer,
        sheet_name="03_Variables",
        index=False
    )

    tabla_4.to_excel(
        writer,
        sheet_name="04_Hiper_Arboles",
        index=False
    )

    tabla_5.to_excel(
        writer,
        sheet_name="05_Hiper_RNN_LSTM",
        index=False
    )

    tabla_6.to_excel(
        writer,
        sheet_name="06_Desviaciones",
        index=False
    )

    tabla_7.to_excel(
        writer,
        sheet_name="07_Preliminar",
        index=False
    )

    resultados_df.to_excel(
        writer,
        sheet_name="08_Metricas",
        index=False
    )

    tabla_9.to_excel(
        writer,
        sheet_name="09_Ranking",
        index=False
    )

    tabla_10.to_excel(
        writer,
        sheet_name="10_Niveles_Altos",
        index=False
    )

    tabla_11.to_excel(
        writer,
        sheet_name="11_Importancia",
        index=False
    )

    tabla_12.to_excel(
        writer,
        sheet_name="12_Archivos",
        index=False
    )


log(
    f"Excel generado: {excel_path}"
)


# ==============================================================================
# 39. FIGURA 1 — PARTICIÓN TEMPORAL
# ==============================================================================

fig = plt.figure(
    figsize=(14, 6)
)


# Utilizamos la variable objetivo de cada partición

x_train = train.index

x_val = val.index

x_test = test.index


y_train_plot = inverse_target(
    train[TARGET_COL].values
)

y_val_plot = inverse_target(
    val[TARGET_COL].values
)

y_test_plot = inverse_target(
    test[TARGET_COL].values
)


plt.plot(
    x_train,
    y_train_plot,
    label="Entrenamiento",
    linewidth=0.8
)


plt.plot(
    x_val,
    y_val_plot,
    label="Validación",
    linewidth=0.8
)


plt.plot(
    x_test,
    y_test_plot,
    label="Prueba",
    linewidth=0.8
)


plt.title(
    "Figura 1 — Nivel horario y partición cronológica 70/15/15"
)


plt.xlabel(
    "Fecha"
)


plt.ylabel(
    "Nivel"
)


plt.legend()


plt.grid(
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "01_particion_cronologica.png"
)


# ==============================================================================
# 40. FIGURA 2 — DISTRIBUCIÓN POR PARTICIÓN
# ==============================================================================

fig = plt.figure(
    figsize=(10, 6)
)


plt.boxplot(

    [
        inverse_target(
            train["y"].dropna().values
        ),

        inverse_target(
            val["y"].dropna().values
        ),

        inverse_target(
            test["y"].dropna().values
        )
    ],

    tick_labels=[
        "Entrenamiento",
        "Validación",
        "Prueba"
    ],

    showfliers=False
)


plt.title(
    "Figura 2 — Distribución del nivel por partición"
)


plt.ylabel(
    "Nivel"
)


plt.grid(
    axis="y",
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "02_distribucion_particiones.png"
)


# ==============================================================================
# 41. FIGURA 3 — CURVAS DE ENTRENAMIENTO
# ==============================================================================

fig = plt.figure(
    figsize=(12, 7)
)


for nombre, hist in HISTORIALES.items():

    plt.plot(

        hist["train_loss"],

        label=f"{nombre} - Train"

    )

    plt.plot(

        hist["val_loss"],

        linestyle="--",

        label=f"{nombre} - Val"

    )


plt.title(
    "Figura 3 — Curvas de entrenamiento y validación"
)


plt.xlabel(
    "Época"
)


plt.ylabel(
    "MSE"
)


plt.legend(
    fontsize=8
)


plt.grid(
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "03_curvas_entrenamiento_validacion.png"
)


# ==============================================================================
# 42. FIGURA 4 — MAE
# ==============================================================================

metricas_plot = resultados_df.copy()


fig = plt.figure(
    figsize=(13, 7)
)


for part in [
    "train",
    "val",
    "test"
]:

    d = metricas_plot[
        metricas_plot[
            "particion"
        ] == part
    ]

    plt.plot(

        d["modelo"],

        d["MAE"],

        marker="o",

        label=part

    )


plt.title(
    "Figura 4 — MAE por modelo y partición"
)


plt.xlabel(
    "Modelo"
)


plt.ylabel(
    "MAE"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.legend()


plt.grid(
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "04_MAE_modelos_particiones.png"
)


# ==============================================================================
# 43. FIGURA 5 — R2 TEST
# ==============================================================================

fig = plt.figure(
    figsize=(12, 6)
)


datos_r2 = resultados_df[
    resultados_df["particion"]
    == "test"
].copy()


plt.bar(

    datos_r2["modelo"],

    datos_r2["R2"]

)


plt.title(
    "Figura 5 — R² de los modelos en prueba"
)


plt.ylabel(
    "R²"
)


plt.xlabel(
    "Modelo"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.grid(
    axis="y",
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "05_R2_modelos_test.png"
)


# ==============================================================================
# 44. FIGURA 6 — REAL VS PREDICCIONES
# ==============================================================================

modelos_grafica = [

    "Persistencia",

    "RandomForest",

    "XGBoost",

    "LSTM Escenario 2"

]


fig = plt.figure(
    figsize=(15, 7)
)


plt.plot(

    pred_df.index,

    pred_df["y_real"],

    label="Real",

    linewidth=1.2

)


for modelo in modelos_grafica:

    if modelo in pred_df.columns:

        plt.plot(

            pred_df.index,

            pred_df[modelo],

            label=modelo,

            linewidth=0.8

        )


plt.title(
    "Figura 6 — Nivel observado y predicho a +3 horas"
)


plt.xlabel(
    "Fecha / hora"
)


plt.ylabel(
    "Nivel"
)


plt.legend()


plt.grid(
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "06_REAL_vs_PREDICCIONES.png"
)


# ==============================================================================
# 45. FIGURA 7 — ACERCAMIENTO AL PICO
# ==============================================================================

if len(pred_df) > 0:

    idx_max = pred_df[
        "y_real"
    ].idxmax()


    if isinstance(
        pred_df.index,
        pd.DatetimeIndex
    ):

        inicio_evento = (
            pred_df.index[
                pred_df.index.get_loc(
                    idx_max
                )
            ]
            -
            pd.Timedelta(
                hours=48
            )
        )

        fin_evento = (
            pred_df.index[
                pred_df.index.get_loc(
                    idx_max
                )
            ]
            +
            pd.Timedelta(
                hours=48
            )
        )


        evento = pred_df.loc[
            (pred_df.index >= inicio_evento)
            &
            (pred_df.index <= fin_evento)
        ]

    else:

        pos = pred_df.index.get_loc(
            idx_max
        )

        inicio_pos = max(
            0,
            pos - 48
        )

        fin_pos = min(
            len(pred_df),
            pos + 49
        )

        evento = pred_df.iloc[
            inicio_pos:fin_pos
        ]


    fig = plt.figure(
        figsize=(14, 7)
    )


    plt.plot(

        evento.index,

        evento["y_real"],

        label="Real",

        linewidth=1.5

    )


    for modelo in modelos_grafica:

        if modelo in evento.columns:

            plt.plot(

                evento.index,

                evento[modelo],

                label=modelo

            )


    plt.title(
        "Figura 7 — Acercamiento al evento de máximo nivel"
    )


    plt.xlabel(
        "Fecha / hora"
    )


    plt.ylabel(
        "Nivel"
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    guardar_y_mostrar(
        fig,
        "07_EVENTO_MAXIMO.png"
    )


# ==============================================================================
# 46. FIGURA 8 — DISPERSIÓN REAL VS PREDICHO
# ==============================================================================

fig = plt.figure(
    figsize=(10, 8)
)


for modelo in [

    "RandomForest",

    "XGBoost",

    "LSTM Escenario 2"

]:

    if modelo not in pred_df.columns:

        continue


    tmp = pred_df[
        [
            "y_real",
            modelo
        ]
    ].dropna()


    plt.scatter(

        tmp["y_real"],

        tmp[modelo],

        s=8,

        alpha=0.35,

        label=modelo

    )


# Línea ideal

limites = [

    pred_df["y_real"].min(),

    pred_df["y_real"].max()

]


plt.plot(

    limites,

    limites,

    linestyle="--",

    label="Predicción ideal"

)


plt.title(
    "Figura 8 — Nivel real frente al nivel predicho"
)


plt.xlabel(
    "Nivel real"
)


plt.ylabel(
    "Nivel predicho"
)


plt.legend()


plt.grid(
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "08_DISPERSION_REAL_VS_PREDICHO.png"
)


# ==============================================================================
# 47. FIGURA 9 — IMPORTANCIA DE VARIABLES
# ==============================================================================

fig = plt.figure(
    figsize=(14, 8)
)


# Top 10 RF

rf_top = IMPORTANCIAS[
    "RandomForest"
].head(10).sort_values()


plt.barh(

    rf_top.index,

    rf_top.values,

    alpha=0.7,

    label="Random Forest"

)


plt.title(
    "Figura 9 — Diez variables más importantes de Random Forest"
)


plt.xlabel(
    "Importancia"
)


plt.ylabel(
    "Variable"
)


plt.legend()


plt.grid(
    axis="x",
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "09_IMPORTANCIA_RANDOM_FOREST.png"
)


# ------------------------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------------------------

fig = plt.figure(
    figsize=(14, 8)
)


xgb_top = IMPORTANCIAS[
    "XGBoost"
].head(10).sort_values()


plt.barh(

    xgb_top.index,

    xgb_top.values,

    alpha=0.7,

    label="XGBoost"

)


plt.title(
    "Figura 9b — Diez variables más importantes de XGBoost"
)


plt.xlabel(
    "Importancia"
)


plt.ylabel(
    "Variable"
)


plt.legend()


plt.grid(
    axis="x",
    alpha=0.3
)


guardar_y_mostrar(
    fig,
    "09b_IMPORTANCIA_XGBOOST.png"
)


# ==============================================================================
# 48. GRÁFICOS INDIVIDUALES DE LOSS
# ==============================================================================

for nombre, hist in HISTORIALES.items():

    fig = plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        hist["train_loss"],

        label="Train"

    )


    plt.plot(

        hist["val_loss"],

        label="Validación"

    )


    plt.title(
        f"Curva de entrenamiento — {nombre}"
    )


    plt.xlabel(
        "Época"
    )


    plt.ylabel(
        "MSE"
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    nombre_seguro = (
        nombre
        .replace(
            " ",
            "_"
        )
    )


    guardar_y_mostrar(

        fig,

        f"loss_{nombre_seguro}.png"

    )


# ==============================================================================
# 49. GRÁFICOS INDIVIDUALES REAL VS MODELO
# ==============================================================================

for modelo in pred_df.columns:

    if modelo == "y_real":

        continue


    fig = plt.figure(
        figsize=(14, 6)
    )


    tmp = pred_df[
        [
            "y_real",
            modelo
        ]
    ].dropna()


    plt.plot(

        tmp.index,

        tmp["y_real"],

        label="Real",

        linewidth=1.2

    )


    plt.plot(

        tmp.index,

        tmp[modelo],

        label=modelo,

        linewidth=0.9

    )


    plt.title(

        f"Real vs predicción — {modelo}"

    )


    plt.xlabel(
        "Fecha / hora"
    )


    plt.ylabel(
        "Nivel"
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    nombre_seguro = (
        modelo
        .replace(
            " ",
            "_"
        )
    )


    guardar_y_mostrar(

        fig,

        f"REAL_vs_{nombre_seguro}.png"

    )


# ==============================================================================
# 50. TABLAS VISUALES EN PNG
# ==============================================================================

def guardar_tabla_png(
    df,
    titulo,
    nombre_archivo,
    max_rows=30
):

    data = df.head(
        max_rows
    ).copy()


    # Redondear números

    for col in data.columns:

        if pd.api.types.is_numeric_dtype(
            data[col]
        ):

            data[col] = data[col].round(3)


    fig_height = max(
        4,
        0.45 * len(data) + 1.5
    )


    fig, ax = plt.subplots(

        figsize=(
            16,
            fig_height
        )

    )


    ax.axis("off")


    tabla = ax.table(

        cellText=data.values,

        colLabels=data.columns,

        loc="center",

        cellLoc="center"

    )


    tabla.auto_set_font_size(
        False
    )


    tabla.set_fontsize(
        8
    )


    tabla.scale(
        1,
        1.5
    )


    ax.set_title(

        titulo,

        fontsize=14,

        pad=20

    )


    guardar_y_mostrar(

        fig,

        nombre_archivo

    )


# ==============================================================================
# 51. EXPORTACIÓN VISUAL DE TABLAS
# ==============================================================================

guardar_tabla_png(

    tabla_1,

    "Tabla 1 — Particiones del dataset",

    "TABLA_01_particiones.png"

)


guardar_tabla_png(

    tabla_2,

    "Tabla 2 — Estadísticos del nivel",

    "TABLA_02_estadisticos.png"

)


guardar_tabla_png(

    tabla_3,

    "Tabla 3 — Variables utilizadas",

    "TABLA_03_variables.png"

)


guardar_tabla_png(

    tabla_4,

    "Tabla 4 — Hiperparámetros de modelos de árboles",

    "TABLA_04_hiperparametros_arboles.png"

)


guardar_tabla_png(

    tabla_5,

    "Tabla 5 — Hiperparámetros RNN/LSTM",

    "TABLA_05_hiperparametros_recurrentes.png"

)


guardar_tabla_png(

    tabla_6,

    "Tabla 6 — Desviaciones respecto de las directrices",

    "TABLA_06_desviaciones.png"

)


guardar_tabla_png(

    tabla_7,

    "Tabla 7 — Iteración preliminar",

    "TABLA_07_iteracion_preliminar.png"

)


guardar_tabla_png(

    resultados_df,

    "Tabla 8 — Métricas por modelo y partición",

    "TABLA_08_metricas.png",

    max_rows=30

)


guardar_tabla_png(

    tabla_9,

    "Tabla 9 — Ranking del conjunto de prueba",

    "TABLA_09_ranking.png"

)


guardar_tabla_png(

    tabla_10,

    "Tabla 10 — Sesgo y desempeño en niveles altos",

    "TABLA_10_niveles_altos.png"

)


guardar_tabla_png(

    tabla_11,

    "Tabla 11 — Importancia de variables",

    "TABLA_11_importancia.png"

)


guardar_tabla_png(

    tabla_12,

    "Tabla 12 — Archivos generados",

    "TABLA_12_archivos.png"

)


# ==============================================================================
# 52. RESUMEN FINAL
# ==============================================================================

log("=" * 80)

log("ENTRENAMIENTO FINALIZADO")

log("=" * 80)


print(
    "\n"
    "============================================================\n"
    " RESULTADOS FINALES — MACHÁNGARA LLANTERA +3 HORAS\n"
    "============================================================"
)


# Mostrar métricas de prueba

print(
    "\nMÉTRICAS DE PRUEBA:"
)


print(

    resultados_df[
        resultados_df["particion"]
        == "test"
    ]
    .sort_values(
        "RMSE"
    )
    .to_string(
        index=False
    )

)


print(
    "\n"
    "============================================================"
)


print(
    "DIRECTORIO DE SALIDA:"
)


print(
    OUT_DIR
)


print(
    "\n"
    "Archivos principales generados:"
)


print(
    "1. resultados_metricas_completo.csv"
)


print(
    "2. predicciones_test_completo.csv"
)


print(
    "3. historiales_entrenamiento.json"
)


print(
    "4. Informe_Tablas_Machangara.xlsx"
)


print(
    "5. tabla_01_particiones.csv"
)


print(
    "6. tabla_02_estadisticos_nivel.csv"
)


print(
    "7. tabla_03_variables.csv"
)


print(
    "8. tabla_04_hiperparametros_arboles.csv"
)


print(
    "9. tabla_05_hiperparametros_recurrentes.csv"
)


print(
    "10. tabla_06_desviaciones.csv"
)


print(
    "11. tabla_07_iteracion_preliminar.csv"
)


print(
    "12. tabla_08_metricas_completas.csv"
)


print(
    "13. tabla_09_ranking_prueba.csv"
)


print(
    "14. tabla_10_sesgo_niveles_altos.csv"
)


print(
    "15. tabla_11_importancia_variables.csv"
)


print(
    "16. tabla_12_archivos_generados.csv"
)


print(
    "\n"
    "Gráficos:"
)


print(
    "01_particion_cronologica.png"
)


print(
    "02_distribucion_particiones.png"
)


print(
    "03_curvas_entrenamiento_validacion.png"
)


print(
    "04_MAE_modelos_particiones.png"
)


print(
    "05_R2_modelos_test.png"
)


print(
    "06_REAL_vs_PREDICCIONES.png"
)


print(
    "07_EVENTO_MAXIMO.png"
)


print(
    "08_DISPERSION_REAL_VS_PREDICHO.png"
)


print(
    "09_IMPORTANCIA_RANDOM_FOREST.png"
)


print(
    "09b_IMPORTANCIA_XGBOOST.png"
)


print(
    "============================================================"
)


log("PROCESO COMPLETADO")