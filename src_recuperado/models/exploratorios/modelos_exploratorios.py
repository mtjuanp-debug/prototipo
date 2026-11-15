# -*- coding: utf-8 -*-
"""Modelos exploratorios del Yanuncay (anexo; no compiten en la hipótesis oficial).

* LightGBM y CatBoost (opcional): envoltorios con ``sample_weight`` y early stopping en el tramo de control.
* GRU: igual que el ``RegresorRecurrente`` oficial con celda GRU (cabeza densa 32 + ReLU).
* TCN (Bai et al., 2018): convoluciones causales dilatadas en bloques residuales; se usa la salida del último paso.
* N-HiTS (Challu et al., 2023): pilas de MLP con *pooling* multi-tasa e interpolación jerárquica, adaptado a la
  estrategia directa (una salida, el horizonte τ) con variables exógenas en la ventana.

Todas las redes reciben la ventana (lote, L = 24, variables) y devuelven (lote,). Las redes se entrenan sobre el
objetivo estandarizado con media y desviación del ajuste (guardadas como *buffers* en el modelo).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch  # se importa antes que numpy (Windows: evita el error de DLL c10.dll)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import numpy as np

PACIENCIA_LR = 3
FAMILIAS_ARBOL = ("LGBM", "CAT")
FAMILIAS_RED = ("GRU", "TCN", "NHITS")


# ---------------------------------------------------------------------------------------------------------------
# Árboles
# ---------------------------------------------------------------------------------------------------------------
def _pesos_normalizados(w: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if w is None else (w / np.mean(w)).astype(np.float32)


@dataclass
class ResultadoArbol:
    modelo: object
    mejor_iteracion: int
    curva_train: list
    curva_control: list

    def predecir(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.modelo.predict(X), dtype=np.float32)


def ajustar_lgbm(cfg: dict, comun: dict, X_fit: np.ndarray, y_fit: np.ndarray, w_fit: Optional[np.ndarray],
                 X_es: np.ndarray, y_es: np.ndarray, X_curva: Optional[np.ndarray] = None,
                 y_curva: Optional[np.ndarray] = None) -> ResultadoArbol:
    """LightGBM con early stopping en (X_es, y_es). El conjunto de curva se nombra 'training' para que
    LightGBM no lo use en el early stopping."""
    import lightgbm as lgb
    comun = dict(comun)
    rondas_es = comun.pop("early_stopping_rounds")
    modelo = lgb.LGBMRegressor(**comun, **cfg)
    eval_set, nombres = [(X_es, y_es)], ["control"]
    if X_curva is not None:
        eval_set, nombres = [(X_curva, y_curva)] + eval_set, ["training"] + nombres
    modelo.fit(X_fit, y_fit, sample_weight=_pesos_normalizados(w_fit), eval_set=eval_set, eval_names=nombres,
               eval_metric="rmse", callbacks=[lgb.early_stopping(rondas_es, verbose=False)])
    ev = modelo.evals_result_
    return ResultadoArbol(modelo, int(modelo.best_iteration_ or modelo.n_estimators),
                          list(ev.get("training", {}).get("rmse", [])), list(ev["control"]["rmse"]))


def ajustar_catboost(cfg: dict, comun: dict, X_fit: np.ndarray, y_fit: np.ndarray, w_fit: Optional[np.ndarray],
                     X_es: np.ndarray, y_es: np.ndarray, X_curva=None, y_curva=None) -> ResultadoArbol:
    from catboost import CatBoostRegressor
    comun = dict(comun)
    rondas_es = comun.pop("early_stopping_rounds")
    modelo = CatBoostRegressor(**comun, **cfg, od_type="Iter", od_wait=rondas_es, use_best_model=True,
                               allow_writing_files=False, verbose=False)
    modelo.fit(X_fit, y_fit, sample_weight=_pesos_normalizados(w_fit), eval_set=(X_es, y_es))
    ev = modelo.get_evals_result()
    return ResultadoArbol(modelo, int(modelo.get_best_iteration()) + 1,
                          list(ev.get("learn", {}).get("RMSE", [])), list(ev["validation"]["RMSE"]))


AJUSTADORES_ARBOL: dict[str, Callable[..., ResultadoArbol]] = {"LGBM": ajustar_lgbm, "CAT": ajustar_catboost}


# ---------------------------------------------------------------------------------------------------------------
# Redes
# ---------------------------------------------------------------------------------------------------------------
def _cabeza(entrada: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(entrada, 32), nn.ReLU(), nn.Linear(32, 1))


class RegresorGRU(nn.Module):
    def __init__(self, n_variables: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(n_variables, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.cabeza = _cabeza(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        salida, _ = self.rnn(x)
        return self.cabeza(salida[:, -1, :]).squeeze(1)


class BloqueTCN(nn.Module):
    """Dos convoluciones causales dilatadas (relleno solo a la izquierda) con conexión residual."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilatacion: int, dropout: float) -> None:
        super().__init__()
        self.relleno = (kernel - 1) * dilatacion
        self.conv1 = weight_norm(nn.Conv1d(c_in, c_out, kernel, dilation=dilatacion))
        self.conv2 = weight_norm(nn.Conv1d(c_out, c_out, kernel, dilation=dilatacion))
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (lote, canales, tiempo)
        h = self.drop(F.relu(self.conv1(F.pad(x, (self.relleno, 0)))))
        h = self.drop(F.relu(self.conv2(F.pad(h, (self.relleno, 0)))))
        return F.relu(h + self.residual(x))


class RegresorTCN(nn.Module):
    def __init__(self, n_variables: int, canales: int, n_niveles: int, kernel: int, dropout: float) -> None:
        super().__init__()
        bloques, c_in = [], n_variables
        for i in range(n_niveles):
            bloques.append(BloqueTCN(c_in, canales, kernel, 2 ** i, dropout))
            c_in = canales
        self.red = nn.Sequential(*bloques)
        self.cabeza = _cabeza(canales)

    @staticmethod
    def campo_receptivo(n_niveles: int, kernel: int) -> int:
        return 1 + 2 * (kernel - 1) * (2 ** n_niveles - 1)

    def secuencia(self, x: torch.Tensor) -> torch.Tensor:
        """Salida de la red convolucional en todos los pasos (lote, canales, tiempo)."""
        return self.red(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.secuencia(x)[:, :, -1]).squeeze(1)


def interpolar(coef: torch.Tensor, n: int) -> torch.Tensor:
    """Interpolación lineal de (lote, k) coeficientes a (lote, n) puntos."""
    if coef.shape[1] == n:
        return coef
    if coef.shape[1] == 1:
        return coef.expand(-1, n)
    return F.interpolate(coef.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)


class BloqueNHiTS(nn.Module):
    """Pooling (máximo) de tasa k sobre la ventana -> MLP -> coeficientes de backcast (L/k) y de pronóstico
    (n_coef), que se interpolan a L y a τ pasos (interpolación jerárquica)."""

    def __init__(self, lookback: int, n_exog: int, tau: int, k_pool: int, n_coef: int, ancho: int,
                 n_capas: int, dropout: float) -> None:
        super().__init__()
        self.lookback, self.tau = lookback, tau
        self.pool = nn.MaxPool1d(k_pool, stride=k_pool, ceil_mode=True)
        self.n_back = math.ceil(lookback / k_pool)
        capas, d = [], self.n_back * (1 + n_exog)
        for _ in range(n_capas):
            capas += [nn.Linear(d, ancho), nn.ReLU(), nn.Dropout(dropout)]
            d = ancho
        self.mlp = nn.Sequential(*capas)
        self.theta = nn.Linear(d, self.n_back + n_coef)

    def forward(self, y_res: torch.Tensor, exog: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zy = self.pool(y_res.unsqueeze(1)).flatten(1)
        zx = self.pool(exog).flatten(1) if exog.shape[1] else zy[:, :0]
        theta = self.theta(self.mlp(torch.cat([zy, zx], dim=1)))
        backcast = interpolar(theta[:, :self.n_back], self.lookback)
        pronostico = interpolar(theta[:, self.n_back:], self.tau)
        return backcast, pronostico


class RegresorNHiTS(nn.Module):
    """N-HiTS directo de una salida. La serie residual es el canal del nivel objetivo; el resto de variables
    de la ventana son exógenas y entran (agrupadas con la misma tasa) en todos los bloques. Cada pila s tiene
    tasa de pooling k_s y n_coef_s coeficientes de pronóstico sobre la trayectoria t+1..t+τ; la predicción es
    el último punto de la suma de pronósticos. Con supervisión de un solo horizonte la interpolación actúa
    como parametrización estructural (baja frecuencia en las pilas gruesas), no como pronóstico multi-paso."""

    def __init__(self, n_variables: int, idx_objetivo: int, lookback: int, tau: int, pooling: list,
                 n_bloques: int, ancho: int, n_capas: int, dropout: float) -> None:
        super().__init__()
        self.idx = idx_objetivo
        coef = [1, max(1, math.ceil(tau / 2)), tau][-len(pooling):] if len(pooling) <= 3 else [tau] * len(pooling)
        self.bloques = nn.ModuleList(
            BloqueNHiTS(lookback, n_variables - 1, tau, k, c, ancho, n_capas, dropout)
            for k, c in zip(pooling, coef) for _ in range(n_bloques))
        exog = [i for i in range(n_variables) if i != idx_objetivo]
        self.register_buffer("idx_exog", torch.tensor(exog, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_res = x[:, :, self.idx]
        exog = x.index_select(2, self.idx_exog).transpose(1, 2)
        total = None
        for bloque in self.bloques:
            backcast, pronostico = bloque(y_res, exog)
            y_res = y_res - backcast
            total = pronostico if total is None else total + pronostico
        return total[:, -1]


class RedEscalada(nn.Module):
    """Envuelve una red: el núcleo predice el objetivo estandarizado; forward devuelve la escala original."""

    def __init__(self, nucleo: nn.Module, media: float, desv: float) -> None:
        super().__init__()
        self.nucleo = nucleo
        self.register_buffer("media", torch.tensor(float(media)))
        self.register_buffer("desv", torch.tensor(float(desv)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nucleo(x) * self.desv + self.media


def construir_red(familia: str, cfg: dict, n_variables: int, idx_objetivo: int, lookback: int,
                  tau: int) -> nn.Module:
    if familia == "GRU":
        return RegresorGRU(n_variables, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
    if familia == "TCN":
        return RegresorTCN(n_variables, cfg["canales"], cfg["n_niveles"], cfg["kernel"], cfg["dropout"])
    if familia == "NHITS":
        return RegresorNHiTS(n_variables, idx_objetivo, lookback, tau, list(cfg["pooling"]), cfg["n_bloques"],
                             cfg["ancho"], cfg["n_capas"], cfg["dropout"])
    raise ValueError(f"Familia de red desconocida: {familia}")


# ---------------------------------------------------------------------------------------------------------------
# Entrenamiento de redes (ventanas indexadas sobre el tensor de la malla, como el notebook oficial)
# ---------------------------------------------------------------------------------------------------------------
def a_tensor(arr: np.ndarray, dispositivo: torch.device, dtype=torch.float32) -> torch.Tensor:
    arr = np.ascontiguousarray(arr)
    try:
        return torch.from_numpy(arr).to(device=dispositivo, dtype=dtype)
    except RuntimeError:      # PyTorch compilado con NumPy 1.x
        return torch.tensor(arr.tolist(), dtype=dtype, device=dispositivo)


def a_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t.tolist(), dtype=np.float32)


@dataclass
class DatosRed:
    """Tensores de la malla en el dispositivo: variables por paso (n, v), objetivo (n,) y pesos (n,)."""
    x_seq: torch.Tensor
    y: torch.Tensor
    pesos: Optional[torch.Tensor]
    lookback: int

    def __post_init__(self) -> None:
        self.offsets = torch.arange(-self.lookback + 1, 1, device=self.x_seq.device)

    @property
    def dispositivo(self) -> torch.device:
        return self.x_seq.device

    def ventanas(self, filas: torch.Tensor) -> torch.Tensor:
        return self.x_seq[filas[:, None] + self.offsets]


def lotes(datos: DatosRed, pos: torch.Tensor, batch_size: int, barajar: bool,
          generador: Optional[torch.Generator] = None):
    n = len(pos)
    orden = (torch.randperm(n, generator=generador).to(datos.dispositivo) if barajar
             else torch.arange(n, device=datos.dispositivo))
    for ini in range(0, n, batch_size):
        filas = pos[orden[ini:ini + batch_size]]
        w = datos.pesos[filas] if datos.pesos is not None else None
        yield datos.ventanas(filas), datos.y[filas], w


@torch.no_grad()
def predecir_red(modelo: nn.Module, datos: DatosRed, pos: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    modelo.eval()
    if len(pos) == 0:
        return np.zeros(0, dtype=np.float32)
    pos_t = a_tensor(pos, datos.dispositivo, torch.long)
    return np.concatenate([a_numpy(modelo(X)) for X, _, _ in lotes(datos, pos_t, batch_size, False)])


def entrenar_red(familia: str, cfg: dict, datos: DatosRed, pos_fit: np.ndarray, pos_ctrl: np.ndarray,
                 idx_objetivo: int, tau: int, semilla: int = 42, rango_cm: float = 1.0,
                 etiqueta: str = "", verboso: bool = False) -> tuple[RedEscalada, dict, int]:
    """MSE (ponderado si hay pesos) sobre el objetivo estandarizado; early stopping con el MSE sin ponderar del
    tramo de control; recorte de gradiente 1.0 y ReduceLROnPlateau(0.5, 3), como el notebook oficial."""
    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)
    y_fit = a_numpy(datos.y[a_tensor(pos_fit, datos.dispositivo, torch.long)])
    media, desv = float(np.mean(y_fit)), float(np.std(y_fit)) or 1.0
    nucleo = construir_red(familia, cfg, datos.x_seq.shape[1], idx_objetivo, datos.lookback, tau)
    modelo = RedEscalada(nucleo, media, desv).to(datos.dispositivo)
    clase_opt = torch.optim.AdamW if cfg["optimizador"] == "adamw" else torch.optim.Adam
    opt = clase_opt(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=PACIENCIA_LR)
    pos_fit_t = a_tensor(pos_fit, datos.dispositivo, torch.long)
    y_ctrl = a_numpy(datos.y[a_tensor(pos_ctrl, datos.dispositivo, torch.long)])
    escala_w = 1.0
    if datos.pesos is not None:
        escala_w = 1.0 / float(datos.pesos[pos_fit_t].mean())      # pesos con media 1 en el ajuste
    hist = {"train": [], "val": [], "lr": [], "s_epoca": []}
    mejor, mejor_estado, espera, mejor_epoca = np.inf, None, 0, 0
    for epoca in range(1, cfg["epocas"] + 1):
        t0 = time.time()
        modelo.train()
        suma, n = 0.0, 0
        for X, y, w in lotes(datos, pos_fit_t, cfg["batch_size"], True, generador):
            opt.zero_grad()
            err2 = (modelo.nucleo(X) - (y - modelo.media) / modelo.desv) ** 2
            perdida = (err2 * w * escala_w).mean() if w is not None else err2.mean()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            suma += float(err2.detach().sum()) * desv ** 2
            n += len(y)
        mse_ctrl = float(np.mean((predecir_red(modelo, datos, pos_ctrl) - y_ctrl) ** 2))
        if not np.isfinite(mse_ctrl):
            raise FloatingPointError(f"{etiqueta}: pérdida no finita en la época {epoca}")
        planificador.step(mse_ctrl)
        hist["train"].append(float(np.sqrt(suma / n) * rango_cm))
        hist["val"].append(float(np.sqrt(mse_ctrl) * rango_cm))
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["s_epoca"].append(time.time() - t0)
        if verboso:
            print(f"    {etiqueta} época {epoca:3d} | RMSE ajuste {hist['train'][-1]:.2f} cm | control "
                  f"{hist['val'][-1]:.2f} cm | lr {hist['lr'][-1]:.1e} | {hist['s_epoca'][-1]:.1f} s")
        if mse_ctrl < mejor:
            mejor, espera, mejor_epoca = mse_ctrl, 0, epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
            if espera >= cfg["patience"]:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, hist, mejor_epoca
