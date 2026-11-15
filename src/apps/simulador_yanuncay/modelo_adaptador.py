"""Capa de adaptación entre el simulador y los modelos entrenados.

ÚNICO archivo que debe cambiar cuando lleguen los modelos definitivos. El resto del simulador solo
usa estas cuatro funciones públicas:

- ``cargar_contrato(config) -> ContratoModelo``: describe entradas, escalado y tipo de objetivo.
- ``cargar_modelo(rio, familia, h, config) -> ModeloCargado``.
- ``calcular_variables(ventana, contrato) -> DataFrame`` (unidades físicas, sin escalar).
- ``predecir(modelo, variables, nivel_actual_cm) -> float`` (NIVEL en cm en t+h).

Contrato JSON de los modelos definitivos (``rutas.contrato_modelos`` en la configuración; hoy
``Docs/resources/modelos_yanuncay/modelos/formulacion_yanuncay.json``). Claves reconocidas (las ausentes
toman el valor de ``rutas.configuracion_modelos``):
    target | objetivo            columna del nivel objetivo (``objetivo`` solo si no es "nivel"/"delta")
    tipo_objetivo | objetivo     "nivel" (y(t+τ)) o "delta" (y(t+τ) − y(t)); la formulación final usa
                                 ``"objetivo": "delta"`` y ``"target": <columna>``
    horizontes | horizontes_h    [4, 7]
    features_tabulares           orden exacto de columnas de los árboles
    features_secuencia           columnas por paso de tiempo de RNN/LSTM
    lookback_h                   longitud de la ventana de las redes
    escaladores                  ruta a CSV (columna,min,max) o {col: {"min":..,"max":..}}
    escalar_entradas             true/false (por defecto true)
    escala_objetivo | escala_objetivo_cm
                                 {"4": {"min":..,"max":..}, ...} o {"min":..,"max":..}; null = sin escala.
                                 Con "delta" se usa Δ = v·(max − min) (convención ``desescalar_cambio``)
    entradas, config_variables   parámetros de ``variables_yanuncay.ConfigVariables``; sin ``entradas`` se
                                 toman de ``horizontes_json`` y ``config_variables`` las sobrescribe
    modelos_dir                  carpeta de los modelos (por defecto, la carpeta del propio contrato)
    archivos                     {"XGB": "XGB_h{h}.json", "RF": "RF_h{h}.joblib", ...}
    modelos                      {"RF|h4": {"archivo": .., "archivo_torchscript": ..}, ...} (formulación
                                 final): de aquí se derivan ``archivos``; las redes usan el TorchScript
    redes                        {"LSTM": {"hidden_size":..,"num_layers":..}, ...} (opcional; si no,
                                 se infiere del state_dict)

Con un contrato definitivo las variables se calculan con ``variables_yanuncay.calcular_variables``
(importado por nombre desde ``src/data_prep``); con la corrida preliminar se usa la réplica causal
``variables_preliminar`` (catálogo preliminar).
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from configuracion import RAIZ_REPO, ruta
from variables_preliminar import calcular_variables_preliminar

ARCHIVOS_PRELIMINARES: dict[str, str] = {
    "XGB": "XGB_h{h}.json", "RF": "RF_h{h}.joblib", "ADA": "ADA_h{h}.joblib",
    "RNN": "RNN_h{h}.pt", "LSTM": "LSTM_h{h}.pt",
}
FAMILIAS_SECUENCIA = ("RNN", "LSTM")


# --------------------------------------------------------------------------------------------
# Contrato
# --------------------------------------------------------------------------------------------
@dataclass
class ContratoModelo:
    objetivo: str
    tipo_objetivo: str
    horizontes: list[int]
    features_tabulares: list[str]
    features_secuencia: list[str]
    lookback_h: int
    escaladores: dict[str, tuple[float, float]]
    escala_objetivo: dict[int, tuple[float, float] | None]
    modelos_dir: Path
    archivos: dict[str, str]
    escalar_entradas: bool = True
    redes: dict[str, dict[str, Any]] = field(default_factory=dict)
    origen: str = "preliminar"
    crudo: dict[str, Any] = field(default_factory=dict)

    @property
    def ventana_minima_h(self) -> int:
        """Horas de historia necesarias: 72 h de acumulados + ventana de las redes."""
        return 72 + max(self.lookback_h, 1)


def _leer_escaladores(fuente: Any) -> dict[str, tuple[float, float]]:
    if isinstance(fuente, dict):
        return {k: (float(v["min"]), float(v["max"])) for k, v in fuente.items()}
    p = Path(fuente)
    p = p if p.is_absolute() else RAIZ_REPO / p
    df = pd.read_csv(p)
    return {r.columna: (float(r.min), float(r.max)) for r in df.itertuples()}


def _escala_objetivo(valor: Any, horizontes: list[int],
                     escaladores: dict[str, tuple[float, float]], objetivo: str) -> dict[int, tuple[float, float] | None]:
    if valor is None and objetivo in escaladores:
        return {h: escaladores[objetivo] for h in horizontes}
    if valor is None or valor == "ninguna":
        return {h: None for h in horizontes}
    if isinstance(valor, dict) and "min" in valor:
        return {h: (float(valor["min"]), float(valor["max"])) for h in horizontes}
    return {int(h): (None if v is None else (float(v["min"]), float(v["max"]))) for h, v in valor.items()}


TIPOS_OBJETIVO = ("nivel", "delta")


def _archivos_desde_modelos(modelos: dict[str, Any]) -> dict[str, str]:
    """``{"RF|h4": {"archivo": "RF_h4.joblib", ...}}`` -> ``{"RF": "RF_h{h}.joblib"}`` (TorchScript si existe)."""
    patrones: dict[str, str] = {}
    for clave, m in modelos.items():
        familia, _, h_txt = clave.partition("|")
        h = int(str(h_txt).lstrip("h") or m.get("horizonte"))
        archivo = str(m.get("archivo_torchscript") or m["archivo"])
        patron = archivo.replace(f"_h{h}", "_h{h}", 1)
        if patrones.setdefault(familia, patron) != patron:
            raise ValueError(f"Nombres de archivo inconsistentes entre horizontes para {familia}: {patron}")
    return patrones


def _datos_contrato(config: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    p_contrato = ruta(config, "contrato_modelos")
    if p_contrato is None:
        return {}, None
    if not p_contrato.exists():
        raise FileNotFoundError(f"No existe el contrato de modelos: {p_contrato}")
    return json.loads(p_contrato.read_text(encoding="utf-8")), p_contrato


def cargar_contrato(config: dict[str, Any]) -> ContratoModelo:
    """Lee el contrato de los modelos definitivos o, si no existe, el de la corrida preliminar."""
    pre = json.loads(ruta(config, "configuracion_modelos").read_text(encoding="utf-8"))
    escaladores_pre = _leer_escaladores(ruta(config, "escaladores"))
    datos, p_contrato = _datos_contrato(config)
    origen = str(p_contrato) if p_contrato is not None else "preliminar"
    valor_obj = datos.get("objetivo")
    es_tipo = isinstance(valor_obj, str) and valor_obj in TIPOS_OBJETIVO
    objetivo = datos.get("target") or (None if es_tipo else valor_obj) or pre["target"]
    tipo = datos.get("tipo_objetivo") or (valor_obj if es_tipo else config.get("tipo_objetivo", "nivel"))
    if tipo not in TIPOS_OBJETIVO:
        raise ValueError(f"tipo de objetivo no reconocido: {tipo}")
    horizontes = [int(h) for h in datos.get("horizontes", datos.get("horizontes_h", pre["horizontes"]))]
    escaladores = _leer_escaladores(datos["escaladores"]) if "escaladores" in datos else escaladores_pre
    if "modelos_dir" in datos:
        modelos_dir = Path(datos["modelos_dir"])
    elif p_contrato is not None:
        modelos_dir = p_contrato.parent
    else:
        modelos_dir = ruta(config, "modelos_dir")
    if not modelos_dir.is_absolute():
        modelos_dir = RAIZ_REPO / modelos_dir
    escala_cruda = datos.get("escala_objetivo", datos.get("escala_objetivo_cm", "__ausente__"))
    archivos = {**ARCHIVOS_PRELIMINARES}
    if isinstance(datos.get("modelos"), dict):
        archivos = _archivos_desde_modelos(datos["modelos"])
    archivos.update(datos.get("archivos", {}))
    return ContratoModelo(
        objetivo=objetivo,
        tipo_objetivo=str(tipo),
        horizontes=horizontes,
        features_tabulares=list(datos.get("features_tabulares", pre["features_tabulares"])),
        features_secuencia=list(datos.get("features_secuencia", pre["features_secuencia"])),
        lookback_h=int(datos.get("lookback_h", pre.get("lookback_h", 24))),
        escaladores=escaladores,
        escala_objetivo=(_escala_objetivo(escala_cruda, horizontes, escaladores, objetivo)
                         if escala_cruda != "__ausente__" else {h: escaladores.get(objetivo) for h in horizontes}),
        modelos_dir=modelos_dir,
        archivos=archivos,
        escalar_entradas=bool(datos.get("escalar_entradas", True)),
        redes=dict(datos.get("redes", {})),
        origen=origen,
        crudo=datos,
    )


# --------------------------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------------------------
def _cargar_calculo_externo() -> Callable[..., pd.DataFrame] | None:
    carpeta = RAIZ_REPO / "src" / "data_prep"
    if not (carpeta / "variables_yanuncay.py").exists():
        return None
    if str(carpeta) not in sys.path:
        sys.path.insert(0, str(carpeta))  # también resuelve un posible ``variables_rio`` reexportado
    try:
        modulo = importlib.import_module("variables_yanuncay")
        return getattr(modulo, "calcular_variables", None)
    except Exception as exc:  # noqa: BLE001 - el respaldo es intencional
        print(f"[modelo_adaptador] variables_yanuncay no utilizable ({exc}); se usa la réplica preliminar.")
        return None


_CALCULO_EXTERNO = _cargar_calculo_externo()
_CONFIG_EXTERNA: dict[str, Any] = {}


def _usa_externo(contrato: ContratoModelo | None) -> bool:
    # Los modelos preliminares se entrenaron con el catálogo preliminar: se usa siempre la réplica.
    return _CALCULO_EXTERNO is not None and contrato is not None and contrato.origen != "preliminar"


def origen_variables(contrato: ContratoModelo | None = None) -> str:
    return "src/data_prep/variables_yanuncay.py" if _usa_externo(contrato) else "réplica causal preliminar"


def _config_variables_externa(contrato: ContratoModelo) -> Any:
    """``ConfigVariables`` del módulo externo a partir del contrato (``entradas`` + ``config_variables``
    o ``horizontes_json``)."""
    if contrato.origen in _CONFIG_EXTERNA:
        return _CONFIG_EXTERNA[contrato.origen]
    vy = sys.modules["variables_yanuncay"]
    crudo = contrato.crudo
    extra = {k: tuple(v) if isinstance(v, list) else v for k, v in (crudo.get("config_variables") or {}).items()}
    if "entradas" in crudo:
        cfg = vy.ConfigVariables(entradas=tuple(crudo["entradas"]), **extra)
    else:
        p = Path(crudo.get("horizontes_json", "Docs/resources/horizontes_yanuncay.json"))
        cfg = vy.ConfigVariables.desde_horizontes_json(p if p.is_absolute() else RAIZ_REPO / p, **extra)
    _CONFIG_EXTERNA[contrato.origen] = cfg
    return cfg


def calcular_variables(ventana: pd.DataFrame, contrato: ContratoModelo | None = None) -> pd.DataFrame:
    """Variables en unidades físicas para cada hora de ``ventana`` (la última fila es el instante t)."""
    if _usa_externo(contrato):
        return _CALCULO_EXTERNO(ventana, _config_variables_externa(contrato))
    return calcular_variables_preliminar(ventana)


# --------------------------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------------------------
@dataclass
class ModeloCargado:
    familia: str
    h: int
    objeto: Any
    es_secuencia: bool
    contrato: ContratoModelo


def _red_desde_state_dict(familia: str, archivo: Path, contrato: ContratoModelo) -> Any:
    import torch
    from torch import nn

    estado = torch.load(archivo, map_location="cpu")
    n_capas = sum(1 for k in estado if k.startswith("rnn.weight_ih_l"))
    n_var = estado["rnn.weight_ih_l0"].shape[1]
    puertas = 4 if familia == "LSTM" else 1
    oculto = estado["rnn.weight_hh_l0"].shape[1]
    assert estado["rnn.weight_hh_l0"].shape[0] == puertas * oculto
    params = {"hidden_size": oculto, "num_layers": n_capas, **contrato.redes.get(familia, {})}

    class RegresorRecurrente(nn.Module):  # misma arquitectura que el entrenamiento
        def __init__(self) -> None:
            super().__init__()
            clase = nn.LSTM if familia == "LSTM" else nn.RNN
            self.rnn = clase(input_size=n_var, hidden_size=params["hidden_size"],
                             num_layers=params["num_layers"], batch_first=True)
            self.cabeza = nn.Sequential(nn.Linear(params["hidden_size"], 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            salida, _ = self.rnn(x)
            return self.cabeza(salida[:, -1, :]).squeeze(1)

    red = RegresorRecurrente()
    red.load_state_dict(estado)
    red.eval()
    return red


_CACHE: dict[tuple[str, str, int, str], ModeloCargado] = {}


def config_variables(contrato: ContratoModelo) -> Any:
    """``ConfigVariables`` usada con un contrato definitivo (None con la corrida preliminar)."""
    return _config_variables_externa(contrato) if _usa_externo(contrato) else None


def familias_disponibles(contrato: ContratoModelo) -> list[str]:
    """Familias con archivo para todos los horizontes (+ Persistencia, que no se entrena)."""
    salida = [f for f, patron in contrato.archivos.items()
              if all((contrato.modelos_dir / patron.format(h=h)).exists() for h in contrato.horizontes)]
    return salida + ["Persistencia"]


def cargar_modelo(rio: str, familia: str, h: int, config: dict[str, Any]) -> ModeloCargado:
    """Carga (con caché) el modelo de ``familia`` para el horizonte ``h``."""
    contrato = cargar_contrato(config)
    clave = (rio, familia, int(h), str(contrato.modelos_dir))
    if clave in _CACHE:
        return _CACHE[clave]
    if familia == "Persistencia":
        modelo = ModeloCargado(familia, h, None, False, contrato)
    else:
        archivo = contrato.modelos_dir / contrato.archivos[familia].format(h=h)
        if not archivo.exists():
            raise FileNotFoundError(f"No existe el modelo {familia} h={h}: {archivo}")
        if familia in FAMILIAS_SECUENCIA and archivo.stem.endswith("_ts"):
            import torch
            objeto = torch.jit.load(str(archivo), map_location="cpu")
            objeto.eval()
        elif familia in FAMILIAS_SECUENCIA:
            objeto = _red_desde_state_dict(familia, archivo, contrato)
        elif archivo.suffix == ".json":
            import xgboost as xgb
            objeto = xgb.XGBRegressor()
            objeto.load_model(archivo)
            objeto.set_params(device="cpu")
        else:
            import joblib
            objeto = joblib.load(archivo)
        modelo = ModeloCargado(familia, h, objeto, familia in FAMILIAS_SECUENCIA, contrato)
    _CACHE[clave] = modelo
    return modelo


def _escalar(df: pd.DataFrame, contrato: ContratoModelo) -> pd.DataFrame:
    if not contrato.escalar_entradas:
        return df
    salida = df.copy()
    for col in salida.columns:
        if col in contrato.escaladores:
            mn, mx = contrato.escaladores[col]
            salida[col] = (salida[col] - mn) / (mx - mn) if mx > mn else 0.0
    return salida


def _a_unidades(valor: float, modelo: ModeloCargado, nivel_actual_cm: float) -> float:
    """Salida del modelo -> nivel en cm.

    - "nivel": ŷ = v·(max − min) + min.
    - "delta": Δ = v·(max − min) (el mínimo se cancela, convención de ``desescalar_cambio``) y ŷ = y(t) + Δ.
    """
    escala = modelo.contrato.escala_objetivo.get(modelo.h)
    if modelo.contrato.tipo_objetivo == "delta":
        delta = valor * (escala[1] - escala[0]) if escala is not None else valor
        return float(nivel_actual_cm + delta)
    if escala is not None:
        valor = valor * (escala[1] - escala[0]) + escala[0]
    return float(valor)


def predecir(modelo: ModeloCargado, variables: pd.DataFrame, nivel_actual_cm: float) -> float:
    """Nivel pronosticado (cm) en t+h, con t = última fila de ``variables``. NaN si faltan datos."""
    if modelo.familia == "Persistencia":
        return float(nivel_actual_cm)
    c = modelo.contrato
    if modelo.es_secuencia:
        bloque = variables[c.features_secuencia].iloc[-c.lookback_h:]
        if len(bloque) < c.lookback_h or bloque.isna().any().any():
            return math.nan
        import torch
        x = _escalar(bloque, c).to_numpy(np.float32)[None, :, :]
        with torch.no_grad():
            valor = float(modelo.objeto(torch.tensor(x.tolist(), dtype=torch.float32)).item())
    else:
        fila = variables[c.features_tabulares].iloc[[-1]]
        if fila.isna().any().any():
            return math.nan
        valor = float(modelo.objeto.predict(_escalar(fila, c).to_numpy(np.float32))[0])
    return _a_unidades(valor, modelo, nivel_actual_cm)
