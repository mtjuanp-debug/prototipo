"""Prototipo Alerta Temprana Ríos de Cuenca (prueba de concepto, objetivo específico 4).

Tablero consolidado con una pestaña por río: Yanuncay (τ = 4 y 7 h), Tomebamba (3 y 5 h), Tarqui (5 y 9 h) y
Machángara (3 y 6 h) (``rios`` de ``config_simulador.json``). Arriba, la vista general (``vista_general.py``) muestra
de un vistazo los cuatro ríos: controles globales, tarjetas de estado observado y pronosticado, ríos que requieren
atención, registro consolidado de alertas y curvas de nivel o caudal. Es informativa: la decisión es del operador.
En cada pestaña se simulan los sensores hidrometeorológicos del
río (las estaciones que usan sus modelos) reproduciendo hora a hora el periodo de prueba, se pronostica el nivel
a +τ1 y +τ2 y cada nivel se traduce en un estado de alerta (normal, prealerta, alerta) con los umbrales de
caudal de la estación y la curva de descarga, sin un clasificador independiente.

Ejecución (desde la raíz del repositorio):
    pip install -r src/apps/simulador_yanuncay/requirements.txt
    streamlit run src/apps/simulador_yanuncay/app.py

Cada pestaña (``tablero_rio.py``) tiene su propia simulación y sus propios controles: ⏮ Reiniciar, ▶ Simular /
⏸ Pausar, avance manual (+1 h, +24 h), velocidad, sensores en vivo, semáforos, top de modelos con tacómetros,
banda de confianza y consenso, gráficos y registro de alertas. Su configuración (corrida, top de modelos, modelo
principal, fecha de inicio y escenarios de tormenta, falla de sensor o valor congelado) está en el expander
«⚙ Configuración» de la pestaña. Simular un río no avanza a los demás; pueden simularse varios a la vez.

Top de modelos: los modelos oficiales se ordenan por su desempeño en crecidas en validación (eventos de
prealerta anticipados y falsas alarmas, ver ``confianza.py``); el primero elegido es el principal y emite las
alertas. Para cambiar de modelos solo se modifica ``modelo_adaptador.py`` (o ``rutas.contrato_modelos`` en
``config_simulador.json``).
"""
from __future__ import annotations

import torch  # noqa: F401  (importar antes que xgboost/sklearn evita el error de c10.dll en Windows)

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import vista_general  # noqa: E402
from configuracion import rios_disponibles  # noqa: E402
from tablero_rio import TableroRio  # noqa: E402

TITULO = "Prototipo Alerta Temprana Ríos de Cuenca"
st.set_page_config(page_title=TITULO, page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")


RIOS = rios_disponibles()
TABLEROS = {rio: TableroRio(rio) for rio in RIOS}
for tablero in TABLEROS.values():  # cada río avanza solo si está simulando y ya pasó su intervalo
    tablero.avance_automatico()

st.title(f"🌊 {TITULO}")
st.caption("Herramienta informativa: los estados salen de los umbrales de caudal de cada estación y de los "
           "pronósticos de los modelos; la decisión es del operador.")
vista_general.mostrar(TABLEROS)

st.divider()
st.subheader("Detalle por río")
for pestana, tablero in zip(st.tabs([f"🌊 {nombre}" for nombre in RIOS.values()]), TABLEROS.values()):
    with pestana:
        tablero.mostrar()

with st.sidebar:
    st.header("Leyenda")
    st.markdown("✅ normal · ⚠️ prealerta · 🚨 alerta · ❔ sin dato")
    for tablero in TABLEROS.values():
        c = tablero.config
        st.caption(f"**{c.get('nombre_rio', tablero.rio)}** — τ = {' y '.join(str(h) for h in c['horizontes'])} h · "
                   f"umbrales {c['umbrales_m3s']['prealerta']:.0f}/{c['umbrales_m3s']['alerta']:.0f} m³/s · "
                   f"modelo principal {tablero.s['pron'].familia}")
    st.caption("La configuración de cada río está en el expander **⚙ Configuración** de su pestaña.")

# ------------------------------------------------------------------ bucle no bloqueante
activos = [t for t in TABLEROS.values() if t.reproduciendo and not t.s["sim"].agotado]
if activos:
    time.sleep(min(t.intervalo for t in activos))
    st.rerun()
