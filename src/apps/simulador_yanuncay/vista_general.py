"""Vista general «de un vistazo» de los cuatro ríos (parte superior del tablero de control).

Muestra, sin entrar a ninguna pestaña, el estado observado y pronosticado de cada río, los ríos que requieren
atención, el registro consolidado de alertas y las curvas de nivel o caudal. Es solo informativa: presenta los
estados que salen de los umbrales de caudal y de los pronósticos; la decisión es del operador (no se prescriben
acciones). Lo específico de cada río y sus estaciones queda en su pestaña (``TableroRio.mostrar``).

Las tarjetas mantienen siempre el mismo orden de ríos (nunca se reordenan por gravedad) y el color de estado se
acompaña siempre de ícono y texto. Usa el contrato de ``TableroRio``: ``resumen()``, ``alertas()``,
``figura_compacta(unidad)``, ``pausar()``, ``reanudar()`` y ``avanzar(n)``.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from hidrologia import COLORES_ESTADO, SIN_DATO, rango_estado

ICONOS_ESTADO: dict[str, str] = {"normal": "✅", "prealerta": "⚠️", "alerta": "🚨", SIN_DATO: "❔"}
COLUMNAS_REGISTRO: dict[str, str] = {
    "rio": "Río", "emitida": "Emitida", "origen": "Tipo", "estado": "Estado", "t_objetivo": "Para",
    "nivel_cm": "Nivel (cm)", "caudal_m3s": "Caudal (m³/s)", "resultado": "Resultado",
}


# ------------------------------------------------------------------ utilidades de formato
def _finito(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _color(estado: str) -> str:
    return COLORES_ESTADO.get(estado, COLORES_ESTADO[SIN_DATO])


def _etiqueta(estado: str) -> str:
    """Ícono + texto en mayúsculas (nunca solo color)."""
    return f"{ICONOS_ESTADO.get(estado, ICONOS_ESTADO[SIN_DATO])} {str(estado).upper()}"


def _hora(t_local: Any) -> str:
    """HH:MM de una hora local en texto ("2024-03-01 19:00" → "19:00")."""
    return str(t_local)[-5:] if t_local else "—"


def _tendencia(cm_h: Any) -> str:
    if not _finito(cm_h):
        return "→ —"
    v = float(cm_h)
    flecha = "→" if abs(v) < 0.5 else ("↑" if v > 0 else "↓")
    return f"{flecha} {v:+.1f} cm/h"


def _probabilidad(p: dict) -> str:
    """P del propio nivel de estado del pronóstico (alerta → P(alerta); si no, P(prealerta))."""
    clave, nombre = ("p_alerta", "alerta") if p.get("estado") == "alerta" else ("p_prealerta", "prealerta")
    valor = p.get(clave)
    return f"P({nombre}) {float(valor) * 100:.0f} %" if _finito(valor) else ""


def _consenso(p: dict) -> str:
    c = p.get("consenso") or {}
    if not c or not c.get("n"):
        return ""
    return f"{c.get('votos', {}).get(p.get('estado'), 0)}/{c['n']} mod."


def _detalle_pronostico(p: dict) -> str:
    return " · ".join(x for x in (_probabilidad(p), _consenso(p)) if x)


def peor_estado(r: dict) -> str:
    """El estado más grave entre el observado y los pronosticados."""
    estados = [r.get("estado_obs", SIN_DATO)] + [p.get("estado", SIN_DATO) for p in r.get("pronosticos", {}).values()]
    return max(estados, key=rango_estado)


# ------------------------------------------------------------------ controles globales
def barra_global(tableros: dict) -> None:
    """Simular/pausar y avanzar todos los ríos a la vez."""
    alguno = any(t.reproduciendo for t in tableros.values())
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 4])
    if c1.button("⏸ Pausar todos" if alguno else "▶ Simular todos", key="global_simular", use_container_width=True):
        for t in tableros.values():
            (t.pausar if alguno else t.reanudar)()
        st.rerun()
    for col, n, texto, clave in ((c2, 1, "⏭ +1 hora (todos)", "global_mas1"),
                                 (c3, 24, "⏩ +24 horas (todos)", "global_mas24")):
        if col.button(texto, key=clave, use_container_width=True):
            for t in tableros.values():
                t.avanzar(n)
            st.rerun()
    c4.caption("Controles para los cuatro ríos a la vez. Cada pestaña conserva sus propios controles, fecha de "
               "inicio y configuración.")


# ------------------------------------------------------------------ tarjetas
def tarjeta_html(rio: str, r: dict) -> str:
    """Tarjeta de un río: estado actual, tendencia, pronósticos por horizonte y calidad de datos."""
    estado = r.get("estado_obs", SIN_DATO)
    c_obs, c_borde = _color(estado), _color(peor_estado(r))
    nivel = f"{r['nivel']:,.1f} cm" if _finito(r.get("nivel")) else "— cm"
    caudal = f"{r['caudal']:,.1f} m³/s" if _finito(r.get("caudal")) else "— m³/s"
    lineas = []
    for h, p in r.get("pronosticos", {}).items():
        e, det = p.get("estado", SIN_DATO), _detalle_pronostico(p)
        lineas.append(f"<div>+{h} h ({_hora(p.get('t_local'))}): <b style='color:{_color(e)}'>{_etiqueta(e)}</b>"
                      f"{' · ' + det if det else ''}</div>")
    n_mal = int(r.get("sensores_problema") or 0)
    calidad = f"📡 {n_mal} sensores imputados/sin dato" if n_mal else "📡 sensores OK"
    marca = "▶ simulando" if r.get("simulando") else "⏸ en pausa"
    punto = f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{c_obs}'></span>"
    return f"""
<div data-rio="{rio}" data-estado="{estado}" style="border-radius:12px;padding:10px 14px;border:2px solid {c_borde};
background:{c_borde}14;color:inherit;line-height:1.45">
  <div style="font-size:1.15rem;font-weight:700">{r.get('nombre', rio)}</div>
  <div style="font-size:1.05rem">{punto} <b style="color:{c_obs}">{_etiqueta(estado)}</b></div>
  <div>{nivel} · {caudal} &nbsp; <span style="opacity:0.85">{_tendencia(r.get('tendencia_cm_h'))}</span></div>
  <div style="font-size:0.85rem;margin-top:4px">{''.join(lineas) or '<div>Sin pronóstico</div>'}</div>
  <div style="font-size:0.8rem;margin-top:4px">{calidad}</div>
  <div style="font-size:0.75rem;opacity:0.7">Hora simulada {r.get('ts', '—')} · {marca}</div>
</div>"""


def tarjetas_alerta(tableros: dict) -> None:
    """Una tarjeta por río en orden fijo."""
    for col, (rio, t) in zip(st.columns(len(tableros)), tableros.items()):
        col.markdown(tarjeta_html(rio, t.resumen()), unsafe_allow_html=True)


# ------------------------------------------------------------------ atención y registro
def _motivos(r: dict) -> list[tuple[int, int, str]]:
    """(gravedad, horizonte, texto) de cada prealerta/alerta observada o pronosticada de un río."""
    motivos = []
    v = r.get("tendencia_cm_h")
    tend = ""
    if _finito(v):
        v = float(v)
        tend = " · estable" if abs(v) < 0.5 else f" · {'subiendo' if v > 0 else 'bajando'} {abs(v):.1f} cm/h"
    e = r.get("estado_obs", SIN_DATO)
    if rango_estado(e) >= 1:
        motivos.append((rango_estado(e), 0, f"{_etiqueta(e)} observada ahora{tend}"))
    for h, p in r.get("pronosticos", {}).items():
        e = p.get("estado", SIN_DATO)
        if rango_estado(e) >= 1:
            det = _detalle_pronostico(p)
            motivos.append((rango_estado(e), int(h), f"{_etiqueta(e)} pronosticada a +{h} h "
                            f"({_hora(p.get('t_local'))}){' · ' + det if det else ''}{tend}"))
    return motivos


def panel_atencion(tableros: dict) -> None:
    """Ríos con prealerta o alerta (observada o pronosticada), del más grave y próximo al menos."""
    st.subheader("Requiere atención")
    filas = []
    for t in tableros.values():
        r = t.resumen()
        motivos = _motivos(r)
        if motivos:
            g, h, texto = max(motivos, key=lambda m: (m[0], -m[1]))
            filas.append((g, h, f"**{r.get('nombre', t.rio)}** — {texto}"))
    if not filas:
        st.success("Sin prealertas ni alertas en los 4 ríos.")
        return
    for _, _, linea in sorted(filas, key=lambda f: (-f[0], f[1])):
        st.markdown(linea)


def registro_consolidado(tableros: dict, n: int = 10) -> None:
    """Las últimas alertas emitidas en los cuatro ríos."""
    st.subheader("Últimas alertas (4 ríos)")
    tablas = [t.alertas() for t in tableros.values()]
    tablas = [x for x in tablas if x is not None and len(x)]
    if not tablas:
        st.caption("Sin alertas todavía.")
        return
    df = pd.concat(tablas, ignore_index=True).sort_values("emitida", ascending=False).head(n)
    for col in ("emitida", "t_objetivo"):
        df[col] = df[col].map(lambda ts: ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "—")
    df = df[list(COLUMNAS_REGISTRO)].rename(columns=COLUMNAS_REGISTRO)
    st.dataframe(df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ curvas
def curvas_2x2(tableros: dict) -> None:
    """Nivel o caudal observado y pronosticado de cada río en una cuadrícula 2×2 de posición fija."""
    st.subheader("Nivel y caudal por río")
    unidad = st.radio("Unidad", ["cm", "m³/s"], horizontal=True, key="global_unidad", label_visibility="collapsed")
    items = list(tableros.items())
    for i in range(0, len(items), 2):
        for col, (rio, t) in zip(st.columns(2), items[i:i + 2]):
            r, u = t.resumen(), t.config["umbrales_m3s"]
            estado = r.get("estado_obs", SIN_DATO)
            with col:
                st.markdown(f"{ICONOS_ESTADO.get(estado, ICONOS_ESTADO[SIN_DATO])} **{r.get('nombre', rio)}** · "
                            f"{str(estado).upper()} · umbrales {u['prealerta']:.0f}/{u['alerta']:.0f} m³/s")
                st.plotly_chart(t.figura_compacta(unidad), use_container_width=True, key=f"{rio}_fig_compacta")


def mostrar(tableros: dict) -> None:
    """Vista general completa (controles, tarjetas, atención, registro y curvas)."""
    barra_global(tableros)
    tarjetas_alerta(tableros)
    izq, der = st.columns([2, 3])
    with izq:
        panel_atencion(tableros)
    with der:
        registro_consolidado(tableros)
    curvas_2x2(tableros)
