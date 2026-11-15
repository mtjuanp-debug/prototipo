"""Figuras Plotly y componentes visuales del simulador (semáforo, nivel/caudal, lluvia)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hidrologia import COLORES_ESTADO, SIN_DATO, caudal_desde_nivel

COLOR_OBS = "#1f4e79"
COLORES_HORIZONTE = ["#e07a1f", "#8e44ad", "#16a085", "#7f8c8d"]  # por posición en la lista de horizontes
COLORES_LLUVIA = ["#2b8cbe", "#41ab5d", "#fd8d3c", "#807dba", "#e7298a", "#66a61e"]


def texto_confianza(p: dict[str, Any]) -> str:
    """'90 %: 172–201 cm · P(prealerta) 62 % · P(alerta) 18 %' a partir de un pronóstico calibrado."""
    if not np.isfinite(p.get("inf90", np.nan)):
        return ""
    return (f"90 %: {p['inf90']:,.0f}–{p['sup90']:,.0f} cm · P(prealerta) {p['p_prealerta']:.0%} · "
            f"P(alerta) {p['p_alerta']:.0%}")


def semaforo_html(titulo: str, estado: str, nivel: float, caudal: float, subtitulo: str = "",
                  confianza: str = "") -> str:
    """Tarjeta tipo semáforo con el estado, el nivel (cm), el caudal (m³/s) y la confianza."""
    color = COLORES_ESTADO.get(estado, COLORES_ESTADO[SIN_DATO])
    txt_n = f"{nivel:,.1f} cm" if np.isfinite(nivel) else "—"
    txt_q = f"{caudal:,.1f} m³/s" if np.isfinite(caudal) else "—"
    return f"""
    <div style="border-radius:14px;padding:14px 16px;background:{color}1f;border:2px solid {color};
                min-height:150px">
      <div style="font-size:0.9rem;opacity:0.8">{titulo}</div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:6px">
        <div style="width:46px;height:46px;border-radius:50%;background:{color};
                    box-shadow:0 0 14px {color}"></div>
        <div style="font-size:1.6rem;font-weight:700;text-transform:uppercase;color:{color}">{estado}</div>
      </div>
      <div style="margin-top:8px;font-size:1.05rem">{txt_n} &nbsp;·&nbsp; {txt_q}</div>
      <div style="font-size:0.8rem;opacity:0.7">{subtitulo}</div>
      <div style="font-size:0.8rem;margin-top:4px">{confianza}</div>
    </div>"""


def _rgba(color: str | None, alfa: float) -> str:
    c = (color or "#888888").lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alfa})"


def _local(indice: pd.DatetimeIndex, zona: str) -> pd.DatetimeIndex:
    return indice.tz_convert(zona).tz_localize(None)


def colores_horizontes(horizontes: list[int]) -> dict[int, str]:
    """Color de cada horizonte según su posición (1.º naranja, 2.º morado), sea cual sea el río."""
    return {int(h): COLORES_HORIZONTE[i % len(COLORES_HORIZONTE)] for i, h in enumerate(horizontes)}


def figura_nivel(obs: pd.Series, registro: pd.DataFrame, ahora: pd.Timestamp, horizontes: list[int],
                 umbrales_cm: dict[str, float], config: dict[str, Any], unidad: str = "cm",
                 horas_atras: int = 72, alto: int = 420, compacta: bool = False) -> go.Figure:
    """Nivel (o caudal) observado frente a los pronósticos +τ, con bandas de estado. ``compacta`` es la versión
    para la vista general: sin leyenda ni marcadores de los otros modelos del top, márgenes y textos cortos."""
    curva = config["curva_descarga"]
    zona = config.get("zona_horaria", "UTC")
    en_q = unidad != "cm"
    conv = (lambda v: caudal_desde_nivel(np.asarray(v, float), curva)) if en_q else (lambda v: np.asarray(v, float))
    t_ini = ahora - pd.Timedelta(hours=horas_atras)
    t_fin = ahora + pd.Timedelta(hours=max(horizontes) + 1)
    colores = colores_horizontes(horizontes)
    fig = go.Figure()
    o = obs[obs.index >= t_ini]
    fig.add_trace(go.Scatter(x=_local(o.index, zona), y=conv(o.values), name="Observado",
                             mode="lines", line=dict(color=COLOR_OBS, width=2.4)))
    if len(registro):
        reg = registro[registro.index >= t_ini - pd.Timedelta(hours=max(horizontes))]
        for h in horizontes:
            col = f"nivel_h{h}"
            if col not in reg:
                continue
            s = reg[col].copy()
            s.index = s.index + pd.Timedelta(hours=h)  # se dibuja en el instante pronosticado
            if f"inf90_h{h}" in reg and reg[f"inf90_h{h}"].notna().any():
                x_b = _local(s.index, zona)
                fig.add_trace(go.Scatter(x=x_b, y=conv(reg[f"sup90_h{h}"].values), mode="lines",
                                         line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x_b, y=conv(reg[f"inf90_h{h}"].values), mode="lines",
                                         line=dict(width=0), fill="tonexty", fillcolor=_rgba(colores[h], 0.15),
                                         name=f"Banda 90 % +{h} h", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=_local(s.index, zona), y=conv(s.values), name=f"Pronóstico +{h} h",
                                     mode="lines", line=dict(color=colores[h], width=1.6, dash="dot")))
            for otra in [] if compacta else [c for c in reg.columns if c.startswith(f"{col}_")]:
                v = reg[otra].iloc[-1]
                if np.isfinite(v):
                    fig.add_trace(go.Scatter(x=_local(pd.DatetimeIndex([ahora + pd.Timedelta(hours=h)]), zona),
                                             y=conv([v]), mode="markers", name=f"{otra.split('_')[-1]} +{h} h",
                                             marker=dict(size=8, color=colores[h], symbol="circle-open",
                                                         line=dict(width=2))))
            ultimo = reg[col].iloc[-1]
            if np.isfinite(ultimo):
                fig.add_trace(go.Scatter(x=_local(pd.DatetimeIndex([ahora + pd.Timedelta(hours=h)]), zona),
                                         y=conv([ultimo]), mode="markers", showlegend=False,
                                         marker=dict(size=11, color=colores[h], symbol="diamond",
                                                     line=dict(color="white", width=1))))
    pre = umbrales_cm["prealerta"]
    ale = umbrales_cm["alerta"]
    lim_pre, lim_ale = (config["umbrales_m3s"]["prealerta"], config["umbrales_m3s"]["alerta"]) if en_q else (pre, ale)
    y_vals = [v for tr in fig.data for v in np.asarray(tr.y, float) if np.isfinite(v)]
    y_max = max([lim_ale * 1.15] + [v * 1.08 for v in y_vals])
    y_min = 0 if en_q else min([lim_pre * 0.6] + [v * 0.97 for v in y_vals])
    for y0, y1, estado in ((y_min, lim_pre, "normal"), (lim_pre, lim_ale, "prealerta"), (lim_ale, y_max, "alerta")):
        fig.add_hrect(y0=y0, y1=y1, fillcolor=COLORES_ESTADO[estado], opacity=0.10, line_width=0, layer="below")
    etiqueta = "m³/s" if en_q else "cm"
    for y, nombre in ((lim_pre, "prealerta"), (lim_ale, "alerta")):
        fig.add_hline(y=y, line=dict(color=COLORES_ESTADO[nombre], width=1, dash="dash"),
                      annotation_text=nombre if compacta else f"{nombre} {y:,.1f} {etiqueta}",
                      annotation_position="top left")
    x_ahora = _local(pd.DatetimeIndex([ahora]), zona)[0]
    fig.add_shape(type="line", x0=x_ahora, x1=x_ahora, y0=0, y1=1, yref="paper",
                  line=dict(color="#555", width=1, dash="dot"))
    fig.add_annotation(x=x_ahora, y=1, yref="paper", text="ahora", showarrow=False, yanchor="bottom")
    rango_x = [_local(pd.DatetimeIndex([t_ini]), zona)[0], _local(pd.DatetimeIndex([t_fin]), zona)[0]]
    if compacta:
        fig.update_layout(height=alto, margin=dict(l=4, r=4, t=24, b=4), hovermode="x unified", showlegend=False,
                          font=dict(size=11), yaxis=dict(title=etiqueta, range=[y_min, y_max]),
                          xaxis=dict(range=rango_x))
        return fig
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis=dict(title=("Caudal (m³/s)" if en_q else "Nivel (cm)"), range=[y_min, y_max]),
                      xaxis=dict(title=f"Hora local ({zona})", range=rango_x), legend=dict(orientation="h", y=-0.2))
    return fig


def figura_tacometro(titulo: str, p: dict[str, Any], caudal_actual: float, umbrales_m3s: dict[str, float],
                     curva: dict[str, float], q_max: float) -> go.Figure:
    """Tacómetro del caudal pronosticado por un modelo: zonas de estado, aguja = pronóstico, línea negra = caudal
    actual, delta frente a ahora y, debajo, la banda del 90 % y las probabilidades de superar los umbrales."""
    q = p.get("caudal_m3s", np.nan)
    pre, ale = umbrales_m3s["prealerta"], umbrales_m3s["alerta"]
    color = COLORES_ESTADO.get(p.get("estado"), COLORES_ESTADO[SIN_DATO])
    detalle = ""
    if np.isfinite(p.get("inf90", np.nan)):
        q_inf, q_sup = caudal_desde_nivel(np.array([p["inf90"], p["sup90"]]), curva)
        detalle = (f"90 %: {q_inf:,.0f}–{q_sup:,.0f} m³/s<br>"
                   f"P(pre) {p['p_prealerta']:.0%} · P(alerta) {p['p_alerta']:.0%}")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if np.isfinite(caudal_actual) else "gauge+number",
        value=q if np.isfinite(q) else None,
        number=dict(suffix=" m³/s", valueformat=",.1f", font=dict(size=22, color=color)),
        delta=dict(reference=caudal_actual, valueformat=",.1f", suffix=" vs ahora", font=dict(size=12),
                   increasing=dict(color=COLORES_ESTADO["alerta"]),
                   decreasing=dict(color=COLORES_ESTADO["normal"])),
        title=dict(text=f"<b>{titulo}</b><br><span style='font-size:0.8em;color:{color}'>"
                        f"{str(p.get('estado', SIN_DATO)).upper()}</span>", font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, q_max], tickvals=[0, pre, ale, q_max], ticktext=["0", f"{pre:.0f}", f"{ale:.0f}",
                                                                                f"{q_max:.0f}"]),
            bar=dict(color="#1f2a36", thickness=0.28),
            steps=[dict(range=[0, pre], color=_rgba(COLORES_ESTADO["normal"], 0.35)),
                   dict(range=[pre, ale], color=_rgba(COLORES_ESTADO["prealerta"], 0.45)),
                   dict(range=[ale, q_max], color=_rgba(COLORES_ESTADO["alerta"], 0.45))],
            threshold=(dict(line=dict(color="black", width=3), thickness=0.85, value=caudal_actual)
                       if np.isfinite(caudal_actual) else None),
        )))
    if detalle:
        fig.add_annotation(text=detalle, x=0.5, y=-0.32, xref="paper", yref="paper", showarrow=False,
                           font=dict(size=11), align="center")
    fig.update_layout(height=280, margin=dict(l=25, r=25, t=60, b=62))
    return fig


def figura_lluvia(crudo: pd.DataFrame, columnas: list[str], etiquetas: dict[str, str], ahora: pd.Timestamp,
                  zona: str, horas_atras: int = 72) -> go.Figure:
    """Lluvia horaria por pluviómetro (barras agrupadas)."""
    d = crudo[crudo.index >= ahora - pd.Timedelta(hours=horas_atras)]
    fig = go.Figure()
    for i, col in enumerate(columnas):
        if col in d:
            fig.add_trace(go.Bar(x=_local(d.index, zona), y=d[col], name=etiquetas.get(col, col),
                                 marker_color=COLORES_LLUVIA[i % len(COLORES_LLUVIA)]))
    fig.update_layout(barmode="group", height=280, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Lluvia (mm/h)", legend=dict(orientation="h", y=-0.3), hovermode="x unified")
    return fig


def figura_meteo(crudo: pd.DataFrame, columnas: list[str], etiquetas: dict[str, str], ahora: pd.Timestamp,
                 zona: str, horas_atras: int = 72) -> go.Figure:
    """Series meteorológicas (humedad, radiación, temperatura) normalizadas en su propio eje."""
    d = crudo[crudo.index >= ahora - pd.Timedelta(hours=horas_atras)]
    fig = go.Figure()
    for i, col in enumerate(columnas):
        if col in d:
            fig.add_trace(go.Scatter(x=_local(d.index, zona), y=d[col], name=etiquetas.get(col, col),
                                     mode="lines", line=dict(color=COLORES_LLUVIA[i % len(COLORES_LLUVIA)])))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3),
                      hovermode="x unified")
    return fig
