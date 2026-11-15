"""Tablero de un río dentro del prototipo consolidado (una pestaña por río en ``app.py``).

Cada río tiene su propia simulación, guardada en ``st.session_state[rio]`` (pronosticador, simulador de sensores,
último paso, reproducción y velocidad), y todos sus controles llevan la clave ``"<rio>_…"``. Así, las tres pestañas
conviven: cambiar de pestaña o simular un río no reinicia ni avanza a los otros. La configuración de cada río
(corrida, top de modelos, modelo principal, fecha de inicio y escenarios) está en el expander «⚙ Configuración»
de su pestaña, con los mismos valores por defecto que tenía la barra lateral.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import modelo_adaptador as adaptador
from confianza import calibrar, ranking
from configuracion import cargar_config, columnas_entrada, corridas_disponibles, ruta
from graficos import (figura_lluvia, figura_meteo, figura_nivel, figura_tacometro, semaforo_html,
                      texto_confianza)
from hidrologia import SIN_DATO, rango_estado, umbrales_cm
from imputacion_causal import ParametrosImputacion, ajustar_parametros
from pronosticador import Pronosticador, ResultadoPaso
from sensores import FallaSensor, SimuladorSensores, Tormenta, ValorCongelado, cargar_datos_observados

VELOCIDADES = {"🐢 Lenta (1 h cada 2 s)": (1, 2.0), "▶ Normal (1 h por segundo)": (1, 1.0),
               "🐇 Rápida (1 h cada 0,3 s)": (1, 0.3), "⚡ Turbo (6 h por paso)": (6, 0.5)}
VELOCIDAD_DEFECTO = "▶ Normal (1 h por segundo)"
UNIDADES = {"Level_Avg": "cm", "Lluvia_Tot": "mm/h", "HumAire_Avg": "%", "RS_W_Avg": "W/m²", "TempAire_Avg": "°C"}


@st.cache_resource(show_spinner="Cargando datos y modelos…")
def recursos(rio: str, corrida: str | None) -> tuple[dict, pd.DataFrame, ParametrosImputacion, list[str]]:
    config = cargar_config(rio, corrida)
    datos = cargar_datos_observados(config)
    params = ajustar_parametros(datos.loc[datos["particion"] == "train", columnas_entrada(config)], config)
    contrato = adaptador.cargar_contrato(config)
    disponibles = adaptador.familias_disponibles(contrato)
    orden = [f for f in config.get("familias", []) if f in disponibles]
    orden += [f for f in disponibles if f not in orden]
    return config, datos, params, orden


@st.cache_resource(show_spinner="Calibrando la confianza con los errores de validación…")
def confianza(rio: str, corrida: str | None, familias: tuple[str, ...]) -> tuple[dict, pd.DataFrame]:
    """Calibración (bandas y probabilidades) y ranking por desempeño en crecidas en validación."""
    config = cargar_config(rio, corrida)
    archivo = ruta(config, "predicciones")
    if archivo is None or not archivo.exists():
        return {}, pd.DataFrame()
    p95 = None
    cfg_modelos = ruta(config, "configuracion_modelos")
    if cfg_modelos is not None and cfg_modelos.exists():
        p95 = json.loads(cfg_modelos.read_text(encoding="utf-8")).get("p95_train_cm")
    pred = pd.read_parquet(archivo)
    if p95 is None:
        p95 = float(pred.loc[pred["particion"] == "train", "y_real_cm"].quantile(0.95))
    cal = calibrar(pred, umbrales_cm(config), float(p95), modelos=list(familias))
    horizontes = config.get("horizontes") or adaptador.cargar_contrato(config).horizontes
    return cal, ranking(cal, [int(h) for h in horizontes], list(familias))


class TableroRio:
    """Datos, modelos y simulación de un río; ``mostrar`` dibuja su pestaña completa."""

    def __init__(self, rio: str) -> None:
        self.rio = rio
        self.corrida = st.session_state.get(f"{rio}_corrida")
        self.corridas = corridas_disponibles(cargar_config(rio))
        self.config, self.datos, self.params, self.familias = recursos(rio, self.corrida)
        self.calibracion, self.ranking = confianza(rio, self.corrida, tuple(self.familias))
        rk = self.ranking
        self.pesos = dict(zip(rk["modelo"], rk["puntaje"])) if len(rk) else {}
        self.orden_top = list(rk["modelo"]) if len(rk) else list(self.familias)
        self.columnas = columnas_entrada(self.config)
        self.zona = self.config.get("zona_horaria", "UTC")
        self.umbrales_cm = umbrales_cm(self.config)
        self.et = self.config.get("etiquetas", {})
        if rio not in st.session_state:
            top = [m for m in self.orden_top if m != "Persistencia"][:3] or self.familias[:1]
            self.iniciar(pd.Timestamp(self.config.get("inicio_defecto", self.datos.index[0])), top)

    # ------------------------------------------------------------------ estado de la simulación
    @property
    def s(self) -> dict[str, Any]:
        return st.session_state[self.rio]

    @property
    def reproduciendo(self) -> bool:
        return bool(self.s["reproduciendo"])

    @property
    def intervalo(self) -> float:
        return VELOCIDADES[self.s.get("velocidad", VELOCIDAD_DEFECTO)][1]

    def a_local(self, ts: pd.Timestamp) -> str:
        return ts.tz_convert(self.zona).strftime("%Y-%m-%d %H:%M")

    def iniciar(self, inicio: pd.Timestamp, top: list[str]) -> None:
        """Crea el simulador y el pronosticador, precarga la historia y procesa la primera hora."""
        pron = Pronosticador(self.config, self.datos, top[0], parametros=self.params)
        pron.configurar_top(top, self.calibracion, self.pesos)
        sim = SimuladorSensores(self.datos, inicio, self.columnas)
        pron.precargar(sim.historia(pron.ventana_h + pron.margen_h))
        anterior = st.session_state.get(self.rio, {})
        st.session_state[self.rio] = {"pron": pron, "sim": sim, "ultimo": pron.recibir(sim.siguiente()),
                                      "inicio": inicio, "reproduciendo": False, "t_ultimo_avance": time.time(),
                                      "avisos": [], "velocidad": anterior.get("velocidad", VELOCIDAD_DEFECTO)}

    def avanzar(self, n_horas: int) -> None:
        s = self.s
        for _ in range(n_horas):
            if s["sim"].agotado:
                s["reproduciendo"] = False
                break
            s["ultimo"] = s["pron"].recibir(s["sim"].siguiente())
            s["avisos"].extend(s["ultimo"].alertas_nuevas)
        s["t_ultimo_avance"] = time.time()

    def avance_automatico(self) -> None:
        """Avanza si el río está simulando y ya pasó su intervalo (se llama en cada recarga de la app)."""
        if self.reproduciendo and time.time() - self.s["t_ultimo_avance"] >= 0.9 * self.intervalo:
            self.avanzar(VELOCIDADES[self.s.get("velocidad", VELOCIDAD_DEFECTO)][0])

    def pausar(self) -> None:
        self.s["reproduciendo"] = False

    def reanudar(self) -> None:
        if not self.s["sim"].agotado:
            self.s["reproduciendo"] = True
            self.s["t_ultimo_avance"] = 0.0

    def _tendencia_cm_h(self) -> float:
        """(nivel ahora − nivel 3 h antes) / 3, desde los valores crudos del sensor de nivel."""
        serie = self.s["pron"].crudo[self.config["objetivo"]]
        if not len(serie):
            return float("nan")
        t = serie.index[-1]
        ahora, antes = serie.iloc[-1], serie.get(t - pd.Timedelta(hours=3), float("nan"))
        return float((ahora - antes) / 3) if pd.notna(ahora) and pd.notna(antes) else float("nan")

    def resumen(self) -> dict[str, Any]:
        """Estado observado, pronósticos, peor estado pronosticado, tendencia y sensores (vista general)."""
        res: ResultadoPaso = self.s["ultimo"]
        crudo = self.s["pron"].crudo
        peor_h, peor = None, SIN_DATO
        pronosticos: dict[int, dict[str, Any]] = {}
        for h, p in res.pronosticos.items():
            if rango_estado(p["estado"]) > rango_estado(peor):
                peor_h, peor = int(h), p["estado"]
            c = res.consenso.get(h) or {}
            pronosticos[int(h)] = {
                "estado": p["estado"], "nivel_cm": float(p["nivel_cm"]), "caudal_m3s": float(p["caudal_m3s"]),
                "t_local": self.a_local(p["t_objetivo"]), "p_prealerta": float(p.get("p_prealerta", float("nan"))),
                "p_alerta": float(p.get("p_alerta", float("nan"))),
                "consenso": {"estado": c["estado"], "n": c["n"], "votos": c["votos"]} if c else {}}
        ultimos = crudo[self.columnas].iloc[-1] if len(crudo) else pd.Series(float("nan"), index=self.columnas)
        problema = {c for c in self.columnas if pd.isna(ultimos.get(c))} | (set(res.imputadas) & set(self.columnas))
        return {"nombre": self.config.get("nombre_rio", self.rio), "ts": self.a_local(res.ts),
                "estado_obs": res.estado_obs, "nivel": float(res.nivel_obs), "caudal": float(res.caudal_obs),
                "simulando": self.reproduciendo, "tendencia_cm_h": self._tendencia_cm_h(),
                "sensores_problema": len(problema), "pronosticos": pronosticos,
                "peor_pronostico": peor, "horizonte_peor": peor_h}

    def alertas(self) -> pd.DataFrame:
        """Registro de alertas del río con las horas en su zona horaria (para el registro consolidado)."""
        columnas = ["rio", "emitida", "origen", "estado", "nivel_cm", "caudal_m3s", "t_objetivo", "resultado"]
        tabla = self.s["pron"].tabla_alertas()
        if not len(tabla):
            return pd.DataFrame(columns=columnas)
        tabla = tabla.assign(rio=self.config.get("nombre_rio", self.rio))
        for c in ("emitida", "t_objetivo"):
            tabla[c] = pd.to_datetime(tabla[c], utc=True).dt.tz_convert(self.zona)
        return tabla[columnas].reset_index(drop=True)

    def figura_compacta(self, unidad: str) -> go.Figure:
        """Gráfico de nivel/caudal compacto (últimas 48 h + pronósticos) para la vista general."""
        pron: Pronosticador = self.s["pron"]
        return figura_nivel(pron.crudo[self.config["objetivo"]], pron.tabla_registro(), self.s["ultimo"].ts,
                            pron.horizontes, self.umbrales_cm, self.config, unidad, horas_atras=48, alto=260,
                            compacta=True)

    # ------------------------------------------------------------------ pestaña
    def mostrar(self) -> None:
        self._configuracion()
        self._barra()
        self._sensores()
        self._semaforos_y_top()
        self._mensajes()
        self._graficos_y_registro()

    def _configuracion(self) -> None:
        k_ = self.rio
        pron: Pronosticador = self.s["pron"]
        sim: SimuladorSensores = self.s["sim"]
        with st.expander(f"⚙ Configuración de {self.config.get('nombre_rio', self.rio)}"):
            modelos, inicio, escenarios = st.columns(3)
            with modelos:
                st.markdown("**Modelos**")
                if len(self.corridas) > 1:
                    claves = list(self.corridas)
                    corrida = st.selectbox("Corrida de modelos", claves,
                                           index=claves.index(self.config.get("corrida", claves[0])),
                                           format_func=lambda k: self.corridas[k], key=f"{k_}_sel_corrida",
                                           help="Por defecto, los modelos finales. La corrida preliminar sigue "
                                                "disponible.")
                    if corrida != self.config.get("corrida"):
                        del st.session_state[self.rio]
                        st.session_state[f"{k_}_corrida"] = corrida
                        st.rerun()
                elif self.corridas:
                    st.caption(f"Corrida: {next(iter(self.corridas.values()))}")
                if len(self.ranking):
                    rk = self.ranking
                    cols_rk = ["rango", "modelo", "puntaje"] + [c for c in rk if c.startswith("eventos_h")]
                    st.dataframe(rk[cols_rk].rename(columns=lambda c: c.replace("eventos_h", "eventos +") + " h"
                                                    if c.startswith("eventos_h") else c),
                                 hide_index=True, use_container_width=True,
                                 column_config={"puntaje": st.column_config.ProgressColumn(
                                     "puntaje crecidas", min_value=0, max_value=1, format="%.2f")})
                    st.caption("Orden por desempeño en crecidas en validación: eventos de prealerta anticipados "
                               "× (1 − falsas alarmas). Empate (< 0,01) → menor error en niveles altos.")
                n = len(self.orden_top)
                k = st.slider("Cuántos modelos correr", 1, n, min(3, n), key=f"{k_}_k",
                              help="Se toman los primeros del ranking; abajo puedes cambiar la selección a mano.")
                elegidos = st.multiselect("Modelos a ejecutar", self.orden_top,
                                          default=[m for m in self.orden_top if m != "Persistencia"][:k]
                                          or self.orden_top[:k], key=f"{k_}_top_{k}")
                if elegidos:
                    principal = st.selectbox("Modelo principal (emite las alertas)", elegidos, index=0,
                                             key=f"{k_}_principal",
                                             help="Sus pronósticos alimentan los semáforos y el registro de "
                                                  "alertas.")
                    top = [principal] + [m for m in elegidos if m != principal]
                    if top != pron.top:
                        pron.configurar_top(top, self.calibracion, self.pesos)
            with inicio:
                st.markdown("**Inicio de la simulación**")
                test = self.datos.index[self.datos["particion"] == "test"]
                fecha = st.date_input("Fecha de inicio (UTC)",
                                      value=pd.Timestamp(self.config["inicio_defecto"]).date(),
                                      min_value=self.datos.index[0].date(), max_value=self.datos.index[-1].date(),
                                      key=f"{k_}_fecha",
                                      help=f"Periodo de prueba: {test[0].date()} a {test[-1].date()}.")
                hora = st.number_input("Hora (UTC)", 0, 23, 0, key=f"{k_}_hora")
                if st.button("↺ Reiniciar en la fecha", use_container_width=True, key=f"{k_}_reiniciar_fecha"):
                    self.iniciar(pd.Timestamp(fecha).tz_localize("UTC") + pd.Timedelta(hours=int(hora)), pron.top)
                    st.rerun()
            with escenarios:
                st.markdown("**🌧 Tormenta sintética**")
                lluvia = self.config["estaciones"]["lluvia"]
                mm = st.slider("Lámina total por pluviómetro (mm)", 5, 100, 30, key=f"{k_}_mm")
                dur = st.slider("Duración (h)", 1, 12, 4, key=f"{k_}_dur")
                forma = st.selectbox("Forma del hietograma", ["triangular", "uniforme"], key=f"{k_}_forma")
                est = st.multiselect("Pluviómetros", lluvia, lluvia, format_func=lambda c: self.et.get(c, c),
                                     key=f"{k_}_pluvios")
                if st.button("Inyectar tormenta en la próxima hora", use_container_width=True,
                             key=f"{k_}_tormenta") and est:
                    sim.agregar_escenario(Tormenta(sim.t_siguiente, dur, est, total_mm=float(mm), forma=forma))
                st.markdown("**⚠ Falla de sensor / valor congelado**")
                sensor = st.selectbox("Sensor", self.columnas, format_func=lambda c: self.et.get(c, c),
                                      key=f"{k_}_sensor")
                dur_f = st.slider("Duración de la anomalía (h)", 1, 96, 12, key=f"{k_}_dur_f")
                f1, f2 = st.columns(2)
                if f1.button("Falla (sin datos)", use_container_width=True, key=f"{k_}_falla"):
                    sim.agregar_escenario(FallaSensor(sim.t_siguiente, dur_f, [sensor]))
                if f2.button("Congelar valor", use_container_width=True, key=f"{k_}_congelar"):
                    sim.agregar_escenario(ValorCongelado(sim.t_siguiente, dur_f, [sensor]))
                for e in sim.escenarios:
                    nombres = ", ".join(self.et.get(c, c).split(" (")[0] for c in e.columnas)
                    st.caption(f"• {e.describir()} — {nombres}; {self.a_local(e.inicio)} → {self.a_local(e.fin)}")
                if sim.escenarios and st.button("Quitar escenarios", use_container_width=True,
                                                key=f"{k_}_quitar"):
                    sim.limpiar_escenarios()

    def _barra(self) -> None:
        k_, s = self.rio, self.s
        b0, b1, b2, b3, b4 = st.columns([1, 1.3, 1, 1, 2.2])
        if b0.button("⏮ Reiniciar", use_container_width=True, key=f"{k_}_reiniciar",
                     help="Vuelve al inicio de esta simulación (en pausa) y borra el registro y las alertas. Para "
                          "empezar en otra fecha, usa «Reiniciar en la fecha» en la configuración."):
            self.iniciar(s.get("inicio", pd.Timestamp(self.config["inicio_defecto"])), s["pron"].top)
            st.rerun()
        if b1.button("⏸ Pausar" if s["reproduciendo"] else "▶ Simular", type="primary", use_container_width=True,
                     key=f"{k_}_simular", help="Los sensores de este río envían una lectura por hora simulada y "
                                               "los modelos pronostican en cada paso."):
            s["reproduciendo"] = not s["reproduciendo"]
            s["t_ultimo_avance"] = 0.0
            st.rerun()
        if b2.button("⏭ +1 hora", use_container_width=True, key=f"{k_}_mas1"):
            self.avanzar(1)
            st.rerun()  # la vista general se dibuja antes que las pestañas
        if b3.button("⏩ +24 horas", use_container_width=True, key=f"{k_}_mas24"):
            self.avanzar(24)
            st.rerun()
        s["velocidad"] = b4.select_slider("Velocidad", list(VELOCIDADES), value=s.get("velocidad", VELOCIDAD_DEFECTO),
                                          label_visibility="collapsed", key=f"{k_}_velocidad")
        res: ResultadoPaso = s["ultimo"]
        t_ini, t_fin = self.datos.index[self.datos["particion"] == "test"][[0, -1]]
        avance = min(max((res.ts - t_ini) / (t_fin - t_ini), 0.0), 1.0)
        st.progress(avance, text=f"{'🟢 Simulando' if s['reproduciendo'] else '⏸ En pausa'} · hora simulada "
                                 f"**{self.a_local(res.ts)}** · {len(s['pron'].registro)} horas procesadas · "
                                 f"{avance:.0%} del periodo de prueba")

    def _tarjeta_sensor(self, col, columna: str, decimales: int = 1) -> None:
        """Valor crudo que acaba de llegar del sensor, su cambio frente a la hora anterior y su estado."""
        res: ResultadoPaso = self.s["ultimo"]
        serie = self.s["pron"].crudo[columna]
        actual = serie.iloc[-1] if len(serie) else float("nan")
        previo = serie.iloc[-2] if len(serie) > 1 else float("nan")
        unidad = UNIDADES.get(columna.split("__")[-1], "")
        nombre = self.et.get(columna, columna).split(" (")[0]
        if pd.isna(actual):
            col.metric(f"❌ {nombre}", "sin dato", "imputado" if columna in res.imputadas else None,
                       delta_color="off", help="El sensor no envió dato en esta hora (falla o hueco del registro).")
            return
        delta = None if pd.isna(previo) else f"{actual - previo:+.{decimales}f} {unidad}"
        marca = "⚠ " if columna in res.imputadas else "🟢 "
        col.metric(marca + nombre, f"{actual:,.{decimales}f} {unidad}", delta,
                   delta_color="inverse" if "Lluvia" in columna or "Level" in columna else "normal")

    def _sensores(self) -> None:
        res: ResultadoPaso = self.s["ultimo"]
        est = self.config["estaciones"]
        with st.container(border=True):
            st.markdown(f"**📡 Sensores en vivo** — lectura de las {self.a_local(res.ts)} "
                        "(🟢 dato recibido · ⚠ imputado · ❌ sin dato)")
            fila1 = st.columns(2 + len(est["lluvia"]))
            self._tarjeta_sensor(fila1[0], self.config["objetivo"])
            fila1[1].metric("💧 Caudal estimado", f"{res.caudal_obs:,.1f} m³/s" if pd.notna(res.caudal_obs) else "—",
                            res.estado_obs, delta_color="off")
            for col, c in zip(fila1[2:], est["lluvia"]):
                self._tarjeta_sensor(col, c)
            otros = est["nivel"][1:] + est["meteo"]
            for i in range(0, len(otros), 6):  # filas de hasta 6 tarjetas (el Tomebamba tiene 12 sensores más)
                for col, c in zip(st.columns(6), otros[i:i + 6]):
                    self._tarjeta_sensor(col, c, 0 if "RS_W" in c else 1)

    def _semaforos_y_top(self) -> None:
        pron: Pronosticador = self.s["pron"]
        res: ResultadoPaso = self.s["ultimo"]
        st.caption(f"Hora simulada: **{self.a_local(res.ts)}** ({self.zona}) · corrida "
                   f"**{self.config.get('corrida', '-')}** · modelo **{pron.familia}** · variables: "
                   f"{adaptador.origen_variables(pron.contrato)} · contrato: {pron.contrato.origen} · "
                   f"objetivo: {pron.contrato.tipo_objetivo}")
        cols = st.columns(1 + len(res.pronosticos))
        cols[0].markdown(semaforo_html("Estado actual (observado)", res.estado_obs, res.nivel_obs, res.caudal_obs,
                                       self.a_local(res.ts)), unsafe_allow_html=True)
        for col, (h, p) in zip(cols[1:], res.pronosticos.items()):
            col.markdown(semaforo_html(f"Pronóstico +{h} h ({pron.familia})", p["estado"], p["nivel_cm"],
                                       p["caudal_m3s"], f"para {self.a_local(p['t_objetivo'])}", texto_confianza(p)),
                         unsafe_allow_html=True)
        if not res.top:
            return
        st.subheader("Top de modelos: pronóstico, confianza y consenso")
        rk = self.ranking
        rango = dict(zip(rk["modelo"], rk["rango"])) if len(rk) else {}
        um = self.config["umbrales_m3s"]
        q_vals = [p["caudal_m3s"] for pm in res.top.values() for p in pm.values() if pd.notna(p["caudal_m3s"])]
        q_vals += [res.caudal_obs] if pd.notna(res.caudal_obs) else []
        q_max = max([um["alerta"] * 1.6] + [v * 1.15 for v in q_vals])
        for h, por_modelo in res.top.items():
            st.markdown(f"**+{h} h** — caudal pronosticado para {self.a_local(res.ts + pd.Timedelta(hours=h))}")
            for col, (m, p) in zip(st.columns(len(por_modelo)), por_modelo.items()):
                titulo = f"#{rango.get(m, '–')} {m}" + (" ★" if m == pron.familia else "")
                col.plotly_chart(figura_tacometro(titulo, p, res.caudal_obs, um, self.config["curva_descarga"], q_max),
                                 use_container_width=True, key=f"{self.rio}_tacometro_{h}_{m}")
            c = res.consenso.get(h) or {}
            if c:
                votos = " · ".join(f"{n} {e}" for e, n in sorted(c["votos"].items(), key=lambda x: -x[1]))
                st.caption(f"Consenso ({c['n']} modelos): {votos}. Nivel combinado (mediana ponderada por "
                           f"puntaje): **{c['nivel_cm']:,.1f} cm** → {c['caudal_m3s']:,.1f} m³/s, "
                           f"**{c['estado']}**; rango entre modelos {c['min_cm']:,.0f}–{c['max_cm']:,.0f} cm.")
        with st.expander("Ver la tabla de pronósticos del top"):
            for h, por_modelo in res.top.items():
                filas = [{"+h": h, "#": rango.get(m, ""), "modelo": m + (" ★" if m == pron.familia else ""),
                          "nivel cm": p["nivel_cm"],
                          "banda 90 %": f"{p['inf90']:,.0f}–{p['sup90']:,.0f}" if "inf90" in p else "—",
                          "caudal m³/s": p["caudal_m3s"], "estado": p["estado"],
                          "P(prealerta)": p.get("p_prealerta"), "P(alerta)": p.get("p_alerta")}
                         for m, p in por_modelo.items()]
                st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True,
                             column_config={"nivel cm": st.column_config.NumberColumn(format="%.1f"),
                                            "caudal m³/s": st.column_config.NumberColumn(format="%.1f"),
                                            "P(prealerta)": st.column_config.NumberColumn(format="percent"),
                                            "P(alerta)": st.column_config.NumberColumn(format="percent")})
        st.caption("Tacómetro: la aguja es el caudal pronosticado, la línea negra el caudal actual y el número pequeño "
                   "cuánto sube (rojo) o baja (verde) frente a ahora. ★ modelo principal; #n su puesto en el ranking. "
                   "La banda y las probabilidades salen de los errores reales de cada modelo en validación (fuera de "
                   "muestra) para pronósticos de nivel parecido; son algo conservadoras porque esos modelos se "
                   "entrenaron con menos datos que el modelo final.")

    def _mensajes(self) -> None:
        res: ResultadoPaso = self.s["ultimo"]
        nombre = self.config.get("nombre_rio", self.rio)
        if res.mensaje:
            st.warning(res.mensaje)
        for a in self.s["avisos"]:
            if a["origen"] != "observado":
                st.toast(f"⚠ {nombre}: {a['estado'].upper()} pronosticada ({a['origen']}) para "
                         f"{self.a_local(a['t_objetivo'])}")
            else:
                st.toast(f"🚨 {nombre}: {a['estado'].upper()} observada a las {self.a_local(a['emitida'])}")
        self.s["avisos"] = []

    def _graficos_y_registro(self) -> None:
        k_ = self.rio
        pron: Pronosticador = self.s["pron"]
        res: ResultadoPaso = self.s["ultimo"]
        est = self.config["estaciones"]
        izq, der = st.columns([3, 2])
        with izq:
            unidad = st.radio("Unidad", ["cm", "m³/s"], horizontal=True, label_visibility="collapsed",
                              key=f"{k_}_unidad")
            st.plotly_chart(figura_nivel(pron.crudo[self.config["objetivo"]], pron.tabla_registro(), res.ts,
                                         pron.horizontes, self.umbrales_cm, self.config, unidad),
                            use_container_width=True, key=f"{k_}_fig_nivel")
        with der:
            st.markdown("**Lluvia por estación (mm/h)**")
            st.plotly_chart(figura_lluvia(pron.crudo, est["lluvia"], self.et, res.ts, self.zona),
                            use_container_width=True, key=f"{k_}_fig_lluvia")
            um = self.config["umbrales_m3s"]
            st.caption(f"Umbrales: prealerta {um['prealerta']:.0f} m³/s (≈ {self.umbrales_cm['prealerta']:.1f} cm) · "
                       f"alerta {um['alerta']:.0f} m³/s (≈ {self.umbrales_cm['alerta']:.1f} cm). "
                       "Curva Q = a·(H − H0)^p.")
        st.subheader("Registro de alertas")
        alertas = pron.tabla_alertas()
        if len(alertas):
            vista = alertas.copy()
            for c in ("emitida", "t_objetivo", "confirmada"):
                vista[c] = vista[c].map(lambda t: self.a_local(t) if isinstance(t, pd.Timestamp) else "")
            vista = vista[["emitida", "origen", "estado", "nivel_cm", "caudal_m3s", "t_objetivo",
                           "anticipacion_prevista_h", "resultado", "confirmada", "anticipacion_real_h", "modelo"]]
            st.dataframe(vista.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.caption("Sin alertas en lo que va de la simulación.")
        with st.expander("Meteorología y datos de la simulación"):
            if est["meteo"]:
                st.plotly_chart(figura_meteo(pron.crudo, est["meteo"], self.et, res.ts, self.zona),
                                use_container_width=True, key=f"{k_}_fig_meteo")
            else:
                st.caption("Los modelos de este río no usan estaciones meteorológicas.")
            reg = pron.tabla_registro()
            if len(reg):
                st.download_button("Descargar registro (CSV)", reg.to_csv().encode("utf-8"),
                                   file_name=f"registro_simulacion_{self.rio}.csv", mime="text/csv",
                                   key=f"{k_}_descargar")
                st.dataframe(reg.tail(48).iloc[::-1], use_container_width=True)
