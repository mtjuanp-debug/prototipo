## Especificación común de entrenamiento y evaluación (4 ríos)

Especificación común para los cuatro ríos. Se deriva de los capítulos 1 y 2 de la tesis
(`Docs/TesisCardenasPlazaSanchezRivadeneira.docx`), que son la **fuente de verdad**: si algo de este README
contradice la tesis, se corrige el README. Las citas "(Tesis §x, Pnn)" remiten a la sección y al número de
párrafo no vacío del documento (numerado desde 0).

| Responsable | Río | Estación de nivel (Tabla 2) |
|---|---|---|
| Juan | Machángara | LIM_MachángaraLlantera |
| Ruth | Tarqui | LIM_Tarqui-AJ-Yanuncay |
| César | Tomebamba | LIM_Tomebamba-DJ-Mazan |
| Andrés | Yanuncay | LIM_Yanuncay-AJ-Tarqui |

### 1. Objetivo y variable objetivo
- Se predice el **nivel del agua en cm**, con resolución horaria, en la estación de cada río (Tesis §1.3, P45).
  El problema es de **regresión**. Los estados de alerta se derivan del nivel y no hay un clasificador aparte (P68, P196).
- Etiqueta: `y = y_obs.shift(-τ)`. `y_obs` es el nivel **observado** y **nunca se imputa**. Las filas sin `y_obs`
  observado en t+τ no entran ni en el entrenamiento ni en la evaluación.
- Todas las métricas se calculan en cm (después de `inverse_target`) y **solo sobre niveles observados** (P48, P159).

### 2. Horizontes por río (Tesis §2.2, P117–P121)
- **Tr** es el tiempo de respuesta (*lag time*): el intervalo entre el centro de masa de la lluvia y el pico del nivel.
  Se estima por evento, **solo en el train**, y se reportan la mediana y el rango. Se contrasta con la literatura:
  en el Tomebamba es de 1 a 4 h (P120).
- **τ₂ = ⌊Tr⌋**: anticipación máxima sin pronóstico de lluvia.
- **τ₁ = max(1, round(τ₂/2))**.
- Si τ₁ = τ₂, se evalúa un solo horizonte.
- Los horizontes se **fijan antes de entrenar** y no cambian según el desempeño. Los resultados se comparan también con τ/Tr.

### 3. Partición (Tesis §1.3, P55 y P57; §2.2, P163)
- Partición **cronológica** por río: 70 % entrenamiento, 15 % validación y 15 % prueba. No se mezcla al azar.
  Justificación en `Docs/Informe partición 70-15-15.docx`.
- **Margen** entre train/val y val/test = **ventana máxima de rezagos + τ₂** horas. Las filas del margen se descartan.
- **Eventos mínimos:** val y test deben tener cada uno **≥ 3 eventos de prealerta**. Si no los tienen, los cortes se
  desplazan hacia atrás conservando la proporción val/test. Esto se hace **una sola vez**, antes de entrenar, solo con
  los niveles observados y sin mirar ningún modelo. Las fechas finales se documentan y no se vuelven a tocar.
- **Normalización:** Min-Max ajustada **solo en el train** y aplicada sin reajuste a val y test (P162). Solo es necesaria
  para las redes; a los árboles no les afecta.

### 4. Datos faltantes y control de calidad (Tesis §1.3, P46–P48; §2.2, P143–P159)
**Control de calidad**
- Se aplican límites físicos por variable.
- **Niveles congelados:** si el nivel no cambia durante 24 h o más, esos valores pasan a NaN.
- **Radiación:** solo el límite físico (0–1500 W/m²) y el chequeo nocturno. No se usa IQR porque borra picos reales.

**Regla de 72 h (P47)**
- Los huecos de **más de 72 h no se imputan**.
- Toda fila cuyo objetivo o cuyas entradas (incluidos la ventana de rezagos y los acumulados) toquen un hueco largo
  queda con `valido = 0` y se excluye del entrenamiento y la evaluación.
- **Selección de estaciones de entrada por costo:** una estación se conserva solo si las horas de train que excluiría,
  más allá de las que ya excluye el propio objetivo, son **≤ 5 % del train**. La tabla de costos se documenta.
- **Regla de cuenca:** se conserva siempre al menos un pluviómetro de la propia cuenca (el de menor costo), aunque
  supere el 5 %, porque es la lluvia que anticipa la crecida y la tesis asocia a cada río las estaciones de su cuenca
  (Tesis §1.3, P45).

**Métodos por tipo de variable** (parámetros ajustados solo en el train)

| Variable | Métodos comparados |
|---|---|
| Nivel | Lineal, PCHIP (Fritsch y Carlson, 1980) y regresión con una estación vecina con el **desfase óptimo** (correlación cruzada). Climatología hora×mes **solo** en huecos de 24–72 h **fuera de crecidas** (P147) |
| Precipitación | **CCW** como método prioritario (Teegavarapu y Chandramouli, 2005); IDW como alternativa (P152) |
| Temperatura, HR, radiación | **missForest iterativo** (Stekhoven y Bühlmann, 2012) frente a interpolación y climatología (P154) |

**Validación de la imputación (P48, P156–P159)**
- Huecos artificiales de **1, 6, 24 y 72 h**, generados solo en el train.
- Métricas: MAE, RMSE, sesgo, nMAE y nRMSE **normalizados por el rango** (máx − mín observado en train).
- En la lluvia se evalúa además la detección de horas con y sin lluvia.

**Trazabilidad y sin información futura**
- Cada variable imputada tiene una columna indicadora `__imp`.
- Ninguna variable del instante t usa valores posteriores a t. Las filas que dependen de interpolaciones ancladas
  después de t se marcan como no válidas.

### 5. Variables de entrada (Tesis §1.3, P49; §2.2, P161)
No hay una lista fija de variables: se aplica una **regla de construcción** y el conteo resultante se reporta por río.
- Rezagos del nivel objetivo y de las estaciones de nivel vecinas.
- Acumulados de precipitación en varias ventanas.
- Variaciones del nivel (deltas).
- Variables cíclicas de hora y día del año (seno y coseno).
- Humedad del suelo, si la cuenca la registra.
- Indicadores `__imp`.

La selección final de ventanas se apoya en el análisis exploratorio y en Tr. Los árboles reciben la fila tabular del
instante t. **RNN y LSTM** reciben una ventana de las últimas **L = 24 h**.

### 6. Modelos (Tesis §1.3, P54; §2.2, P129–P139)
- **Persistencia:** ŷ(t+τ) = y(t). Es la referencia mínima y no se entrena.
- **Random Forest** (Breiman, 2001).
- **XGBoost** (Chen y Guestrin, 2016) con `objective="reg:squarederror"`.
- **AdaBoost.R2** (Drucker, 1997): `AdaBoostRegressor(estimator=DecisionTreeRegressor(...))`.
- **RNN base** y **LSTM** (Hochreiter y Schmidhuber, 1997) en PyTorch.

**Formulación por defecto (fase A, común a los cuatro ríos):** los modelos aprenden el cambio
Δ = y_obs(t+τ) − nivel(t), con las variables de lluvia extendidas y sin pesos de crecida (α = 0); el pronóstico de
nivel es nivel(t) + Δ̂ y la persistencia equivale a Δ̂ = 0. Las métricas y la selección siguen en nivel (cm). Es la
ganadora de la ablación walk-forward del Yanuncay (`Docs/resources/modelos_yanuncay/tablas/ablacion_walkforward.csv`);
se cambia con `<RIO>_OBJETIVO`, `<RIO>_VARIABLES` y `<RIO>_PESO_ALFA`.

Se usa la **estrategia directa**: un modelo por horizonte. Todo es **regresión**, así que se eliminan `class_weight`,
`scale_pos_weight`, `binary:logistic` y `max_features="sqrt"`. El diagrama de la red está en `Docs/Diagrama Red Neuronal.docx`.

### 7. Búsqueda de hiperparámetros (Tesis §1.3, P55; §2.2, P141)
**Procedimiento**
- Búsqueda aleatoria (Bergstra y Bengio, 2012) con **walk-forward expansivo** (Cerqueira et al., 2020).
- La validación se divide en **5 bloques consecutivos** de igual duración.
- El fold k entrena con train + los bloques anteriores a k y evalúa en el bloque k. Entre bloques se deja el mismo margen
  de la sección 3.
- **Test nunca participa en la búsqueda.**
- **Early stopping** de XGBoost y de las redes: con el último 10 % cronológico del entrenamiento de cada fold.
- Presupuesto orientativo y configurable por modelo y horizonte: **~30 configuraciones en los árboles y ~12 en las redes**.

**Espacios de búsqueda**

Parten de los rangos anteriores y se ajustan con la evidencia del primer entrenamiento del Yanuncay
(`Docs/Analisis_Preliminar_Modelos_Yanuncay.docx`).

| Modelo | Hiperparámetro | Rango |
|---|---|---|
| RF | n_estimators | 200–500 |
| | max_depth | 10–20 o None |
| | min_samples_leaf | 1–10 |
| | max_features (fracción) | 0.3–0.8 |
| XGBoost | learning_rate | 0.01–0.1 |
| | n_estimators | hasta 3000, con early stopping de 50 rondas |
| | max_depth | 3–8 |
| | subsample / colsample_bytree | 0.6–1.0 / 0.5–1.0 |
| | min_child_weight / reg_lambda | 1–10 / 1–10 |
| AdaBoost.R2 | estimator max_depth | 3–6 |
| | n_estimators | 25–150 |
| | learning_rate | 0.01–0.3 |
| | loss | linear |
| RNN / LSTM | hidden_size | 32–128 |
| | num_layers | 1–3 |
| | dropout | 0–0.3 |
| | lr | 3e-4–1e-3 |
| | batch_size | 64–512 |
| | épocas / patience | 30–100 / 8–10 |

Elementos fijos de las redes:
- Optimizador Adam o AdamW (weight decay de 1e-4) y pérdida MSE.
- Recorte del gradiente por norma a 1.0.
- `ReduceLROnPlateau(factor=0.5, patience=3)`.
- Cabeza densa de 32 neuronas con ReLU, seguida de la salida.
- Semilla 42; los árboles también usan `random_state=42` y `n_jobs=-1`.

**Configuraciones anteriores como candidatos iniciales**

Se evalúan siempre, aunque algún valor quede fuera de los rangos.

| Config. | Épocas | patience | lr | Optimizador | hidden | capas | dropout | batch |
|---|---|---|---|---|---|---|---|---|
| LSTM E1 | 12 | 3 | 1e-3 | Adam | 64 | 2 | 0.2 | 512 |
| LSTM E2 | 100 | 10 | 1e-3 | Adam | 32 | 3 | 0.2 | 32 |
| LSTM E3 | 100 | 10 | 1e-4 | AdamW + weight decay | 128 | 3 | 0.2 | 256 |

- Todas usan L = 24 h, MSE y recorte del gradiente a 1.0.
- La **RNN base** usa la configuración **LSTM E1** con celda RNN.
- En los árboles, E1–E3 eran idénticos, así que se incluye como candidato la antigua configuración "inicial":
  - RF: 300 árboles, `max_depth=None`, `min_samples_leaf=2` y `max_features=0.5`.
  - XGB: `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1` y `reg_lambda=1`, con 1000 rondas y early stopping de 50.
  - AdaBoost: profundidad 3, 200 estimadores, `learning_rate=0.1` y pérdida linear.

### 8. Selección del mejor modelo (Tesis §1.3, P56)
- Gana el modelo con el **mayor SS medio** entre los 5 folds y entre los dos horizontes del río.
- Si dos modelos difieren en **menos de 0.01** de SS, gana el de menor RMSE en niveles mayores al **P95 del train**.
- El modelo final se **reentrena con train + val** usando la mejor configuración.
- **Test se evalúa una sola vez**, al terminar el diseño.

### 9. Métricas (Tesis §1.3, P58; §2.2, P165–P211)
**Criterio principal**
- **SS = 1 − RMSE_modelo / RMSE_persistencia**, en el mismo horizonte y sobre el mismo conjunto. SS > 0 indica que el modelo aporta.
- La diferencia frente a la persistencia se prueba con **Diebold–Mariano** (varianza HAC) y **corrección de Holm**
  para hasta 8 comparaciones río-horizonte (H1, P70).

**Métricas complementarias**
- NSE, KGE (con r, α y β; Gupta et al., 2009), RMSE, MAE, sesgo medio, R² y **PBIAS** con la convención de Moriasi
  (positivo = subestimación).
- Los umbrales de Moriasi (NSE > 0.50, etc.) **no se usan como criterio** (P173).

**Crecidas y niveles altos**
- Eventos = episodios continuos en que el nivel observado, transformado a caudal, supera el umbral de prealerta.
- Por evento: error en el pico, % de picos subestimados y desfase del pico (h).
- Niveles mayores al **P95 del train**: MAE, RMSE y sesgo.

**Alertas**
- La comparación se hace **en caudal** (Tesis §1.3 P103; §2.2.9 P247). El nivel pronosticado y el observado se
  convierten con la curva de descarga **Q = A(H − H₀)^p** (Tabla 2) y se comparan con los caudales de alerta de ETAPA EP.
  Los umbrales se fijan antes de la prueba.
- Tres estados (regla continua de P102):
  - Normal: Q < Q_pre.
  - Prealerta: Q_pre ≤ Q < Q_al.
  - Alerta: Q ≥ Q_al.
- Como la curva es creciente, comparar en caudal equivale a comparar el nivel con su equivalente. La columna en cm se
  incluye solo como referencia.

| Río | A | H₀ | p | Prealerta / alerta (m³/s) | Equivalente en nivel (cm, referencia) |
|---|---|---|---|---|---|
| Tomebamba | 0.00998477 | 84 | 1.59009 | 30 / 50 | ≈ 238 / 296 |
| Yanuncay | 1.45966e-6 | 20 | 3.3435 | 33 / 50 | ≈ 178 / 199 |
| Tarqui | 7.63569e-6 | 26 | 3.01851 | 16 / 30 | ≈ 150 / 179 |
| Machángara | 0.000878136 | 20 | 2.39401 | 20 / 50 | ≈ 86 / 117 |

- Matriz de confusión **3×3**, con análisis de errores de uno y dos niveles, y matriz **2×2** de alerta frente a no alerta.
- Por clase: POD, precisión y F1. Además, **F1 macro**, **FAR**, **POFD**, eventos omitidos y anticipación efectiva (h).
- Un evento es un cruce de umbral; los cruces separados por **menos de 6 h** se agrupan en un mismo evento.
- Un acierto es una alerta emitida antes del cruce, con anticipación ≤ τ, y sostenida hasta el cruce (P53, P197).
- El **estado emitido** es el más severo de los dos horizontes.
- La evaluación es una **simulación secuencial** sobre el test.

**Diagnóstico e interpretabilidad**
- **Brecha de generalización** train/val/test, comparada con la de la persistencia (P142).
- **Interpretabilidad:** importancia por permutación (calculada en val) y **SHAP** (P49).

Detalle de cálculo en `Docs/Metricas_y_Graficas_Evaluacion.docx`.

### 10. Gráficas
- **Curva de sobreajuste** por modelo:
  - Redes: loss de train y val por época, con la época de early stopping y el lr.
  - XGBoost: `evals_result()` con `best_iteration`.
  - AdaBoost: `staged_predict()`.
  - RF: curva de validación.
- Real vs predicho en el tiempo, con acercamiento a los mayores eventos y las líneas de umbral.
- Dispersión observado vs predicho con línea 1:1 (`PredictionErrorDisplay`).
- Residuos vs predicho e histograma de residuos.
- Error por rango de nivel, incluido el tramo mayor al P95.
- Matriz de confusión 3×3 de alertas en test.
- Importancia por permutación y resumen SHAP.
- SS por fold y modelo, y métrica vs horizonte (τ₁, τ₂ y τ/Tr).
- Una vez por río: serie `y_obs` coloreada por partición, con los márgenes marcados.

### 11. Valores por río
Se completan después de la estimación de Tr y del preprocesamiento.

**Notebooks por río.** Los tres notebooks de cada río (preprocesamiento, modelos y métricas) se generan desde una
plantilla común con los parámetros de `config/rios.json` (estación y columna objetivo, curva, umbrales, subcuencas,
prefijo `<RIO>_` de las variables de entorno y sufijo `_<rio>` de todas las salidas):

```powershell
python src/generadores/generar_notebooks.py --rio <rio>     # o --todos; --listar muestra los ríos
python src/models/unir_partes.py --rio <rio> PARTE1 PARTE2 PARTE3
```

Las variables derivadas salen de `src/data_prep/variables_rio.py` (genérico; `variables_yanuncay.py` lo reexporta
para la aplicación). Detalle de comandos y variables de entorno en `Docs/guia_linea_base.md`.

**Yanuncay (Andrés)**

| Parámetro | Valor |
|---|---|
| Tr (mediana y RIC, train) | 7,4 h (5,3–9,9 h), 97 de 106 eventos de prealerta; tormentas separadas por ≥ 6 h secas |
| τ₁ / τ₂ (h) | 4 / 7 |
| Margen entre particiones (h) | 79 (72 + 7) |
| Cortes train / val / test | train hasta 2022-12-13 17:00 · val 2022-12-17 01:00 a 2024-10-23 00:00 · test 2024-10-26 08:00 a 2026-09-02 08:00 (UTC) |
| Eventos de prealerta train / val / test | 106 / 15 / 16 (cortes sin desplazar) |
| Filas válidas train / val / test | 58,9 % / 97,1 % / 65,3 % |
| Estaciones de entrada conservadas | Nivel Yanuncay-AJ-Tarqui; meteorología Cebollar y Sayausí; lluvia Cebollar, Sayausí, Narancay y Huizhil (regla de cuenca). Excluidas por costo: nivel Tomebamba-DJ-Mazán y Pucán |
| N.º de variables de entrada | 49 tabulares en la corrida preliminar; 85 con la lluvia extendida de la fase A |
| Umbrales prealerta / alerta | 33 / 50 m³/s (equivalen a 178,33 / 199,28 cm) |

**Machángara (Juan), Tarqui (Ruth) y Tomebamba (César)**

| Río | Tr | τ₁ / τ₂ | Margen | Cortes | Estaciones | N.º variables |
|---|---|---|---|---|---|---|
| Machángara | 6,8 h (4,1–10,6 h), 348 de 369 eventos | 3 / 6 | 78 h | train hasta 2022-12-13 17:00 · val 2022-12-17 00:00 a 2024-10-23 00:00 · test 2024-10-26 07:00 a 2026-09-02 08:00; eventos 369 / 64 / 91 | Nivel Llantera; meteorología Cebollar, Tixán, Sayausí y Saucay (RS); lluvia Cebollar, Tixán, Ricaurte, Saucay y Sayausí. Excluidas por costo: nivel Tomebamba-DJ-Mazán, Chanlud y El Labrado | 106 |
| Tarqui | — | — | — | — | Bloqueado: `LIM_Tarqui-AJ-Yanuncay` no está en la tabla maestra (el notebook se detiene en el Paso 1) | — |
| Tomebamba | 5,8 h (4,0–8,6 h), 93 de 95 eventos | 3 / 5 | 77 h | train hasta 2022-12-13 17:00 · val 2022-12-16 23:00 a 2024-10-23 00:00 · test 2024-10-26 06:00 a 2026-09-02 07:00; eventos 95 / 14 / 14 | Nivel DJ-Mazán y Llantera; meteorología Cebollar, Sayausí, Tixán y Saucay (RS); lluvia Ricaurte, Cebollar, Tixán, Saucay y Sayausí. Excluidas por costo: Huizhil, Chanlud, El Labrado, nivel Yanuncay y Pucán | 115 |

Valores de los preprocesamientos de prueba del 2026-10-27 (cortes sin desplazar en ambos ríos). En el Tomebamba el
estado de alerta (Q ≥ 50 m³/s) solo tiene 19 h en train, 1 h en val y 0 h en test: las métricas de esa clase no
son estimables en test.

### 12. Fuentes
**Documentos del repositorio**
- `Docs/TesisCardenasPlazaSanchezRivadeneira.docx`
- `Docs/Informe partición 70-15-15.docx`
- `Docs/Diagrama Red Neuronal.docx`
- `Docs/Metricas_y_Graficas_Evaluacion.docx`
- `Docs/VariableObjetivo.docx`

**Métodos citados en la tesis**
- Breiman (2001), Random Forests. *Machine Learning*, 45, 5–32.
- Drucker (1997), Improving regressors using boosting techniques (AdaBoost.R2). *ICML*.
- Chen y Guestrin (2016), XGBoost: A scalable tree boosting system. *KDD*.
- Hochreiter y Schmidhuber (1997), Long short-term memory. *Neural Computation*, 9(8).
- Bergstra y Bengio (2012), Random search for hyper-parameter optimization. *JMLR*, 13.
- Cerqueira et al. (2020), Evaluating time series forecasting models. *Machine Learning*, 109.
- Stekhoven y Bühlmann (2012), MissForest. *Bioinformatics*, 28(1).
- Teegavarapu y Chandramouli (2005), Improved weighting methods for estimating missing precipitation records. *Journal of Hydrology*, 312.
- Fritsch y Carlson (1980), Monotone piecewise cubic interpolation. *SIAM J. Numer. Anal.*, 17(2).
- Gupta et al. (2009), KGE y descomposición del MSE. *Journal of Hydrology*, 377, 80–91: https://www.semanticscholar.org/paper/3bed3d2ae40f4b771a3b3da8498803031a49637e
- Moriasi et al. (2007), solo para la convención de signo del PBIAS: https://swat.tamu.edu/media/90109/moriasimodeleval.pdf

**Documentación técnica**
- NeuralHydrology, métricas de pico: https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html
- WMO/WWRP, verificación de pronósticos (POD, FAR, POFD): https://wmo.int/wwrpwgne-joint-working-group-forecast-verification-research
- scikit-learn, curvas de validación: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ValidationCurveDisplay.html
- scikit-learn, `PredictionErrorDisplay`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html
- scikit-learn, importancia por permutación frente a impureza: https://scikit-learn.org/stable/modules/permutation_importance.html
- XGBoost, early stopping y `evals_result`: https://xgboost.readthedocs.io/en/stable/python/python_intro.html
