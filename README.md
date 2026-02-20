# Prueba Técnica - Científico de Datos

**Candidata:** Dafne Valeria Castellanos Rosas

## Descripción

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para predecir la probabilidad de que un deudor realice el pago del mes, utilizando como variable objetivo `pago`. El proyecto incluye:

- **Análisis Exploratorio de Datos (EDA)** univariado y bivariado
- **Preprocesamiento** con flags de imputación, escalado y codificación
- **Selección de variables** mediante métodos combinados, importancia y RFE
- **Modelado predictivo** con 4 algoritmos y 3 técnicas de balanceo
- **Consultas SQL** para análisis de cartera

## Estructura del Proyecto

```
Prueba_DS/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── run.py                             # Script principal de ejecución
├── data/
│   └── PruebaDS.xlsx                  # Dataset original (29,613 registros × 18 columnas)
├── notebooks/
│   ├── 01_analisis_completo.ipynb     # Notebook principal: EDA + Preprocesamiento + Modelado
│   └── outputs/
│       └── data_quality_report.json   # Reporte de calidad de datos
├── src/                               # Módulos de código fuente
│   ├── __init__.py
│   ├── data.py                        # Carga y validación de datos
│   ├── eda.py                         # Funciones de análisis exploratorio
│   ├── preprocessing.py               # Preprocesamiento de datos
│   ├── modeling.py                    # Modelado base y evaluación
│   ├── feature_selection.py           # Selección de variables
│   └── balanced_modeling.py           # Modelado con técnicas de balanceo
├── sql/                               # Consultas SQL
│   ├── top_10_clientes_mayor_saldo_capital.sql
│   └── promedio_clientes_con_pago_por_departamento.sql
└── outputs/                           # Resultados generados
    ├── figures/                        # Gráficos del EDA y evaluación
    │   ├── missing_values.png
    │   ├── target_distribution.png
    │   ├── numerical_distributions.png
    │   ├── categorical_distributions.png
    │   ├── correlation_matrix.png
    │   └── outliers_boxplot.png
    └── preprocessor.joblib            # Preprocesador entrenado
```

## Instalación

1. Clonar el repositorio
2. Crear un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Colocar el archivo `PruebaDS.xlsx` en la carpeta `data/`

## Uso

1. Ejecutar el flujo completo:
   ```bash
   python run.py
   ```

2. Explorar el notebook principal:
   ```bash
   jupyter notebook notebooks/01_analisis_completo.ipynb
   ```

## Flujo del Análisis

El notebook `01_analisis_completo.ipynb` implementa el siguiente flujo:

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | Introducción | Configuración del entorno y carga de módulos |
| 2 | Carga y Exploración Inicial | Lectura del dataset, tipos de datos, estadísticas descriptivas |
| 3 | EDA | Análisis univariado, bivariado y distribución temporal del target |
| 4 | Preprocesamiento | Recategorización, duplicados, imputación con flags, escalado, encoding |
| 5 | Modelos Base | Entrenamiento de 4 modelos sin balanceo |
| 6 | Feature Selection | Selección de variables por 3 métodos (Combinado, Importancia, RFE) |
| 7 | Modelos con Variables Seleccionadas | Re-entrenamiento con subconjuntos óptimos |
| 8 | Manejo de Desbalanceo | Undersampling, Oversampling y SMOTE |
| 9 | Selección del Mejor Modelo | Comparación final y justificación |
| 10 | Importancia de Características | Análisis y justificación de SHAP Values |
| 11 | Conclusiones y Próximos Pasos | Hallazgos, recomendaciones y archivos generados |
| 12 | Sentencias SQL | Consultas SQL con resultados |

## Resultados Principales

- **Mejor modelo:** Gradient Boosting con Oversampling
  - **ROC AUC (test):** ~0.88
  - **PR AUC (test):** ~0.31
  - **Recall (test):** ~0.72
- **Drivers principales de pago:** Variables de contacto y gestión reciente (`duracion_llamadas_ultimos_6meses`, `contacto_ultimos_6meses`, `contacto_mes_actual`)
- **Desbalance de clases:** ~98% no pago vs ~2% pago — manejado con técnicas de oversampling

## Sentencias SQL

### Consulta 1: Top 10 clientes con mayor saldo capital

**Archivo:** `sql/top_10_clientes_mayor_saldo_capital.sql`

```sql
-- ============================================================================
-- Archivo:      top_10_clientes_mayor_saldo_capital.sql
-- Descripción:  Obtiene los 10 clientes con mayor saldo capital.
-- Fuente:       Tabla `clientes` (originada de data/PruebaDS.xlsx)
-- Autor:        Dafne Valeria Castellanos Rosas
-- Fecha:        2026-02-20
-- ============================================================================
--
-- CONTEXTO:
--   Esta consulta identifica a los 10 clientes que poseen el mayor saldo
--   de capital vigente. Es útil para priorizar estrategias de cobranza o
--   gestión de cartera sobre los clientes con mayor exposición financiera.
--
-- LÓGICA:
--   1. Se agrupa por cliente (identificacion) para consolidar todos sus
--      registros mensuales y obtener el saldo capital máximo histórico.
--   2. Se ordena de forma descendente por dicho saldo máximo.
--   3. Se limita el resultado a los primeros 10 registros.
--
-- NOTAS:
--   - Se utiliza MAX(saldo_capital) porque un mismo cliente puede tener
--     múltiples filas correspondientes a distintos meses de reporte.
--   - Se incluyen columnas descriptivas (tipo_documento, genero,
--     departamento, banco) tomando el valor más reciente (MAX) para dar
--     contexto al resultado.
--   - La cláusula LIMIT 10 es estándar en PostgreSQL/MySQL/SQLite.
--     En SQL Server se debe reemplazar por SELECT TOP 10.
-- ============================================================================

SELECT
    c.identificacion,
    c.tipo_documento,
    c.genero,
    c.departamento,
    c.banco,
    MAX(c.saldo_capital)    AS max_saldo_capital,
    MAX(c.dias_mora)        AS max_dias_mora
FROM
    clientes AS c
GROUP BY
    c.identificacion,
    c.tipo_documento,
    c.genero,
    c.departamento,
    c.banco
ORDER BY
    max_saldo_capital DESC
LIMIT 10;
```

### Consulta 2: Promedio de clientes con pago por departamento

**Archivo:** `sql/promedio_clientes_con_pago_por_departamento.sql`

```sql
-- ============================================================================
-- Archivo:      promedio_clientes_con_pago_por_departamento.sql
-- Descripción:  Calcula el promedio de clientes que realizaron un pago,
--               agrupado por departamento.
-- Fuente:       Tabla `clientes` (originada de data/PruebaDS.xlsx)
-- Autor:        Dafne Valeria Castellanos Rosas
-- Fecha:        2026-02-20
-- ============================================================================
--
-- CONTEXTO:
--   Esta consulta permite conocer, para cada departamento, cuántos clientes
--   en promedio realizan un pago por mes. Es un indicador clave para evaluar
--   la efectividad de la gestión de cobranza por región geográfica.
--
-- LÓGICA:
--   1. Subconsulta (pagos_por_mes_depto): Para cada combinación de mes y
--      departamento, se cuenta el número de clientes distintos que
--      realizaron un pago (pago = 1).
--   2. Consulta externa: Se calcula el promedio (AVG) de esos conteos
--      mensuales por departamento, redondeado a 2 decimales.
--   3. Se ordena de mayor a menor promedio para facilitar el análisis.
--
-- NOTAS:
--   - Se usa COUNT(DISTINCT identificacion) para evitar contar duplicados
--     de un mismo cliente dentro del mismo mes.
--   - La columna `pago` es un indicador binario (0/1) donde 1 significa
--     que el cliente realizó un pago en ese periodo.
--   - El promedio se calcula sobre los meses disponibles en los datos,
--     lo que permite comparar departamentos de forma normalizada.
-- ============================================================================

SELECT
    pagos.departamento,
    ROUND(AVG(pagos.total_clientes_con_pago), 2)  AS promedio_clientes_con_pago,
    COUNT(pagos.mes)                                AS meses_evaluados
FROM (
    -- Subconsulta: cuenta clientes únicos con pago por mes y departamento
    SELECT
        c.mes,
        c.departamento,
        COUNT(DISTINCT c.identificacion) AS total_clientes_con_pago
    FROM
        clientes AS c
    WHERE
        c.pago = 1
    GROUP BY
        c.mes,
        c.departamento
) AS pagos
GROUP BY
    pagos.departamento
ORDER BY
    promedio_clientes_con_pago DESC;
```

## Supuestos y Consideraciones

1. **Datos**:
   - El archivo `PruebaDS.xlsx` contiene 29,613 registros y 18 columnas, con `pago` como variable objetivo binaria.
   - Un mismo cliente puede tener múltiples registros correspondientes a distintos meses de reporte.
   - Variables con alto porcentaje de valores faltantes (`meses_desde_ultimo_pago`: 98.39%, `antiguedad_deuda`: 71.24%) fueron tratadas con flags de imputación.

2. **Preprocesamiento**:
   - Variables numéricas continuas imputadas con la mediana y escaladas con StandardScaler.
   - Variables categóricas imputadas con valor constante ("missing") y codificadas con OneHotEncoding.
   - Variables binarias imputadas con 0 y acompañadas de flags de imputación.
   - Se excluyeron `identificacion` y `mes` como features predictivas.

3. **Modelado**:
   - Se utiliza un enfoque de clasificación binaria con split estratificado 80/20.
   - Se evaluaron 4 algoritmos: Regresión Logística, Random Forest, Gradient Boosting y XGBoost.
   - Se aplicaron 3 técnicas de balanceo: Undersampling, Oversampling y SMOTE.
   - Se priorizan **ROC AUC** y **PR AUC** como métricas principales dada la naturaleza desbalanceada del problema.

### Diccionario de Datos

| Campo | Descripción | Ejemplo de Valor |
|-------|-------------|-------------------|
| mes | Mes de captura de la información | 2025-04 |
| tipo_documento | Tipo de ID del cliente | C |
| identificacion | Número de ID del cliente | 365960 |
| genero | Género del Cliente | HOMBRE |
| rango_edad_probable | Rango de edad al que pertenece el cliente | 51-55 |
| departamento | Departamento del cliente | CALDAS |
| saldo_capital | Saldo a capital del cliente | 4,760,221 |
| dias_mora | Días en mora del cliente | 2630 |
| banco | Banco con el cual tiene el crédito | colpatria |
| antiguedad_deuda | Fecha en la que se creó el crédito | 2017-10-21 00:00:00 |
| pago_mes_anterior | 1 Si el cliente pagó el mes anterior, 0 si no | 0 |
| meses_desde_ultimo_pago | Número de meses desde el último pago del cliente | 2 |
| sin_pago_previo | 1 si no tiene pago previo, 0 si sí tiene | 1 |
| contacto_mes_actual | 1 si tuvo contacto en el mes, 0 si no | 0 |
| contacto_mes_anterior | 1 si tuvo contacto el mes anterior, 0 si no | 0 |
| contacto_ultimos_6meses | 1 si tuvo contacto en los últimos 6 meses, 0 si no | 0 |
| duracion_llamadas_ultimos_6meses | Segundos de duración de las llamadas en los últimos 6 meses | 0 |
| pago | Variable objetivo: 1 si realizó pago en el mes, 0 si no | 0 |

## Licencia

Este proyecto es para fines de evaluación técnica.
