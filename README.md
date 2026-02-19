# Prueba Técnica - Científico de Datos

## Descripción

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para predecir la probabilidad de que un deudor realice el pago del mes, utilizando como variable objetivo `pago`. El proyecto incluye análisis exploratorio de datos, consultas SQL y el desarrollo de modelos predictivos.

## Estructura del Proyecto

```
Prueba_DS/
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias
├── run.py                     # Script principal
├── data/                      # Datos (no incluido en el repositorio)
│   └── PruebaDS.xlsx          # Dataset original
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb           # Análisis exploratorio
│   └── 02_model.ipynb         # Modelado
├── src/                       # Código fuente
│   ├── __init__.py
│   ├── data.py                # Carga y validación de datos
│   ├── eda.py                 # Funciones de análisis exploratorio
│   ├── preprocessing.py       # Preprocesamiento de datos
│   ├── modeling.py            # Modelado y evaluación
│   └── utils.py               # Utilidades
├── sql/                       # Consultas SQL
│   ├── 01_top10_clientes_saldo_capital.sql
│   └── 02_promedio_pago_por_departamento.sql
└── outputs/                   # Resultados
    ├── figures/               # Gráficos
    ├── metrics.json           # Métricas del modelo
    └── model.pkl              # Modelo entrenado
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

2. Explorar los notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

## Resultados

Los resultados del análisis y modelado se guardan en la carpeta `outputs/`:
- `figures/`: Gráficos generados durante el EDA y evaluación del modelo
- `metrics.json`: Métricas de rendimiento del modelo
- `model.pkl`: Modelo entrenado (si se guarda)

## Supuestos y Consideraciones

1. **Datos**:
   - Se asume que el archivo `PruebaDS.xlsx` contiene al menos una columna llamada `pago` como variable objetivo.
   - Se espera que las columnas numéricas estén en formato numérico o puedan ser convertidas.

2. **Modelado**:
   - Se utiliza un enfoque de clasificación binaria.
   - Se aplica validación cruzada estratificada para evaluar el rendimiento.
   - Se incluye un modelo lineal (Regresión Logística) y un modelo no lineal (Gradient Boosting) para comparación.

3. **Rendimiento**:
   - Se prioriza el AUC-ROC como métrica principal para la evaluación del modelo.
   - Se incluyen métricas adicionales como precisión, recall, F1 y matriz de confusión.

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
