"""
Módulo principal del proyecto de Prueba Técnica para Científico de Datos.

Este paquete contiene los módulos para el procesamiento de datos, análisis exploratorio,
preprocesamiento, modelado y utilidades para el proyecto de predicción de pagos.
"""

__version__ = "0.1.0"

# Importaciones principales
from .data import load_data, validate_dataframe
from .eda import (
    generate_summary_statistics,
    plot_target_distribution,
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    detect_outliers,
    save_plot
)
from .preprocessing import (
    preprocess_data,
    split_data,
    create_preprocessor
)
from .modeling import (
    train_models,
    evaluate_model,
    save_model,
    load_model
)
from .utils import (
    find_similar_columns,
    setup_logging,
    save_metrics
)

__all__ = [
    # data.py
    'load_data',
    'validate_dataframe',
    
    # eda.py
    'generate_summary_statistics',
    'plot_target_distribution',
    'plot_numerical_distributions',
    'plot_categorical_distributions',
    'plot_correlation_matrix',
    'detect_outliers',
    'save_plot',
    
    # preprocessing.py
    'preprocess_data',
    'split_data',
    'create_preprocessor',
    
    # modeling.py
    'train_models',
    'evaluate_model',
    'save_model',
    'load_model',
    
    # utils.py
    'find_similar_columns',
    'setup_logging',
    'save_metrics'
]
