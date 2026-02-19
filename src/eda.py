"""
Módulo para el análisis exploratorio de datos (EDA).

Este módulo contiene funciones para visualizar y analizar los datos,
generando gráficos y estadísticas descriptivas.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Configuración de estilos
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
rcParams['figure.figsize'] = (12, 6)
rcParams['font.size'] = 12

# Configuración de logging
logger = logging.getLogger(__name__)

def setup_plot_style() -> None:
    """Configura el estilo de los gráficos."""
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid")
    rcParams['figure.figsize'] = (12, 6)
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.facecolor'] = 'white'

def save_plot(
    fig,
    filename: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    **kwargs
) -> None:
    """
    Guarda una figura en el directorio especificado.
    
    Args:
        fig: Figura de matplotlib a guardar
        filename: Nombre del archivo o ruta completa
        dpi: Resolución en puntos por pulgada
        bbox_inches: Opción para ajustar los márgenes
        **kwargs: Argumentos adicionales para savefig
    """
    # Asegurarse de que el directorio existe
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar la figura
    fig.savefig(
        output_path, 
        dpi=dpi, 
        bbox_inches=bbox_inches,
        **kwargs
    )
    plt.close(fig)
    logger.info(f"Gráfico guardado en: {output_path}")

def plot_missing_values(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Genera un gráfico de barras mostrando el porcentaje de valores faltantes por columna.
    
    Args:
        df: DataFrame con los datos
        top_n: Número de columnas a mostrar (las que tienen más valores faltantes)
        
    Returns:
        Figura de matplotlib con el gráfico
    """
    # Calcular porcentaje de valores faltantes
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]  # Solo columnas con valores faltantes
    
    # Limitar al top N
    if len(missing) > top_n:
        missing = missing.head(top_n)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Gráfico de barras
    bars = sns.barplot(x=missing.values, y=missing.index, ax=ax, palette="viridis")
    
    # Añadir etiquetas
    ax.set_title(f'Porcentaje de valores faltantes por columna (Top {len(missing)})')
    ax.set_xlabel('Porcentaje de valores faltantes (%)')
    ax.set_ylabel('Columnas')
    
    # Añadir anotaciones
    for i, v in enumerate(missing):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    return fig

def plot_target_distribution(y: pd.Series) -> plt.Figure:
    """
    Genera un gráfico de distribución de la variable objetivo.
    
    Args:
        y: Serie con la variable objetivo
        
    Returns:
        Figura de matplotlib con el gráfico
    """
    # Contar valores
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras
    bars = sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax1, palette="Blues_d")
    ax1.set_title('Distribución de la variable objetivo')
    ax1.set_xlabel('Clase')
    ax1.set_ylabel('Conteo')
    
    # Añadir etiquetas con los valores
    for i, v in enumerate(counts):
        ax1.text(i, v + 5, str(v), ha='center')
    
    # Gráfico de torta
    ax2.pie(
        counts, 
        labels=[f'{idx} ({pct:.1f}%)' for idx, pct in zip(counts.index, percentages)],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("Blues_d", len(counts)),
        wedgeprops=dict(width=0.6, edgecolor='w')
    )
    ax2.set_title('Distribución porcentual')
    
    plt.tight_layout()
    return fig

def plot_numerical_distributions(
    df: pd.DataFrame,
    target_col: str = None,
    exempt_fields: List[str] = None,
    n_cols: int = 3,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Genera gráficos de distribución para las variables numéricas.
    
    Args:
        df: DataFrame con los datos
        target_col: Columna objetivo para estratificar los gráficos (opcional)
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)
        n_cols: Número de columnas en la cuadrícula de gráficos
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con los gráficos
    """
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    # Seleccionar solo columnas numéricas (excluyendo campos exentos)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exempt_fields]
    
    # Si hay una columna objetivo, la excluimos
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if not numeric_cols:
        logger.warning("No se encontraron columnas numéricas para graficar")
        return None
    
    # Calcular el número de filas necesarias
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar el array de ejes para facilitar la iteración
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # Generar un gráfico por cada columna numérica
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        
        if target_col is not None and target_col in df.columns:
            # Si hay una columna objetivo, estratificar por ella
            for target_val in sorted(df[target_col].unique()):
                sns.kdeplot(
                    df[df[target_col] == target_val][col], 
                    label=f'{target_col}={target_val}',
                    ax=ax,
                    alpha=0.6
                )
            ax.legend()
        else:
            # Si no hay columna objetivo, solo mostrar la distribución
            sns.histplot(df[col], kde=True, ax=ax)
        
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_categorical_distributions(
    df: pd.DataFrame,
    target_col: str = None,
    max_categories: int = 10,
    n_cols: int = 2,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Genera gráficos de barras para las variables categóricas.
    
    Args:
        df: DataFrame con los datos
        target_col: Columna objetivo para estratificar los gráficos (opcional)
        max_categories: Número máximo de categorías a mostrar por variable
        n_cols: Número de columnas en la cuadrícula de gráficos
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con los gráficos
    """
    # Seleccionar columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Si hay una columna objetivo, la excluimos
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    
    if not cat_cols:
        logger.warning("No se encontraron columnas categóricas para graficar")
        return None
    
    # Calcular el número de filas necesarias
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar el array de ejes para facilitar la iteración
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # Generar un gráfico por cada columna categórica
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        
        # Contar valores y limitar al máximo de categorías
        value_counts = df[col].value_counts().head(max_categories)
        
        # Crear gráfico de barras
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette="viridis")
        
        # Añadir etiquetas
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('Conteo')
        ax.set_ylabel('')
        
        # Añadir etiquetas con los valores
        for j, v in enumerate(value_counts):
            ax.text(v + 0.1, j, str(v), va='center')
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(
    df: pd.DataFrame,
    exempt_fields: List[str] = None,
    method: str = 'pearson',
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Genera una matriz de correlación para las variables numéricas.
    
    Args:
        df: DataFrame con los datos
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)
        method: Método de correlación ('pearson', 'spearman', 'kendall')
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con la matriz de correlación
    """
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    # Seleccionar solo columnas numéricas (excluyendo campos exentos)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exempt_fields]
    
    if len(numeric_cols) < 2:
        logger.warning("Se necesitan al menos dos columnas numéricas para la matriz de correlación")
        return None
    
    # Calcular matriz de correlación
    corr = df[numeric_cols].corr(method=method)
    
    # Crear máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generar mapa de calor
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, 
        mask=mask,
        cmap=cmap, 
        vmax=1.0, 
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    
    ax.set_title(f'Matriz de correlación ({method.capitalize()})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def detect_outliers(
    df: pd.DataFrame,
    exempt_fields: List[str] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Dict[str, List[int]]:
    """
    Detecta valores atípicos en las columnas numéricas.
    
    Args:
        df: DataFrame con los datos
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)
        method: Método para detectar outliers ('iqr' o 'zscore')
        threshold: Umbral para la detección
        
    Returns:
        Diccionario con los índices de los outliers por columna
    """
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    outliers = {}
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exempt_fields]
    
    for col in numeric_cols:
        if method.lower() == 'iqr':
            # Método del rango intercuartílico (IQR)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Encontrar outliers
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method.lower() == 'zscore':
            # Método de la puntuación Z
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores > threshold
        
        else:
            raise ValueError("Método no soportado. Use 'iqr' o 'zscore'.")
        
        # Obtener índices de los outliers
        outlier_indices = df.index[mask].tolist()
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers

def generate_eda_report(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    target_col: str = 'pago',
    exempt_fields: List[str] = None
) -> Dict[str, Any]:
    """
    Genera un informe completo de análisis exploratorio de datos.
    
    Args:
        df: DataFrame con los datos
        output_dir: Directorio donde guardar los gráficos
        target_col: Nombre de la columna objetivo
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)
        
    Returns:
        Diccionario con estadísticas y rutas a los gráficos generados
    """
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    # Configurar directorio de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo de los gráficos
    setup_plot_style()
    
    # Diccionario para almacenar resultados
    report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numerical_columns': [col for col in df.select_dtypes(include=['number']).columns.tolist() if col not in exempt_fields],
        'categorical_columns': df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
        'plots': {}
    }
    
    try:
        # 1. Gráfico de valores faltantes
        fig_missing = plot_missing_values(df)
        if fig_missing:
            missing_path = output_dir / 'missing_values.png'
            save_plot(fig_missing, missing_path)
            report['plots']['missing_values'] = str(missing_path)
        
        # 2. Distribución de la variable objetivo (si existe)
        if target_col in df.columns:
            fig_target = plot_target_distribution(df[target_col])
            target_path = output_dir / 'target_distribution.png'
            save_plot(fig_target, target_path)
            report['plots']['target_distribution'] = str(target_path)
        
        # 3. Distribuciones de variables numéricas
        fig_num = plot_numerical_distributions(df, target_col=target_col if target_col in df.columns else None, exempt_fields=exempt_fields)
        if fig_num:
            num_path = output_dir / 'numerical_distributions.png'
            save_plot(fig_num, num_path)
            report['plots']['numerical_distributions'] = str(num_path)
        
        # 4. Distribuciones de variables categóricas
        fig_cat = plot_categorical_distributions(df, target_col=target_col if target_col in df.columns else None)
        if fig_cat:
            cat_path = output_dir / 'categorical_distributions.png'
            save_plot(fig_cat, cat_path)
            report['plots']['categorical_distributions'] = str(cat_path)
        
        # 5. Matriz de correlación (solo si hay suficientes columnas numéricas)
        if len(report['numerical_columns']) >= 2:
            fig_corr = plot_correlation_matrix(df, exempt_fields=exempt_fields)
            if fig_corr:
                corr_path = output_dir / 'correlation_matrix.png'
                save_plot(fig_corr, corr_path)
                report['plots']['correlation_matrix'] = str(corr_path)
        
        # 6. Detección de outliers
        outliers = detect_outliers(df, exempt_fields=exempt_fields)
        report['outliers'] = {col: len(indices) for col, indices in outliers.items()}
        
        # Guardar reporte completo
        report_path = output_dir / 'eda_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Informe EDA guardado en: {report_path}")
        return report
    
    except Exception as e:
        logger.error(f"Error al generar el informe EDA: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Ejemplo de uso
    import pandas as pd
    
    # Crear datos de ejemplo
    data = {
        'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'ingresos': [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 100000,
                     50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 100000],
        'pago': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'genero': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'categoria': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(data)
    
    # Generar informe EDA
    report = generate_eda_report(df, 'outputs/eda')
    print("Análisis EDA completado. Gráficos guardados en la carpeta 'outputs/eda'")
    print(f"Resumen del informe: {json.dumps(report, indent=2)}")
