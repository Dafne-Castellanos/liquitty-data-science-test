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
        
        # Verificar si es una variable binaria (solo 2 valores únicos)
        unique_vals = df[col].dropna().nunique()
        is_binary = unique_vals == 2
        
        if is_binary:
            # Para variables binarias, usar gráfico de barras
            value_counts = df[col].value_counts().sort_index()
            
            if target_col is not None and target_col in df.columns:
                # Estratificar por variable objetivo
                for target_val in sorted(df[target_col].unique()):
                    subset = df[df[target_col] == target_val]
                    subset_counts = subset[col].value_counts().sort_index()
                    x_pos = np.arange(len(subset_counts))
                    width = 0.8 / len(df[target_col].unique())
                    offset = list(df[target_col].unique()).index(target_val) * width
                    
                    bars = ax.bar(
                        x_pos + offset, 
                        subset_counts.values, 
                        width, 
                        label=f'{target_col}={target_val}',
                        alpha=0.8
                    )
                
                ax.set_xticks(x_pos + width / 2)
                ax.set_xticklabels([str(val) for val in subset_counts.index])
                ax.legend()
            else:
                # Sin estratificación
                sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax, palette="viridis")
            
            ax.set_title(f'Distribución de {col} (binaria)')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Conteo')
            
            # Añadir etiquetas con los valores
            if target_col is None:
                for j, v in enumerate(value_counts):
                    ax.text(j, v + max(value_counts) * 0.01, str(v), ha='center')
            
        else:
            # Para variables no binarias, usar gráficos de distribución
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

def generate_imputation_flags(
    df: pd.DataFrame,
    columns: List[str] = None,
    strategy: str = 'indicator',
    suffix: str = '_imputed'
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Genera flags de imputación para valores faltantes.
    
    Los flags de imputación son variables binarias que indican si un dato 
    fue originalmente missing y luego fue imputado.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas para generar flags (si es None, todas las columnas con missing)
        strategy: Estrategia para generar flags ('indicator' o 'missing_mask')
        suffix: Sufijo para los nombres de las nuevas columnas de flags
        
    Returns:
        Tuple con:
        - DataFrame con los flags agregados
        - Diccionario con información de los flags generados
        
    Example:
        >>> df_with_flags, flag_info = generate_imputation_flags(df, columns=['edad', 'ingresos'])
        >>> print(f"Flags generados: {list(flag_info.keys())}")
    """
    df_flags = df.copy()
    flag_info = {}
    
    # Si no se especifican columnas, usar todas las que tienen valores faltantes
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Columna '{col}' no encontrada en el DataFrame")
            continue
        
        # Verificar si hay valores faltantes
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            logger.info(f"Columna '{col}' no tiene valores faltantes, omitiendo flag")
            continue
        
        # Generar nombre para la columna de flag
        flag_col = f"{col}{suffix}"
        
        if strategy == 'indicator':
            # Estrategia 1: Variable indicadora binaria
            # 1 si el valor fue imputado (era missing), 0 si es original
            df_flags[flag_col] = df[col].isnull().astype(int)
            
        elif strategy == 'missing_mask':
            # Estrategia 2: Máscara de missing
            # True si fue imputado, False si es original
            df_flags[flag_col] = df[col].isnull()
            
        else:
            raise ValueError(f"Estrategia '{strategy}' no soportada. Use 'indicator' o 'missing_mask'")
        
        # Guardar información del flag
        flag_info[flag_col] = {
            'source_column': col,
            'missing_count': missing_count,
            'missing_rate': missing_count / len(df),
            'strategy': strategy,
            'flag_type': 'binary' if strategy == 'indicator' else 'boolean'
        }
        
        logger.info(f"Flag '{flag_col}' generado para columna '{col}' "
                   f"({missing_count} valores missing, {missing_count/len(df)*100:.1f}%)")
    
    return df_flags, flag_info

def impute_with_flags(
    df: pd.DataFrame,
    columns: List[str] = None,
    numeric_strategy: str = 'mean',
    categorical_strategy: str = 'mode',
    flag_suffix: str = '_imputed'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Realiza imputación de valores faltantes y genera flags de imputación.
    
    Esta función combina la imputación de datos con la creación de flags
    para registrar qué valores fueron imputados.
    
    Args:
        df: DataFrame con valores faltantes
        columns: Columnas a imputar (si es None, todas las columnas con missing)
        numeric_strategy: Estrategia para variables numéricas ('mean', 'median', 'mode', 'constant')
        categorical_strategy: Estrategia para categóricas ('mode', 'constant')
        flag_suffix: Sufijo para las columnas de flags
        
    Returns:
        Tuple con:
        - DataFrame imputado con flags
        - Diccionario con información de imputación y flags
        
    Example:
        >>> df_imputed, info = impute_with_flags(df, numeric_strategy='median')
        >>> print(f"Columnas imputadas: {info['imputed_columns']}")
        >>> print(f"Flags generados: {info['flag_columns']}")
    """
    df_imputed = df.copy()
    
    # Si no se especifican columnas, usar todas las que tienen valores faltantes
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    # Primero generar los flags antes de imputar
    df_imputed, flag_info = generate_imputation_flags(
        df_imputed, 
        columns=columns, 
        strategy='indicator',
        suffix=flag_suffix
    )
    
    imputation_info = {
        'imputed_columns': [],
        'imputation_methods': {},
        'flag_columns': list(flag_info.keys()),
        'flag_info': flag_info
    }
    
    for col in columns:
        if col not in df.columns:
            continue
            
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        
        # Determinar tipo de columna y estrategia
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if is_numeric:
            # Imputación para variables numéricas
            if numeric_strategy == 'mean':
                fill_value = df[col].mean()
            elif numeric_strategy == 'median':
                fill_value = df[col].median()
            elif numeric_strategy == 'mode':
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            elif numeric_strategy == 'constant':
                fill_value = 0  # Valor constante por defecto
            else:
                raise ValueError(f"Estrategia numérica '{numeric_strategy}' no soportada")
                
            method = f"numeric_{numeric_strategy}"
            
        else:
            # Imputación para variables categóricas
            if categorical_strategy == 'mode':
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            elif categorical_strategy == 'constant':
                fill_value = 'Unknown'
            else:
                raise ValueError(f"Estrategia categórica '{categorical_strategy}' no soportada")
                
            method = f"categorical_{categorical_strategy}"
        
        # Realizar la imputación
        df_imputed[col] = df_imputed[col].fillna(fill_value)
        
        # Guardar información
        imputation_info['imputed_columns'].append(col)
        imputation_info['imputation_methods'][col] = {
            'method': method,
            'fill_value': fill_value,
            'missing_count': missing_count,
            'missing_rate': missing_count / len(df)
        }
        
        logger.info(f"Columna '{col}' imputada usando {method} "
                   f"(valor: {fill_value}, {missing_count} registros)")
    
    return df_imputed, imputation_info

def plot_bivariate_numerical_vs_target(
    df: pd.DataFrame,
    target_col: str,
    numerical_cols: List[str] = None,
    exempt_fields: List[str] = None,
    n_cols: int = 2,
    figsize: tuple = (15, 12)
) -> plt.Figure:
    """
    Genera gráficos bivariados para variables numéricas vs variable objetivo.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        numerical_cols: Lista de columnas numéricas (si es None, se detectan automáticamente)
        exempt_fields: Lista de campos exentos de análisis
        n_cols: Número de columnas en la cuadrícula
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con los gráficos
    """
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    # Seleccionar columnas numéricas
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exempt_fields and col != target_col]
    
    if not numerical_cols:
        logger.warning("No se encontraron columnas numéricas para análisis bivariado")
        return None
    
    # Calcular el número de filas necesarias
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar el array de ejes
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # Generar gráfico para cada variable numérica
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        
        # Verificar si es binaria
        unique_vals = df[col].dropna().nunique()
        is_binary = unique_vals == 2
        
        if is_binary:
            # Para variables binarias: gráfico de barras apiladas
            crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            
            # Crear gráfico de barras apiladas
            crosstab.plot(
                kind='bar', 
                stacked=True, 
                ax=ax, 
                colormap='viridis',
                alpha=0.8
            )
            
            ax.set_title(f'{col} (binaria) vs {target_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Porcentaje (%)')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Añadir etiquetas de porcentaje
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f%%', fontsize=8)
        
        else:
            # Para variables numéricas continuas: boxplots y violin plots
            target_values = sorted(df[target_col].unique())
            
            # Crear boxplot
            box_plot_data = []
            labels = []
            
            for target_val in target_values:
                subset = df[df[target_col] == target_val][col].dropna()
                box_plot_data.append(subset)
                labels.append(f'{target_col}={target_val}')
            
            # Boxplot
            bp = ax.boxplot(box_plot_data, labels=labels, patch_artist=True)
            
            # Colores para cada clase
            colors = sns.color_palette("viridis", len(target_values))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Añadir puntos de datos individuales (stripplot)
            for j, (target_val, subset) in enumerate(zip(target_values, box_plot_data)):
                if len(subset) > 0:
                    # Añadir jitter para evitar sobreposición
                    jitter = np.random.normal(0, 0.05, size=len(subset))
                    ax.scatter(
                        np.ones(len(subset)) * (j + 1) + jitter, 
                        subset, 
                        alpha=0.3, 
                        s=20, 
                        color=colors[j]
                    )
            
            ax.set_title(f'{col} vs {target_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(target_col)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_bivariate_categorical_vs_target(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str] = None,
    max_categories: int = 10,
    n_cols: int = 2,
    figsize: tuple = (15, 12)
) -> plt.Figure:
    """
    Genera gráficos bivariados para variables categóricas vs variable objetivo.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        categorical_cols: Lista de columnas categóricas (si es None, se detectan automáticamente)
        max_categories: Número máximo de categorías a mostrar
        n_cols: Número de columnas en la cuadrícula
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con los gráficos
    """
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    # Seleccionar columnas categóricas
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != target_col]
    
    if not categorical_cols:
        logger.warning("No se encontraron columnas categóricas para análisis bivariado")
        return None
    
    # Calcular el número de filas necesarias
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar el array de ejes
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # Generar gráfico para cada variable categórica
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        
        # Verificar si es binaria
        unique_vals = df[col].dropna().nunique()
        is_binary = unique_vals == 2
        
        if is_binary:
            # Para variables binarias: gráfico de barras apiladas
            crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            
            # Crear gráfico de barras apiladas
            crosstab.plot(
                kind='bar', 
                stacked=True, 
                ax=ax, 
                colormap='viridis',
                alpha=0.8
            )
            
            ax.set_title(f'{col} (binaria) vs {target_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Porcentaje (%)')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Añadir etiquetas de porcentaje
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f%%', fontsize=8)
        
        else:
            # Para variables categóricas con múltiples categorías
            value_counts = df[col].value_counts().head(max_categories)
            
            # Crear tabla de contingencia normalizada
            crosstab = pd.crosstab(
                df[col], 
                df[target_col], 
                normalize='index'
            ) * 100
            
            # Filtrar solo las categorías más frecuentes
            crosstab = crosstab.loc[value_counts.index]
            
            # Crear gráfico de barras agrupadas
            crosstab.plot(
                kind='bar', 
                ax=ax, 
                colormap='viridis',
                alpha=0.8,
                width=0.8
            )
            
            ax.set_title(f'{col} vs {target_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Porcentaje (%)')
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Rotar etiquetas si hay muchas categorías
            if len(value_counts) > 5:
                ax.tick_params(axis='x', rotation=45)
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_bivariate_analysis(
    df: pd.DataFrame,
    target_col: str,
    exempt_fields: List[str] = None,
    max_categories: int = 10,
    n_cols: int = 2,
    figsize: tuple = (20, 15)
) -> Dict[str, plt.Figure]:
    """
    Genera análisis bivariado completo vs variable objetivo.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        exempt_fields: Lista de campos exentos de análisis
        max_categories: Número máximo de categorías a mostrar
        n_cols: Número de columnas en la cuadrícula
        figsize: Tamaño de la figura
        
    Returns:
        Diccionario con las figuras generadas
    """
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    figures = {}
    
    # 1. Análisis numérico vs target
    logger.info("Generando análisis numérico vs target...")
    fig_num = plot_bivariate_numerical_vs_target(
        df, 
        target_col=target_col, 
        exempt_fields=exempt_fields,
        n_cols=n_cols,
        figsize=figsize
    )
    if fig_num:
        figures['numerical_vs_target'] = fig_num
    
    # 2. Análisis categórico vs target
    logger.info("Generando análisis categórico vs target...")
    fig_cat = plot_bivariate_categorical_vs_target(
        df, 
        target_col=target_col, 
        max_categories=max_categories,
        n_cols=n_cols,
        figsize=figsize
    )
    if fig_cat:
        figures['categorical_vs_target'] = fig_cat
    
    logger.info(f"Análisis bivariado completado para target '{target_col}'")
    return figures

def calculate_bivariate_statistics(
    df: pd.DataFrame,
    target_col: str,
    exempt_fields: list = None
) -> dict:
    """
    Calcula estadísticas bivariadas robustas entre variables y target.
    
    Soporta:
    - Target continuo
    - Target binario (incluye AUC y mutual information)
    - Target categórico
    
    Returns:
        Diccionario con estadísticas bivariadas
    """
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    if exempt_fields is None:
        exempt_fields = []

    target = df[target_col]
    target_nunique = target.nunique(dropna=True)

    # Detectar tipo de target
    is_numeric_target = pd.api.types.is_numeric_dtype(target)
    is_binary_target = target_nunique == 2

    results = {
        'target_column': target_col,
        'target_type': str(target.dtype),
        'target_unique_values': target_nunique,
        'is_binary_target': is_binary_target,
        'numerical_analysis': {},
        'categorical_analysis': {}
    }

    # =========================
    # NUMERICAL VARIABLES
    # =========================
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in exempt_fields and c != target_col]

    for col in numerical_cols:
        col_stats = {
            'missing_rate': df[col].isnull().mean(),
            'is_binary': df[col].dropna().nunique() == 2
        }

        valid = df[[col, target_col]].dropna()

        if len(valid) > 10:
            try:
                # =========================
                # TARGET CONTINUO
                # =========================
                if is_numeric_target and not is_binary_target:
                    corr = valid[col].corr(valid[target_col])
                    col_stats['correlation'] = corr

                    try:
                        _, pval = scipy_stats.pearsonr(valid[col], valid[target_col])
                        col_stats['correlation_pvalue'] = pval
                    except:
                        col_stats['correlation_pvalue'] = None

                    # Mutual Information
                    try:
                        mi = mutual_info_regression(valid[[col]], valid[target_col])
                        col_stats['mutual_information'] = float(mi[0])
                    except:
                        col_stats['mutual_information'] = None

                # =========================
                # TARGET BINARIO
                # =========================
                elif is_binary_target:
                    # AUC (muy importante en crédito)
                    try:
                        if valid[col].nunique() > 1:
                            auc = roc_auc_score(valid[target_col], valid[col])
                            col_stats['auc'] = auc
                    except:
                        col_stats['auc'] = None

                    # Correlación (opcional)
                    try:
                        corr = valid[col].corr(valid[target_col])
                        col_stats['correlation'] = corr
                    except:
                        col_stats['correlation'] = None

                    # Mutual Information
                    try:
                        mi = mutual_info_classif(valid[[col]], valid[target_col])
                        col_stats['mutual_information'] = float(mi[0])
                    except:
                        col_stats['mutual_information'] = None

                    # Media por clase (muy útil)
                    try:
                        means = valid.groupby(target_col)[col].mean().to_dict()
                        col_stats['group_means'] = means
                    except:
                        pass

                # =========================
                # TARGET CATEGÓRICO
                # =========================
                else:
                    try:
                        means = valid.groupby(target_col)[col].mean().to_dict()
                        col_stats['group_means'] = means
                    except:
                        pass

                    # ANOVA
                    try:
                        groups = [g[col].values for _, g in valid.groupby(target_col) if len(g) > 0]
                        if len(groups) > 1:
                            f_stat, pval = scipy_stats.f_oneway(*groups)
                            col_stats['anova_f_stat'] = f_stat
                            col_stats['anova_p_value'] = pval
                    except:
                        pass

            except Exception:
                pass

        results['numerical_analysis'][col] = col_stats

    # =========================
    # CATEGORICAL VARIABLES
    # =========================
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exempt_fields and c != target_col]

    for col in categorical_cols:
        col_stats = {
            'missing_rate': df[col].isnull().mean(),
            'unique_categories': df[col].nunique(),
            'is_binary': df[col].dropna().nunique() == 2
        }

        valid = df[[col, target_col]].dropna()

        if len(valid) > 0:
            try:
                crosstab = pd.crosstab(valid[col], valid[target_col])
                col_stats['contingency_table'] = crosstab.to_dict()

                # Chi2 test
                try:
                    chi2, pval, dof, _ = scipy_stats.chi2_contingency(crosstab)
                    col_stats['chi2_statistic'] = chi2
                    col_stats['chi2_p_value'] = pval
                    col_stats['chi2_dof'] = dof
                except:
                    pass

                # Mutual Information
                try:
                    encoded = pd.factorize(valid[col])[0].reshape(-1, 1)
                    if is_binary_target:
                        mi = mutual_info_classif(encoded, valid[target_col])
                    else:
                        mi = mutual_info_regression(encoded, valid[target_col])
                    col_stats['mutual_information'] = float(mi[0])
                except:
                    col_stats['mutual_information'] = None

            except Exception:
                pass

        results['categorical_analysis'][col] = col_stats

    return results

def plot_payment_distribution_by_month(
    df: pd.DataFrame,
    payment_col: str = 'pago',
    month_col: str = 'mes',
    figsize: tuple = (14, 8),
    show_percentages: bool = True
) -> plt.Figure:
    """
    Genera un gráfico de la distribución de pagos (0 vs 1) por mes.
    
    Args:
        df: DataFrame con los datos
        payment_col: Nombre de la columna de pago (binaria 0/1)
        month_col: Nombre de la columna de mes
        figsize: Tamaño de la figura
        show_percentages: Si True, muestra porcentajes en las barras
        
    Returns:
        Figura de matplotlib con el gráfico
    """
    if payment_col not in df.columns:
        raise ValueError(f"Columna de pago '{payment_col}' no encontrada")
    
    if month_col not in df.columns:
        raise ValueError(f"Columna de mes '{month_col}' no encontrada")
    
    # Crear tabla de contingencia
    payment_by_month = pd.crosstab(df[month_col], df[payment_col])
    
    # Ordenar meses cronológicamente si es necesario
    if payment_by_month.index.dtype == 'object':
        # Intentar ordenar por nombre de mes
        month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        # Si los datos son numéricos (1-12), mapear a nombres
        if payment_by_month.index.astype(str).str.isnumeric().all():
            month_mapping = {str(i+1): month for i, month in enumerate(month_order)}
            payment_by_month.index = payment_by_month.index.astype(str).map(month_mapping)
        
        # Reordenar si es posible
        available_months = [month for month in month_order if month in payment_by_month.index]
        if available_months:
            payment_by_month = payment_by_month.reindex(available_months)
    
    # Calcular porcentajes
    payment_percentages = payment_by_month.div(payment_by_month.sum(axis=1), axis=0) * 100
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Gráfico 1: Conteos absolutos
    payment_by_month.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=['lightcoral', 'lightgreen'],
        alpha=0.8,
        edgecolor='black'
    )
    
    ax1.set_title('Distribución de Pagos por Mes (Conteos Absolutos)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mes')
    ax1.set_ylabel('Número de Registros')
    ax1.legend(title='Pago', labels=['No Pago (0)', 'Pago (1)'], bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Añadir etiquetas de valor en las barras
    if show_percentages:
        for i, (month, row) in enumerate(payment_by_month.iterrows()):
            height_no_payment = row[0] if 0 in row.index else 0
            height_payment = row[1] if 1 in row.index else 0
            
            # Etiqueta para No Pago
            if height_no_payment > 0:
                ax1.text(i, height_no_payment/2, str(height_no_payment), 
                        ha='center', va='center', fontweight='bold')
            
            # Etiqueta para Pago
            if height_payment > 0:
                ax1.text(i, height_no_payment + height_payment/2, str(height_payment), 
                        ha='center', va='center', fontweight='bold')
    
    # Gráfico 2: Porcentajes
    payment_percentages.plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        color=['lightcoral', 'lightgreen'],
        alpha=0.8,
        edgecolor='black'
    )
    
    ax2.set_title('Distribución de Pagos por Mes (Porcentajes)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mes')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.legend(title='Pago', labels=['No Pago (0)', 'Pago (1)'], bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Añadir etiquetas de porcentaje
    if show_percentages:
        for i, (month, row) in enumerate(payment_percentages.iterrows()):
            pct_no_payment = row[0] if 0 in row.index else 0
            pct_payment = row[1] if 1 in row.index else 0
            
            # Etiqueta para No Pago
            if pct_no_payment > 0:
                ax2.text(i, pct_no_payment/2, f'{pct_no_payment:.1f}%', 
                        ha='center', va='center', fontweight='bold')
            
            # Etiqueta para Pago
            if pct_payment > 0:
                ax2.text(i, pct_no_payment + pct_payment/2, f'{pct_payment:.1f}%', 
                        ha='center', va='center', fontweight='bold')
    
    # Rotar etiquetas de meses si es necesario
    if len(payment_by_month) > 6:
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_payment_trend_by_month(
    df: pd.DataFrame,
    payment_col: str = 'pago',
    month_col: str = 'mes',
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Genera un gráfico de tendencia de tasa de pago por mes.
    
    Args:
        df: DataFrame con los datos
        payment_col: Nombre de la columna de pago (binaria 0/1)
        month_col: Nombre de la columna de mes
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con el gráfico
    """
    if payment_col not in df.columns:
        raise ValueError(f"Columna de pago '{payment_col}' no encontrada")
    
    if month_col not in df.columns:
        raise ValueError(f"Columna de mes '{month_col}' no encontrada")
    
    # Calcular tasa de pago por mes
    payment_rate = df.groupby(month_col)[payment_col].mean() * 100
    
    # Ordenar meses cronológicamente
    if payment_rate.index.dtype == 'object':
        month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        # Si los datos son numéricos (1-12), mapear a nombres
        if payment_rate.index.astype(str).str.isnumeric().all():
            month_mapping = {str(i+1): month for i, month in enumerate(month_order)}
            payment_rate.index = payment_rate.index.astype(str).map(month_mapping)
        
        # Reordenar si es posible
        available_months = [month for month in month_order if month in payment_rate.index]
        if available_months:
            payment_rate = payment_rate.reindex(available_months)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Gráfico de línea
    line = ax.plot(payment_rate.index, payment_rate.values, 
                   marker='o', linewidth=2, markersize=8, 
                   color='green', alpha=0.8)
    
    # Añadir área bajo la curva
    ax.fill_between(payment_rate.index, payment_rate.values, 
                   alpha=0.3, color='green')
    
    # Añadir etiquetas de valor
    for i, (month, rate) in enumerate(payment_rate.items()):
        ax.annotate(f'{rate:.1f}%', 
                   (i, rate), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center', 
                   fontweight='bold')
    
    ax.set_title('Tasa de Pago por Mes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Tasa de Pago (%)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(payment_rate.values) * 1.1)
    
    # Rotar etiquetas de meses si es necesario
    if len(payment_rate) > 6:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_imputation_flags(
    df: pd.DataFrame,
    flag_columns: List[str] = None,
    suffix: str = '_imputed',
    n_cols: int = 3,
    figsize: tuple = (15, 8)
) -> plt.Figure:
    """
    Genera gráficos para visualizar los flags de imputación.
    
    Args:
        df: DataFrame con los flags de imputación
        flag_columns: Lista de columnas de flags (si es None, busca columnas con el sufijo)
        suffix: Sufijo para identificar columnas de flags
        n_cols: Número de columnas en la cuadrícula
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con los gráficos
    """
    # Si no se especifican columnas, buscar todas las que tienen el sufijo
    if flag_columns is None:
        flag_columns = [col for col in df.columns if col.endswith(suffix)]
    
    if not flag_columns:
        logger.warning("No se encontraron columnas de flags de imputación")
        return None
    
    # Calcular el número de filas necesarias
    n_rows = (len(flag_columns) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar el array de ejes
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # Generar gráfico para cada flag
    for i, flag_col in enumerate(flag_columns):
        ax = axes[i]
        
        # Contar valores del flag
        flag_counts = df[flag_col].value_counts().sort_index()
        
        # Crear gráfico de barras
        colors = ['lightcoral', 'lightblue']  # 0=original, 1=imputado
        bars = ax.bar(
            x=['Original', 'Imputado'], 
            height=[flag_counts.get(0, 0), flag_counts.get(1, 0)],
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )
        
        # Añadir etiquetas
        ax.set_title(f'Flag: {flag_col.replace(suffix, "")}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Conteo', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Añadir etiquetas con valores y porcentajes
        total = len(df) - df[flag_col].isnull().sum()
        for j, (bar, count) in enumerate(zip(bars, [flag_counts.get(0, 0), flag_counts.get(1, 0)])):
            height = bar.get_height()
            pct = (count / total) * 100 if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + max(flag_counts) * 0.01,
                f'{count}\n({pct:.1f}%)',
                ha='center', 
                va='bottom',
                fontsize=9
            )
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_outliers_boxplot(
    df: pd.DataFrame,
    exempt_fields: List[str] = None,
    method: str = 'iqr',
    threshold: float = 1.5,
    n_cols: int = 3,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Genera boxplots mostrando los outliers en diferente color para variables numéricas.
    
    Args:
        df: DataFrame con los datos
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)
        method: Método para detectar outliers ('iqr' o 'zscore')
        threshold: Umbral para la detección
        n_cols: Número de columnas en la cuadrícula de gráficos
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib con los boxplots
    """
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    # Seleccionar solo columnas numéricas (excluyendo campos exentos)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exempt_fields]
    
    if not numeric_cols:
        logger.warning("No se encontraron columnas numéricas para graficar")
        return None
    
    # Detectar outliers
    outliers_dict = detect_outliers(df, exempt_fields=exempt_fields, method=method, threshold=threshold)
    
    # Calcular el número de filas necesarias
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar el array de ejes para facilitar la iteración
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_cols == 1 else axes
    
    # Generar un boxplot por cada columna numérica
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        
        # Obtener outliers para esta columna
        outlier_indices = outliers_dict.get(col, [])
        
        # Separar datos normales y outliers
        normal_data = df[col].drop(outlier_indices)
        outlier_data = df[col].iloc[outlier_indices]
        
        # Crear boxplot para datos normales
        if len(normal_data) > 0:
            boxplot = ax.boxplot(
                normal_data.dropna(),
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='darkblue', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=8, 
                              markeredgecolor='darkred', alpha=0.8)
            )
        
        # Añadir outliers como puntos rojos si existen
        if len(outlier_data) > 0:
            # Calcular límites del boxplot para posicionar los outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filtrar outliers que están fuera de los límites
            valid_outliers = outlier_data[(outlier_data < lower_bound) | (outlier_data > upper_bound)]
            
            if len(valid_outliers) > 0:
                # Crear scatter plot para outliers
                y_outliers = valid_outliers.values
                x_outliers = [1] * len(y_outliers)  # Posición x = 1 (centro del boxplot)
                
                ax.scatter(
                    x_outliers, 
                    y_outliers, 
                    color='red', 
                    s=60, 
                    alpha=0.8, 
                    edgecolors='darkred',
                    linewidth=1.5,
                    label=f'Outliers ({len(valid_outliers)})',
                    zorder=10
                )
        
        # Configurar el gráfico
        ax.set_title(f'Boxplot de {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Añadir información de outliers en el título
        if col in outliers_dict:
            n_outliers = len(outliers_dict[col])
            pct_outliers = (n_outliers / len(df[col].dropna())) * 100
            ax.set_title(f'Boxplot de {col}\n({n_outliers} outliers, {pct_outliers:.1f}%)', 
                        fontsize=12, fontweight='bold')
        
        # Añadir leyenda si hay outliers
        if col in outliers_dict and len(outliers_dict[col]) > 0:
            ax.legend(loc='upper right', fontsize=9)
        
        # Ocultar etiquetas del eje x
        ax.set_xticks([])
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

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
        
        # 6. Boxplots de outliers
        fig_outliers = plot_outliers_boxplot(df, exempt_fields=exempt_fields)
        if fig_outliers:
            outliers_path = output_dir / 'outliers_boxplot.png'
            save_plot(fig_outliers, outliers_path)
            report['plots']['outliers_boxplot'] = str(outliers_path)
        
        # 7. Detección de outliers
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
