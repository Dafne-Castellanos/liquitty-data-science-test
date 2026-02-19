"""
Módulo para la carga y validación de datos.

Este módulo proporciona funciones para cargar el dataset de Excel y validar su calidad.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
MAX_FILE_SIZE_MB = 100  # Tamaño máximo de archivo en MB
REQUIRED_COLUMNS = ['pago']  # Columna obligatoria


def _find_similar_columns(df_columns: List[str], target: str, threshold: float = 0.8) -> List[str]:
    """
    Encuentra columnas con nombres similares al objetivo.

    Args:
        df_columns: Lista de nombres de columnas del DataFrame
        target: Nombre de la columna objetivo
        threshold: Umbral de similitud (0-1)

    Returns:
        Lista de nombres de columnas similares
    """
    from difflib import get_close_matches
    return get_close_matches(
        target.lower(),
        [col.lower() for col in df_columns],
        n=5,
        cutoff=threshold
    )


def _validate_file(path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Valida el archivo antes de intentar cargarlo.

    Args:
        path: Ruta al archivo

    Returns:
        Tupla (es_válido, mensaje)
    """
    path = Path(path)

    # Verificar si el archivo existe
    if not path.exists():
        return False, f"El archivo no existe: {path}"

    # Verificar la extensión
    if path.suffix.lower() != '.xlsx':
        return False, f"El archivo debe tener extensión .xlsx, no {path.suffix}"

    # Verificar tamaño del archivo
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, (
            f"El archivo es demasiado grande ({file_size_mb:.2f} MB). "
            f"Tamaño máximo permitido: {MAX_FILE_SIZE_MB} MB"
        )
    if file_size_mb == 0:
        return False, "El archivo está vacío"

    return True, ""


def _generate_quality_report(df: pd.DataFrame, exempt_fields: List[str] = None) -> Dict[str, Any]:
    """
    Genera un reporte de calidad de datos.

    Args:
        df: DataFrame a analizar
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)

    Returns:
        Diccionario con el reporte de calidad
    """
    # Inicializar lista de campos exentos
    if exempt_fields is None:
        exempt_fields = []
    
    report = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_rates': {},
        'warnings': [],
        'errors': [],
        'column_stats': {},
        'exempt_fields': exempt_fields,
        'global_stats': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_missing': df.isnull().sum().sum(),
            'total_duplicates': df.duplicated().sum(),
            'pct_missing': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
    }

    # Análisis por columna
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'pct_missing': float((df[col].isnull().mean() * 100).round(2)),
            'unique': int(df[col].nunique()),
            'is_constant': bool(df[col].nunique() <= 1)
        }

        # Estadísticas para columnas numéricas (que no sean campos exentos)
        if pd.api.types.is_numeric_dtype(df[col]) and col not in exempt_fields:
            col_stats.update({
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                '25%': float(df[col].quantile(0.25)),
                '50%': float(df[col].median()),
                '75%': float(df[col].quantile(0.75)),
                'max': float(df[col].max()),
                'has_negatives': bool((df[col] < 0).any()),
                'has_zeros': bool((df[col] == 0).any()),
                'has_inf': bool(np.isinf(df[col]).any())
            })

        # Para campos exentos, solo guardar información básica
        elif col in exempt_fields:
            col_stats.update({
                'is_exempt': True,
                'exempt_reason': 'Campo identificado como ID o campo exento de análisis',
                'unique_values': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist()
            })
        # Estadísticas para columnas categóricas
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            col_stats.update({
                'top_values': df[col].value_counts().nlargest(5).to_dict(),
                'freq': float(df[col].value_counts(normalize=True).iloc[0] * 100)
            })

        # Detección de fechas
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            min_date = df[col].min()
            max_date = df[col].max()
            col_stats.update({
                'min_date': str(min_date) if not pd.isna(min_date) else None,
                'max_date': str(max_date) if not pd.isna(max_date) else None,
                'time_span': (max_date - min_date).days if len(df[col].dropna()) > 0 else None
            })

        report['column_stats'][col] = col_stats

        # Verificar si la columna es constante
        if col_stats['is_constant'] and col not in ['pago']:
            report['warnings'].append(f"La columna '{col}' es constante (todos los valores son iguales)")

        # Verificar alta tasa de valores faltantes
        if col_stats['pct_missing'] > 50:
            report['warnings'].append(
                f"La columna '{col}' tiene una alta tasa de valores faltantes: {col_stats['pct_missing']}%"
            )

    # Verificar duplicados
    if report['global_stats']['total_duplicates'] > 0:
        report['warnings'].append(
            f"Se encontraron {report['global_stats']['total_duplicates']} filas duplicadas"
        )

    # Verificar si hay columnas completamente nulas
    null_columns = [col for col in df.columns if df[col].isnull().all()]
    if null_columns:
        report['warnings'].append(
            f"Las siguientes columnas están completamente vacías: {', '.join(null_columns)}"
        )

    # Verificar si la variable objetivo existe
    if 'pago' not in df.columns:
        similar_cols = _find_similar_columns(df.columns, 'pago')
        error_msg = "No se encontró la columna 'pago' en el dataset."
        if similar_cols:
            error_msg += f" Columnas similares encontradas: {', '.join(similar_cols)}"
        report['errors'].append(error_msg)
    else:
        # Verificar si la variable objetivo es constante
        if df['pago'].nunique() <= 1:
            report['errors'].append(
                "La variable objetivo 'pago' es constante (todos los valores son iguales)"
            )

        # Verificar si la variable objetivo tiene valores faltantes
        if df['pago'].isnull().any():
            report['warnings'].append("La variable objetivo 'pago' contiene valores faltantes")

    return report


def load_data(file_path: Union[str, Path], exempt_fields: List[str] = None) -> pd.DataFrame:
    """
    Carga y valida el archivo Excel con los datos.

    Args:
        file_path: Ruta al archivo Excel
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)

    Returns:
        DataFrame con los datos cargados

    Raises:
        FileNotFoundError: Si el archivo no existe o no es accesible
        ValueError: Si el archivo no cumple con los requisitos de validación
    """
    file_path = Path(file_path)
    logger.info("Intentando cargar el archivo: %s", file_path)

    # Validar el archivo
    is_valid, error_msg = _validate_file(file_path)
    if not is_valid:
        logger.error("Error de validación del archivo: %s", error_msg)
        raise ValueError(error_msg)

    try:
        # Cargar el archivo Excel
        logger.info("Cargando el archivo Excel...")
        df = pd.read_excel(file_path, engine='openpyxl')

        # Validar que el DataFrame no esté vacío
        if df.empty:
            raise ValueError("El archivo Excel está vacío")

        # Normalizar nombres de columnas (sin modificar los originales)
        df_columns_original = df.columns.tolist()
        df_columns_clean = [str(col).strip().lower() for col in df_columns_original]

        # Verificar columnas duplicadas después de la limpieza
        if len(df_columns_clean) != len(set(df_columns_clean)):
            duplicates = [
                item for item, count in
                pd.Series(df_columns_clean).value_counts().items()
                if count > 1
            ]
            raise ValueError(
                f"Se encontraron nombres de columnas duplicados después de la normalización: {duplicates}"
            )

        # Generar reporte de calidad
        quality_report = _generate_quality_report(df, exempt_fields)

        # Verificar si hay errores críticos
        if quality_report['errors']:
            for error in quality_report['errors']:
                logger.error("Error crítico: %s", error)
            raise ValueError(
                "Se encontraron errores críticos en los datos. Ver el registro para más detalles."
            )

        # Mostrar advertencias
        for warning in quality_report['warnings']:
            logger.warning("Advertencia: %s", warning)

        # Guardar el reporte de calidad
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        report_path = output_dir / "data_quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            # Convertir tipos no serializables a strings
            serializable_report = json.loads(
                json.dumps(
                    quality_report,
                    default=lambda x: str(x) if not isinstance(x, (int, float, str, bool, type(None))) else x
                )
            )
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

        logger.info("Reporte de calidad guardado en: %s", report_path)

        # Mostrar resumen en consola
        print("\n" + "="*50)
        print("RESUMEN DE CALIDAD DE DATOS")
        print("="*50)
        print(f"Archivo: {file_path.name}")
        print(f"Filas: {quality_report['shape'][0]}, Columnas: {quality_report['shape'][1]}")
        print(
            f"Valores faltantes: {quality_report['global_stats']['total_missing']} "
            f"({quality_report['global_stats']['pct_missing']:.2f}%)"
        )
        print(f"Filas duplicadas: {quality_report['global_stats']['total_duplicates']}")

        # Mostrar columnas con mayor tasa de valores faltantes (top 5)
        missing_rates = {}
        for col, stats in quality_report['column_stats'].items():
            missing_rates[col] = stats['pct_missing']

        if missing_rates:
            print("\nColumnas con mayor porcentaje de valores faltantes:")
            for col, rate in sorted(missing_rates.items(), key=lambda x: x[1], reverse=True)[:5]:
                if rate > 0:
                    print(f"  - {col}: {rate:.2f}%")

        # Mostrar advertencias si las hay
        if quality_report['warnings']:
            print("\nAdvertencias:")
            for warning in quality_report['warnings'][:5]:  # Mostrar solo las primeras 5 advertencias
                print(f"  - {warning}")
            if len(quality_report['warnings']) > 5:
                print(f"  - ... y {len(quality_report['warnings']) - 5} advertencias más")

        print("\n" + "="*50 + "\n")

        return df

    except Exception as e:
        logger.error("Error al cargar el archivo: %s", str(e), exc_info=True)
        raise


def validate_dataframe(df: pd.DataFrame, exempt_fields: List[str] = None) -> Dict[str, Any]:
    """
    Valida un DataFrame de pandas según los criterios de calidad de datos.

    
    Args:
        df: DataFrame a validar
        exempt_fields: Lista de nombres de campos exentos de análisis (IDs, etc.)
        
    Returns:
        Diccionario con el reporte de validación
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame de pandas")
    
    return _generate_quality_report(df, exempt_fields)


if __name__ == "__main__":
    # Ejemplo de uso
    try:
        # Ruta de ejemplo (ajustar según sea necesario)
        sample_path = Path("data/PruebaDS.xlsx")
        
        # Crear directorio de datos si no existe
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Verificar si el archivo de ejemplo existe
        if sample_path.exists():
            df = load_data(sample_path)
            print(f"Datos cargados exitosamente. Dimensiones: {df.shape}")
            print("\nPrimeras filas del dataset:")
            print(df.head())
        else:
            print(f"Archivo de ejemplo no encontrado en: {sample_path}")
            print("Por favor, coloque el archivo 'PruebaDS.xlsx' en la carpeta 'data/'")
    
    except Exception as e:
        print(f"Error: {str(e)}")
