"""
Ejemplo de uso de las funciones modificadas con campos exentos de análisis.

Este script demuestra cómo utilizar las funciones de data.py y eda.py 
especificando campos que deben ser excluidos del análisis numérico,
como IDs de clientes.
"""

import pandas as pd
from pathlib import Path
from src.data import load_data, validate_dataframe
from src.eda import generate_eda_report

def main():
    """Función principal que demuestra el uso de campos exentos."""
    
    # Definir campos que deben ser exentos del análisis
    # Estos campos serán tratados como identificadores y no como variables numéricas
    exempt_fields = ['identificacion', 'id_cliente', 'cliente_id', 'codigo']
    
    # Ejemplo 1: Cargar datos con campos exentos
    print("=" * 60)
    print("EJEMPLO 1: Cargar datos con campos exentos")
    print("=" * 60)
    
    try:
        # Ruta al archivo de datos (ajustar según tu ubicación)
        data_path = Path("data/PruebaDS.xlsx")
        
        if data_path.exists():
            # Cargar datos especificando campos exentos
            df = load_data(data_path, exempt_fields=exempt_fields)
            print(f"✅ Datos cargados exitosamente. Dimensiones: {df.shape}")
            print(f"📋 Campos exentos del análisis: {exempt_fields}")
            
            # Mostrar tipos de datos para verificar
            print("\n📊 Tipos de datos:")
            for col in df.columns:
                exempt_marker = " ⚠️ EXENTO" if col in exempt_fields else ""
                print(f"  - {col}: {df[col].dtype}{exempt_marker}")
                
        else:
            print(f"❌ Archivo no encontrado en: {data_path}")
            return
            
    except Exception as e:
        print(f"❌ Error al cargar los datos: {str(e)}")
        return
    
    # Ejemplo 2: Validar DataFrame con campos exentos
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Validar DataFrame con campos exentos")
    print("=" * 60)
    
    try:
        # Validar el DataFrame con campos exentos
        validation_report = validate_dataframe(df, exempt_fields=exempt_fields)
        
        print("✅ Validación completada")
        print(f"📊 Campos exentos registrados: {validation_report.get('exempt_fields', [])}")
        
        # Mostrar información específica de campos exentos
        print("\n🔍 Análisis de campos exentos:")
        for col, stats in validation_report['column_stats'].items():
            if col in exempt_fields:
                print(f"  - {col}:")
                print(f"    • Tipo: {stats['dtype']}")
                print(f"    • Valores únicos: {stats.get('unique_values', 'N/A')}")
                print(f"    • Valores faltantes: {stats['missing']} ({stats['pct_missing']}%)")
                if 'sample_values' in stats:
                    print(f"    • Muestra de valores: {stats['sample_values']}")
                
    except Exception as e:
        print(f"❌ Error en la validación: {str(e)}")
    
    # Ejemplo 3: Generar reporte EDA con campos exentos
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Generar reporte EDA con campos exentos")
    print("=" * 60)
    
    try:
        # Generar reporte EDA excluyendo los campos especificados
        output_dir = Path("outputs/eda_con_exentos")
        eda_report = generate_eda_report(
            df=df,
            output_dir=output_dir,
            target_col='pago',
            exempt_fields=exempt_fields
        )
        
        print("✅ Reporte EDA generado exitosamente")
        print(f"📁 Gráficos guardados en: {output_dir}")
        print(f"📊 Columnas numéricas analizadas: {eda_report['numerical_columns']}")
        print(f"📋 Columnas categóricas analizadas: {eda_report['categorical_columns']}")
        
        # Mostrar resumen de outliers (solo para columnas no exentas)
        if eda_report.get('outliers'):
            print("\n⚠️ Outliers detectados (solo en columnas numéricas no exentas):")
            for col, count in eda_report['outliers'].items():
                print(f"  - {col}: {count} outliers")
        else:
            print("\n✅ No se detectaron outliers en las columnas numéricas analizadas")
            
    except Exception as e:
        print(f"❌ Error al generar el reporte EDA: {str(e)}")
    
    # Ejemplo 4: Demostrar la diferencia con y sin campos exentos
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Comparación con/sin campos exentos")
    print("=" * 60)
    
    try:
        # Análisis SIN campos exentos
        print("📈 Análisis SIN campos exentos:")
        numeric_cols_sin_exentos = df.select_dtypes(include=['number']).columns.tolist()
        print(f"  Columnas numéricas detectadas: {numeric_cols_sin_exentos}")
        
        # Análisis CON campos exentos
        print("\n📊 Análisis CON campos exentos:")
        numeric_cols_con_exentos = [
            col for col in df.select_dtypes(include=['number']).columns.tolist() 
            if col not in exempt_fields
        ]
        print(f"  Columnas numéricas para análisis: {numeric_cols_con_exentos}")
        print(f"  Campos excluidos: {[col for col in numeric_cols_sin_exentos if col in exempt_fields]}")
        
    except Exception as e:
        print(f"❌ Error en la comparación: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ EJEMPLOS COMPLETADOS")
    print("=" * 60)
    print("\n💡 Nota: Los campos exentos son tratados como identificadores")
    print("   y no se incluyen en análisis estadísticos, correlaciones,")
    print("   detección de outliers u otros análisis numéricos.")

if __name__ == "__main__":
    main()
