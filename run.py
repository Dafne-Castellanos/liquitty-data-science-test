"""
Script principal para la ejecución del flujo completo del proyecto.

Este script carga los datos, realiza el análisis exploratorio, entrena el modelo
y genera los resultados finales.
"""

import sys
import logging
import json
from pathlib import Path
import pandas as pd
from src.data import load_data, validate_dataframe
from src.eda import generate_eda_report
from src.preprocessing import preprocess_data, split_data
from src.modeling import train_models, evaluate_model, save_model
from src.utils import setup_logging, save_metrics

# Configuración de logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Función principal que ejecuta el flujo completo del proyecto."""
    try:
        # Configuración de rutas
        project_dir = Path(__file__).parent
        data_dir = project_dir / "data"
        outputs_dir = project_dir / "outputs"
        figures_dir = outputs_dir / "figures"
        
        # Crear directorios necesarios
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Cargar datos
        logger.info("Cargando datos...")
        data_file = data_dir / "PruebaDS.xlsx"
        df = load_data(data_file)
        
        # 2. Análisis exploratorio
        logger.info("Realizando análisis exploratorio...")
        eda_report = generate_eda_report(df, figures_dir)
        
        # 3. Preprocesamiento
        logger.info("Preprocesando datos...")
        X, y = preprocess_data(df, target_column='pago')
        
        # 4. División de datos
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Entrenamiento de modelos
        logger.info("Entrenando modelos...")
        models = train_models(X_train, y_train)
        
        # 6. Evaluación de modelos
        logger.info("Evaluando modelos...")
        metrics = {}
        for name, model in models.items():
            logger.info(f"Evaluando modelo: {name}")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics[name] = evaluate_model(
                y_test, y_pred, y_proba, 
                model_name=name,
                output_dir=figures_dir
            )
        
        # 7. Guardar resultados
        logger.info("Guardando resultados...")
        
        # Guardar métricas
        metrics_path = outputs_dir / "metrics.json"
        save_metrics(metrics, metrics_path)
        logger.info(f"Métricas guardadas en: {metrics_path}")
        
        # Guardar el mejor modelo
        best_model_name = max(metrics.items(), key=lambda x: x[1].get('roc_auc', 0))[0]
        best_model = models[best_model_name]
        model_path = outputs_dir / "model.pkl"
        save_model(best_model, model_path)
        logger.info(f"Mejor modelo guardado en: {model_path}")
        
        # 8. Mostrar resumen
        print("\n" + "="*50)
        print("RESUMEN DE LA EJECUCIÓN")
        print("="*50)
        print(f"- Archivo de datos: {data_file}")
        print(f"- Tamaño del dataset: {len(df)} filas, {len(df.columns)} columnas")
        print("\nMétricas de rendimiento:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            print(f"  - Accuracy: {model_metrics['accuracy']:.4f}")
            print(f"  - ROC AUC: {model_metrics['roc_auc']:.4f}")
            print(f"  - F1 Score: {model_metrics['f1_score']:.4f}")
        
        print(f"\nResultados guardados en: {outputs_dir.absolute()}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error en la ejecución: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
