# src/modeling.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from typing import Dict, List, Tuple, Any
from pathlib import Path

# --------------------------
# EVALUACIÓN DE MODELOS
# --------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, figures_dir=None):
    """
    Evalúa un modelo y genera métricas y visualizaciones.
    
    Args:
        model: Modelo a evaluar
        X_train: Datos de entrenamiento
        y_train: Target de entrenamiento
        X_test: Datos de prueba
        y_test: Target de prueba
        model_name: Nombre del modelo para visualizaciones
        figures_dir: Directorio para guardar gráficos
        
    Returns:
        Tuple con (métricas train, métricas test, modelo entrenado)
    """
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predecir en TRAIN
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)
    
    # Predecir en TEST
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Calcular métricas TRAIN
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train, y_train_pred_proba) if len(np.unique(y_train)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_train, y_train_pred_proba) if len(np.unique(y_train)) > 1 else 0.5
    }
    
    # Calcular métricas TEST
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_test, y_test_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Calcular overfitting
    overfitting_analysis = {}
    for metric in train_metrics.keys():
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        diff = train_val - test_val
        pct_diff = (diff / test_val * 100) if test_val != 0 else 0
        
        overfitting_analysis[metric] = {
            'train': train_val,
            'test': test_val,
            'diff': diff,
            'pct_diff': pct_diff,
            'overfitting_risk': 'High' if pct_diff > 10 else 'Medium' if pct_diff > 5 else 'Low'
        }
    
    # Mostrar matriz de confusión (solo test)
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Pago', 'Pago'],
                yticklabels=['No Pago', 'Pago'])
    plt.title(f'Matriz de Confusión - {model_name} (Test)')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    
    if figures_dir:
        plt.savefig(figures_dir / f'confusion_matrix_{model_name.lower().replace(' ', '_')}.png', 
                   bbox_inches='tight')
    plt.show()
    
    # Mostrar curva ROC (solo test)
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {model_name} (Test)')
        plt.legend(loc='lower right')
        
        if figures_dir:
            plt.savefig(figures_dir / f'roc_curve_{model_name.lower().replace(' ', '_')}.png', 
                       bbox_inches='tight')
        plt.show()
    
    # Imprimir análisis de overfitting
    print(f"\n📊 Análisis de Overfitting - {model_name}:")
    print(f"{'Métrica':<12} {'Train':<8} {'Test':<8} {'Diferencia':<10} {'% Diff':<8} {'Riesgo':<8}")
    print("-" * 60)
    for metric, analysis in overfitting_analysis.items():
        print(f"{metric:<12} {analysis['train']:<8.3f} {analysis['test']:<8.3f} "
              f"{analysis['diff']:<10.3f} {analysis['pct_diff']:<8.1f}% {analysis['overfitting_risk']:<8}")
    
    return train_metrics, test_metrics, model, overfitting_analysis

# --------------------------
# DEFINICIÓN DE MODELOS
# --------------------------

def get_models(y_train):
    """
    Define los modelos a evaluar con balanceo de clases.
    
    Args:
        y_train: Target de entrenamiento para calcular balance
        
    Returns:
        Diccionario con modelos configurados
    """
    # Calcular scale_pos_weight para XGBoost
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    
    models = {
        'Regresión Logística': LogisticRegression(
            random_state=15, 
            max_iter=1000, 
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=15, 
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=15),
        'XGBoost': xgb.XGBClassifier(
            random_state=15, 
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    return models

# --------------------------
# FUNCIÓN PRINCIPAL DE MODELADO
# --------------------------

def train_and_evaluate_models(X_train, y_train, X_test, y_test, 
                            selected_features_importance, selected_features_rfe,
                            feature_names, figures_dir=None, output_dir=None):
    """
    Entrena y evalúa modelos con características seleccionadas.
    
    Args:
        X_train: Datos de entrenamiento completos
        y_train: Target de entrenamiento
        X_test: Datos de prueba completos
        y_test: Target de prueba
        selected_features_importance: Características seleccionadas por importancia
        selected_features_rfe: Características seleccionadas por RFE
        feature_names: Nombres de todas las características
        figures_dir: Directorio para guardar gráficos
        output_dir: Directorio para guardar modelos
        
    Returns:
        Diccionario con resultados completos
    """
    print("🔧 Iniciando modelado con características seleccionadas...")
    
    # Combinar características seleccionadas (top 5 de cada método, sin duplicados)
    top_importance = selected_features_importance[:5]
    top_rfe = selected_features_rfe[:5]
    
    # Unir y eliminar duplicados manteniendo orden
    combined_features = []
    seen = set()
    
    # Agregar top importancia primero
    for feature in top_importance:
        if feature not in seen:
            combined_features.append(feature)
            seen.add(feature)
    
    # Agregar top RFE que no estén ya incluidas
    for feature in top_rfe:
        if feature not in seen:
            combined_features.append(feature)
            seen.add(feature)
    
    print(f"📊 Características por importancia (top 5): {top_importance}")
    print(f"📊 Características por RFE (top 5): {top_rfe}")
    print(f"📊 Características combinadas: {combined_features}")
    print(f"📊 Total características seleccionadas: {len(combined_features)}")
    
    # Crear máscaras para filtrar datos
    mask_importance = [feature in top_importance for feature in feature_names]
    mask_rfe = [feature in top_rfe for feature in feature_names]
    mask_combined = [feature in combined_features for feature in feature_names]
    
    # Filtrar datos
    X_train_importance = X_train[:, mask_importance]
    X_test_importance = X_test[:, mask_importance]
    
    X_train_rfe = X_train[:, mask_rfe]
    X_test_rfe = X_test[:, mask_rfe]
    
    X_train_combined = X_train[:, mask_combined]
    X_test_combined = X_test[:, mask_combined]
    
    # Obtener modelos
    models = get_models(y_train)
    
    results = {}
    train_results = {}
    overfitting_results = {}
    trained_models = {}
    
    # Evaluar con características combinadas (principal)
    print(f"\n🎯 Evaluando con características combinadas ({len(combined_features)} features)...")
    
    for name, model in models.items():
        print(f'🔍 Evaluando modelo: {name}')
        train_metrics, test_metrics, trained_model, overfitting = evaluate_model(
            model, X_train_combined, y_train, 
            X_test_combined, y_test, f"{name} (Combinado)", figures_dir
        )
        
        # Guardar resultados con sufijo
        results[f"{name}_combinado"] = test_metrics
        trained_models[f"{name}_combinado"] = trained_model
        
        # Guardar métricas de train y overfitting
        train_results[f"{name}_combinado"] = train_metrics
        overfitting_results[f"{name}_combinado"] = overfitting
    
    # Evaluar con características por importancia (opcional, para comparación)
    print(f"\n📊 Evaluando con características por importancia ({len(top_importance)} features)...")
    
    for name, model in models.items():
        train_metrics, test_metrics, trained_model, overfitting = evaluate_model(
            model, X_train_importance, y_train, 
            X_test_importance, y_test, f"{name} (Importancia)", figures_dir
        )
        
        results[f"{name}_importancia"] = test_metrics
        trained_models[f"{name}_importancia"] = trained_model
        train_results[f"{name}_importancia"] = train_metrics
        overfitting_results[f"{name}_importancia"] = overfitting
    
    # Evaluar con características por RFE (opcional, para comparación)
    print(f"\n📊 Evaluando con características por RFE ({len(top_rfe)} features)...")
    
    for name, model in models.items():
        train_metrics, test_metrics, trained_model, overfitting = evaluate_model(
            model, X_train_rfe, y_train, 
            X_test_rfe, y_test, f"{name} (RFE)", figures_dir
        )
        
        results[f"{name}_rfe"] = test_metrics
        trained_models[f"{name}_rfe"] = trained_model
        train_results[f"{name}_rfe"] = train_metrics
        overfitting_results[f"{name}_rfe"] = overfitting
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Separar resultados por método de selección
    combined_results = results_df[results_df.index.str.contains('_combinado')]
    importancia_results = results_df[results_df.index.str.contains('_importancia')]
    rfe_results = results_df[results_df.index.str.contains('_rfe')]
    
    # Limpiar nombres de índice
    combined_results.index = combined_results.index.str.replace('_combinado', '')
    importancia_results.index = importancia_results.index.str.replace('_importancia', '')
    rfe_results.index = rfe_results.index.str.replace('_rfe', '')
    
    # Crear tablas consolidadas de train vs test
    print(f"\n📊 TABLAS COMPARATIVAS TRAIN vs TEST")
    
    # Tabla combinada
    combined_train_df = pd.DataFrame.from_dict({k: v for k, v in train_results.items() if '_combinado' in k}, orient='index')
    combined_train_df.index = combined_train_df.index.str.replace('_combinado', '')
    combined_comparison = create_train_test_comparison_table(
        {k: v for k, v in train_results.items() if '_combinado' in k},
        {k: v for k, v in results.items() if '_combinado' in k},
        "Combinado"
    )
    
    # Tabla importancia
    importancia_train_df = pd.DataFrame.from_dict({k: v for k, v in train_results.items() if '_importancia' in k}, orient='index')
    importancia_train_df.index = importancia_train_df.index.str.replace('_importancia', '')
    importancia_comparison = create_train_test_comparison_table(
        {k: v for k, v in train_results.items() if '_importancia' in k},
        {k: v for k, v in results.items() if '_importancia' in k},
        "Importancia"
    )
    
    # Tabla RFE
    rfe_train_df = pd.DataFrame.from_dict({k: v for k, v in train_results.items() if '_rfe' in k}, orient='index')
    rfe_train_df.index = rfe_train_df.index.str.replace('_rfe', '')
    rfe_comparison = create_train_test_comparison_table(
        {k: v for k, v in train_results.items() if '_rfe' in k},
        {k: v for k, v in results.items() if '_rfe' in k},
        "RFE"
    )
    
    # Tabla general (todos los métodos)
    general_comparison = pd.concat([combined_comparison, importancia_comparison, rfe_comparison], ignore_index=True)
    
    print(f"\n🎯 TABLA GENERAL TRAIN vs TEST:")
    display(general_comparison)
    
    # Guardar tablas si hay directorio de salida
    if output_dir:
        general_comparison.to_csv(output_dir / 'train_test_comparison.csv', index=False)
        print(f'✅ Tabla comparativa guardada en: {output_dir / "train_test_comparison.csv"}')
    
    print(f"\n📈 RESULTADOS COMPARATIVOS")
    print(f"🎯 Características Combinadas:")
    print(combined_results.sort_values('roc_auc', ascending=False))
    
    print(f"\n📊 Características por Importancia:")
    print(importancia_results.sort_values('roc_auc', ascending=False))
    
    print(f"\n🔄 Características por RFE:")
    print(rfe_results.sort_values('roc_auc', ascending=False))
    
    # Guardar el mejor modelo (de características combinadas)
    best_model_name = combined_results['roc_auc'].idxmax()
    best_model_key = f"{best_model_name}_combinado"
    best_model = trained_models[best_model_key]
    
    if output_dir:
        # Guardar mejor modelo
        model_path = output_dir / 'best_model_selected_features.joblib'
        joblib.dump(best_model, model_path)
        print(f'✅ Mejor modelo guardado: {best_model_name} en {model_path}')
        
        # Guardar características seleccionadas
        features_path = output_dir / 'selected_features.txt'
        with open(features_path, 'w') as f:
            f.write(f"Características seleccionadas (combinadas):\n")
            for i, feature in enumerate(combined_features, 1):
                f.write(f"{i}. {feature}\n")
        print(f'✅ Características guardadas en: {features_path}')
        
        # Guardar resultados
        results_path = output_dir / 'model_results.csv'
        results_df.to_csv(results_path)
        print(f'✅ Resultados guardados en: {results_path}')
    
    return {
        'results': results,
        'train_results': train_results,
        'overfitting_results': overfitting_results,
        'results_df': results_df,
        'combined_results': combined_results,
        'importancia_results': importancia_results,
        'rfe_results': rfe_results,
        'trained_models': trained_models,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'selected_features': combined_features,
        'top_importance': top_importance,
        'top_rfe': top_rfe,
        'train_test_comparison': general_comparison,
        'combined_comparison': combined_comparison,
        'importancia_comparison': importancia_comparison,
        'rfe_comparison': rfe_comparison
    }

def create_train_test_comparison_table(train_results, test_results, method_name=""):
    """
    Crea una tabla consolidada comparando train vs test para todos los modelos.
    
    Args:
        train_results: Diccionario con métricas de train
        test_results: Diccionario con métricas de test
        method_name: Nombre del método (ej: "Combinado", "Importancia", "RFE")
        
    Returns:
        DataFrame con tabla comparativa
    """
    comparison_data = []
    
    for model_key in test_results.keys():
        if model_key in train_results:
            model_name = model_key.replace(f'_{method_name.lower()}', '') if method_name else model_key
            
            # Para cada métrica
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
                train_val = train_results[model_key].get(metric, 0)
                test_val = test_results[model_key].get(metric, 0)
                diff = train_val - test_val
                pct_diff = (diff / test_val * 100) if test_val != 0 else 0
                
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Train': train_val,
                    'Test': test_val,
                    'Difference': diff,
                    '%_Diff': pct_diff,
                    'Method': method_name
                })
    
    df = pd.DataFrame(comparison_data)
    
    # Reordenar columnas
    df = df[['Model', 'Metric', 'Train', 'Test', 'Difference', '%_Diff', 'Method']]
    
    return df

def plot_train_test_comparison(comparison_df, figsize=(15, 10)):
    """
    Grafica comparación de train vs test para todos los modelos.
    
    Args:
        comparison_df: DataFrame con tabla comparativa
        figsize: Tamaño de la figura
    """
    # Pivot para graficar
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filtrar por métrica
        metric_data = comparison_df[comparison_df['Metric'] == metric.upper()]
        
        if len(metric_data) > 0:
            models = metric_data['Model'].values
            train_vals = metric_data['Train'].values
            test_vals = metric_data['Test'].values
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8, color='blue')
            ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8, color='red')
            
            ax.set_xlabel('Modelos')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Train vs Test - {metric.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(results_dict, figsize=(15, 10)):
    """
    Grafica comparación de modelos por método de selección.
    
    Args:
        results_dict: Resultados devueltos por train_and_evaluate_models
        figsize: Tamaño de la figura
    """
    combined_results = results_dict['combined_results']
    importancia_results = results_dict['importancia_results']
    rfe_results = results_dict['rfe_results']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Métricas a graficar
    metrics = ['roc_auc', 'accuracy', 'precision', 'recall']
    titles = ['ROC AUC', 'Accuracy', 'Precision', 'Recall']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Datos para cada método
        x = np.arange(len(combined_results))
        width = 0.25
        
        # Graficar barras para cada método
        ax.bar(x - width, combined_results[metric], width, 
               label='Combinado', alpha=0.8, color='green')
        ax.bar(x, importancia_results[metric], width, 
               label='Por Importancia', alpha=0.8, color='blue')
        ax.bar(x + width, rfe_results[metric], width, 
               label='Por RFE', alpha=0.8, color='orange')
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel(title)
        ax.set_title(f'Comparación de {title} por Método de Selección')
        ax.set_xticks(x)
        ax.set_xticklabels(combined_results.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
