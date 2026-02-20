# src/balanced_modeling.py
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# --------------------------
# BALANCEO DE CLASES
# --------------------------

def apply_balancing(X_train, y_train, method='undersampling', random_state=42):
    """
    Aplica técnicas de balanceo de clases.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        method: Método de balanceo ('undersampling', 'oversampling', 'smote')
        random_state: Semilla aleatoria
        
    Returns:
        Tuple con (X_balanced, y_balanced)
    """
    print(f"🔧 Aplicando {method}...")
    
    # Mostrar distribución original
    print(f"📊 Distribución original:")
    print(f"   Clase 0 (No Pago): {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"   Clase 1 (Pago): {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
    
    if method == 'undersampling':
        sampler = RandomUnderSampler(random_state=random_state)
        
    elif method == 'oversampling':
        sampler = RandomOverSampler(random_state=random_state)
        
    elif method == 'smote':
        sampler = SMOTE(random_state=random_state)
        
    else:
        raise ValueError(f"Método {method} no soportado. Usa 'undersampling', 'oversampling' o 'smote'")
    
    # Aplicar balanceo
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    
    # Mostrar distribución balanceada
    print(f"📊 Distribución después de {method}:")
    print(f"   Clase 0 (No Pago): {sum(y_balanced == 0)} ({sum(y_balanced == 0)/len(y_balanced)*100:.1f}%)")
    print(f"   Clase 1 (Pago): {sum(y_balanced == 1)} ({sum(y_balanced == 1)/len(y_balanced)*100:.1f}%)")
    print(f"📊 Shape original: {X_train.shape} → Shape balanceado: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def plot_class_distribution(y_original, y_balanced, method, figsize=(12, 4)):
    """
    Grafica la distribución de clases antes y después del balanceo.
    
    Args:
        y_original: Target original
        y_balanced: Target balanceado
        method: Método de balanceo utilizado
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Distribución original
    unique_orig, counts_orig = np.unique(y_original, return_counts=True)
    axes[0].bar(['No Pago (0)', 'Pago (1)'], counts_orig, color=['red', 'green'], alpha=0.7)
    axes[0].set_title('Distribución Original')
    axes[0].set_ylabel('Cantidad')
    
    # Añadir porcentajes
    for i, count in enumerate(counts_orig):
        pct = count / len(y_original) * 100
        axes[0].text(i, count + max(counts_orig)*0.01, f'{count}\n({pct:.1f}%)', 
                    ha='center', va='bottom')
    
    # Distribución balanceada
    unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
    axes[1].bar(['No Pago (0)', 'Pago (1)'], counts_bal, color=['red', 'green'], alpha=0.7)
    axes[1].set_title(f'Distribución después de {method.title()}')
    axes[1].set_ylabel('Cantidad')
    
    # Añadir porcentajes
    for i, count in enumerate(counts_bal):
        pct = count / len(y_balanced) * 100
        axes[1].text(i, count + max(counts_bal)*0.01, f'{count}\n({pct:.1f}%)', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# --------------------------
# EVALUACIÓN DE MODELOS (adaptado de modeling.py)
# --------------------------

def evaluate_model_balanced(model, X_train_balanced, y_train_balanced, X_test, y_test, model_name, figures_dir=None):
    """
    Evalúa un modelo entrenado con datos balanceados.
    
    Args:
        model: Modelo a evaluar
        X_train_balanced: Datos de entrenamiento balanceados
        y_train_balanced: Target de entrenamiento balanceado
        X_test: Datos de prueba (sin balancear)
        y_test: Target de prueba (sin balancear)
        model_name: Nombre del modelo para visualizaciones
        figures_dir: Directorio para guardar gráficos
        
    Returns:
        Tuple con (métricas train, métricas test, modelo entrenado, overfitting analysis)
    """
    # Entrenar modelo con datos balanceados
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predecir en TRAIN (balanceado)
    y_train_pred_proba = model.predict_proba(X_train_balanced)[:, 1]
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)
    
    # Predecir en TEST (originales, sin balancear)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Calcular métricas TRAIN
    train_metrics = {
        'accuracy': accuracy_score(y_train_balanced, y_train_pred),
        'precision': precision_score(y_train_balanced, y_train_pred, zero_division=0),
        'recall': recall_score(y_train_balanced, y_train_pred, zero_division=0),
        'f1': f1_score(y_train_balanced, y_train_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train_balanced, y_train_pred_proba) if len(np.unique(y_train_balanced)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_train_balanced, y_train_pred_proba) if len(np.unique(y_train_balanced)) > 1 else 0.5
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

def get_models_balanced(y_train_balanced):
    """
    Define los modelos para datos balanceados.
    
    Args:
        y_train_balanced: Target balanceado para calcular parámetros
        
    Returns:
        Diccionario con modelos configurados
    """
    models = {
        'Regresión Logística': LogisticRegression(
            random_state=15, 
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=15
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=15),
        'XGBoost': xgb.XGBClassifier(
            random_state=15,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    return models

# --------------------------
# FUNCIÓN PRINCIPAL DE MODELADO BALANCEADO
# --------------------------

def train_balanced_models(X_train, y_train, X_test, y_test, 
                         selected_features, feature_names,
                         figures_dir=None, output_dir=None):
    """
    Entrena modelos con diferentes técnicas de balanceo.
    
    Args:
        X_train: Datos de entrenamiento completos
        y_train: Target de entrenamiento
        X_test: Datos de prueba completos
        y_test: Target de prueba
        selected_features: Lista de características seleccionadas
        feature_names: Nombres de todas las características
        figures_dir: Directorio para guardar gráficos
        output_dir: Directorio para guardar modelos
        
    Returns:
        Diccionario con resultados completos
    """
    print("🔧 Iniciando modelado con balanceo de clases...")
    print(f"📊 Características seleccionadas: {selected_features}")
    print(f"📊 Total características: {len(selected_features)}")
    
    # Crear máscara para características seleccionadas
    mask = [feature in selected_features for feature in feature_names]
    
    # Filtrar datos
    X_train_selected = X_train[:, mask]
    X_test_selected = X_test[:, mask]
    
    print(f"📊 Shape datos filtrados - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")
    
    # Métodos de balanceo
    balancing_methods = ['undersampling', 'oversampling', 'smote']
    
    # Almacenar resultados
    all_results = {}
    all_train_results = {}
    all_trained_models = {}
    all_overfitting_results = {}
    balancing_info = {}
    
    for method in balancing_methods:
        print(f"\n{'='*60}")
        print(f"🔄 MÉTODO DE BALANCEO: {method.upper()}")
        print(f"{'='*60}")
        
        # Aplicar balanceo
        X_balanced, y_balanced = apply_balancing(X_train_selected, y_train, method)
        
        # Guardar información de balanceo
        balancing_info[method] = {
            'original_shape': X_train_selected.shape,
            'balanced_shape': X_balanced.shape,
            'original_distribution': {
                'class_0': sum(y_train == 0),
                'class_1': sum(y_train == 1)
            },
            'balanced_distribution': {
                'class_0': sum(y_balanced == 0),
                'class_1': sum(y_balanced == 1)
            }
        }
        
        # Graficar distribución
        if figures_dir:
            fig = plot_class_distribution(y_train, y_balanced, method)
            plt.savefig(figures_dir / f'class_distribution_{method}.png', bbox_inches='tight')
            plt.show()
        
        # Obtener modelos
        models = get_models_balanced(y_balanced)
        
        # Evaluar modelos
        method_results = {}
        method_train_results = {}
        method_trained_models = {}
        method_overfitting_results = {}
        
        for name, model in models.items():
            model_full_name = f"{name} ({method})"
            print(f'🔍 Evaluando modelo: {model_full_name}')
            
            train_metrics, test_metrics, trained_model, overfitting = evaluate_model_balanced(
                model, X_balanced, y_balanced, 
                X_test_selected, y_test, model_full_name, figures_dir
            )
            
            method_results[name] = test_metrics
            method_train_results[name] = train_metrics
            method_trained_models[name] = trained_model
            method_overfitting_results[name] = overfitting
        
        all_results[method] = method_results
        all_train_results[method] = method_train_results
        all_trained_models[method] = method_trained_models
        all_overfitting_results[method] = method_overfitting_results
    
    # Crear DataFrames de resultados
    print(f"\n📈 ANÁLISIS COMPARATIVO")
    
    # Resultados por método
    results_summary = {}
    
    for method in balancing_methods:
        df = pd.DataFrame.from_dict(all_results[method], orient='index')
        results_summary[method] = df
        print(f"\n🎯 {method.upper()}:")
        print(df.sort_values('roc_auc', ascending=False))
    
    # Crear tablas consolidadas de train vs test para balanceo
    print(f"\n📊 TABLAS COMPARATIVAS TRAIN vs TEST (BALANCEO)")
    
    balanced_comparisons = {}
    
    for method in balancing_methods:
        comparison = create_train_test_comparison_table_balanced(
            all_train_results[method],
            all_results[method],
            method.title()
        )
        balanced_comparisons[method] = comparison
        print(f"\n🎯 TABLA {method.upper()} TRAIN vs TEST:")
        display(comparison)
    
    # Tabla general de balanceo (todos los métodos)
    balanced_general_comparison = pd.concat(list(balanced_comparisons.values()), ignore_index=True)
    
    print(f"\n🎯 TABLA GENERAL BALANCEO TRAIN vs TEST:")
    display(balanced_general_comparison)
    
    # Guardar tablas si hay directorio de salida
    if output_dir:
        balanced_general_comparison.to_csv(output_dir / 'balanced_train_test_comparison.csv', index=False)
        print(f'✅ Tabla comparativa balanceada guardada en: {output_dir / "balanced_train_test_comparison.csv"}')
    
    # Encontrar el mejor modelo general
    best_score = 0
    best_method = None
    best_model_name = None
    
    for method in balancing_methods:
        for model_name, metrics in all_results[method].items():
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_method = method
                best_model_name = model_name
    
    print(f"\n🏆 MEJOR MODELO GENERAL:")
    print(f"Método: {best_method}")
    print(f"Modelo: {best_model_name}")
    print(f"ROC AUC: {best_score:.3f}")
    
    # Guardar el mejor modelo
    if output_dir:
        best_model = all_trained_models[best_method][best_model_name]
        model_path = output_dir / f'best_model_balanced_{best_method}.joblib'
        joblib.dump(best_model, model_path)
        print(f'✅ Mejor modelo guardado en: {model_path}')
        
        # Guardar resultados completos
        results_path = output_dir / 'balanced_modeling_results.csv'
        
        # Crear DataFrame con todos los resultados
        all_results_flat = []
        for method in balancing_methods:
            for model_name, metrics in all_results[method].items():
                row = metrics.copy()
                row['method'] = method
                row['model'] = model_name
                all_results_flat.append(row)
        
        results_df = pd.DataFrame(all_results_flat)
        results_df.to_csv(results_path, index=False)
        print(f'✅ Resultados guardados en: {results_path}')
        
        # Guardar información de balanceo
        import json
        balance_path = output_dir / 'balancing_info.json'
        with open(balance_path, 'w') as f:
            json.dump(balancing_info, f, indent=2)
        print(f'✅ Información de balanceo guardada en: {balance_path}')
    
    return {
        'results': all_results,
        'train_results': all_train_results,
        'overfitting_results': all_overfitting_results,
        'results_summary': results_summary,
        'trained_models': all_trained_models,
        'balancing_info': balancing_info,
        'best_method': best_method,
        'best_model_name': best_model_name,
        'best_score': best_score,
        'selected_features': selected_features,
        'train_test_comparison': balanced_general_comparison,
        'balanced_comparisons': balanced_comparisons,
        'undersampling_comparison': balanced_comparisons.get('undersampling'),
        'oversampling_comparison': balanced_comparisons.get('oversampling'),
        'smote_comparison': balanced_comparisons.get('smote')
    }

def create_train_test_comparison_table_balanced(train_results, test_results, method_name=""):
    """
    Crea una tabla consolidada comparando train vs test para modelos balanceados.
    
    Args:
        train_results: Diccionario con métricas de train
        test_results: Diccionario con métricas de test
        method_name: Nombre del método de balanceo (ej: "Undersampling", "Oversampling", "SMOTE")
        
    Returns:
        DataFrame con tabla comparativa
    """
    comparison_data = []
    
    for model_key in test_results.keys():
        if model_key in train_results:
            model_name = model_key
            
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
                    'Method': method_name,
                    'Process': 'Balanced'
                })
    
    df = pd.DataFrame(comparison_data)
    
    # Reordenar columnas
    df = df[['Model', 'Metric', 'Train', 'Test', 'Difference', '%_Diff', 'Method', 'Process']]
    
    return df

def plot_train_test_comparison_balanced(comparison_df, figsize=(15, 10)):
    """
    Grafica comparación de train vs test para modelos balanceados.
    
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
            ax.set_title(f'Train vs Test - {metric.upper()} (Balanced)')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_balanced_model_comparison(results_summary, figsize=(15, 12)):
    """
    Grafica comparación de modelos por método de balanceo.
    
    Args:
        results_summary: Resumen de resultados por método
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Métricas a graficar
    metrics = ['roc_auc', 'accuracy', 'precision', 'recall']
    titles = ['ROC AUC', 'Accuracy', 'Precision', 'Recall']
    
    methods = list(results_summary.keys())
    colors = ['green', 'blue', 'orange']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Preparar datos para cada método
        x = np.arange(len(results_summary[methods[0]]))
        width = 0.25
        
        for j, method in enumerate(methods):
            df = results_summary[method]
            ax.bar(x + j*width, df[metric], width, 
                   label=method.title(), alpha=0.8, color=colors[j])
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel(title)
        ax.set_title(f'Comparación de {title} por Método de Balanceo')
        ax.set_xticks(x + width)
        ax.set_xticklabels(results_summary[methods[0]].index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
