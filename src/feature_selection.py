# src/feature_selection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# MODELO BASE XGBOOST
# --------------------------

class XGBoostFeatureSelector:
    """
    Selector de características usando XGBoost como modelo base.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Inicializa el selector de características.
        
        Args:
            n_estimators: Número de árboles en XGBoost
            max_depth: Profundidad máxima de los árboles
            learning_rate: Tasa de aprendizaje
            random_state: Semilla aleatoria
            n_jobs: Número de jobs paralelos
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Modelo XGBoost base
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Almacenamiento de resultados
        self.feature_importances_ = None
        self.selected_features_ = None
        self.rfe_selector_ = None
        self.cv_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Ajusta el modelo y calcula importancias.
        
        Args:
            X: Matriz de características
            y: Vector objetivo
            feature_names: Nombres de las características
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Ajustar modelo
        self.model.fit(X, y)
        
        # Obtener importancias
        self.feature_importances_ = self.model.feature_importances_
        
        # Crear DataFrame de importancias
        self.importance_df_ = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        Obtiene las características más importantes.
        
        Args:
            top_n: Número de características a mostrar
            
        Returns:
            DataFrame con características y sus importancias
        """
        if self.importance_df_ is None:
            raise ValueError("El modelo no ha sido ajustado. Llama a fit() primero.")
        
        if top_n:
            return self.importance_df_.head(top_n)
        return self.importance_df_
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Grafica las características más importantes.
        
        Args:
            top_n: Número de características a mostrar
            figsize: Tamaño de la figura
        """
        if self.importance_df_ is None:
            raise ValueError("El modelo no ha sido ajustado. Llama a fit() primero.")
        
        top_features = self.importance_df_.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Características Más Importantes (XGBoost)', fontsize=14, fontweight='bold')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

# --------------------------
# RFE CON XGBOOST
# --------------------------

class XGBoostRFESelector:
    """
    Selector de características usando RFE con XGBoost.
    """
    
    def __init__(self,
                 estimator_params: Dict[str, Any] = None,
                 step: float = 0.1,
                 cv: int = 5,
                 scoring: str = 'roc_auc',
                 random_state: int = 42):
        """
        Inicializa el selector RFE con XGBoost.
        
        Args:
            estimator_params: Parámetros para el estimador XGBoost
            step: Fracción de características a eliminar en cada paso
            cv: Número de folds para validación cruzada
            scoring: Métrica de evaluación
            random_state: Semilla aleatoria
        """
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        
        # Parámetros por defecto para XGBoost
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        if estimator_params:
            default_params.update(estimator_params)
        
        # Crear estimador base
        self.estimator = xgb.XGBClassifier(**default_params)
        
        # Almacenamiento de resultados
        self.rfe_selector_ = None
        self.selected_features_ = None
        self.feature_ranking_ = None
        self.cv_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Ajusta el selector RFE.
        
        Args:
            X: Matriz de características
            y: Vector objetivo
            feature_names: Nombres de las características
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Crear selector RFE
        self.rfe_selector_ = RFE(
            estimator=self.estimator,
            step=self.step
        )
        
        # Ajustar RFE
        self.rfe_selector_.fit(X, y)
        
        # Almacenar resultados
        self.feature_ranking_ = self.rfe_selector_.ranking_
        self.selected_features_ = self.rfe_selector_.support_
        
        # Crear DataFrame de rankings
        self.ranking_df_ = pd.DataFrame({
            'feature': self.feature_names,
            'ranking': self.feature_ranking_,
            'selected': self.selected_features_
        }).sort_values('ranking')
        
        return self
    
    def get_top_features(self, n_features: int) -> Tuple[List[str], np.ndarray]:
        """
        Obtiene las top N características.
        
        Args:
            n_features: Número de características a seleccionar
            
        Returns:
            Tuple con (nombres de características, máscara de selección)
        """
        if self.ranking_df_ is None:
            raise ValueError("El selector no ha sido ajustado. Llama a fit() primero.")
        
        top_features = self.ranking_df_.head(n_features)
        feature_names = top_features['feature'].tolist()
        
        # Crear máscara para las características seleccionadas
        mask = np.zeros(len(self.feature_names), dtype=bool)
        for feature in feature_names:
            idx = self.feature_names.index(feature)
            mask[idx] = True
        
        return feature_names, mask
    
    def plot_rfe_results(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Grafica los resultados de RFE.
        
        Args:
            top_n: Número de características a mostrar
            figsize: Tamaño de la figura
        """
        if self.ranking_df_ is None:
            raise ValueError("El selector no ha sido ajustado. Llama a fit() primero.")
        
        top_features = self.ranking_df_.head(top_n)
        
        plt.figure(figsize=figsize)
        colors = ['green' if selected else 'red' for selected in top_features['selected']]
        sns.barplot(data=top_features, x='ranking', y='feature', palette=colors)
        plt.title(f'Top {top_n} Características por Ranking RFE', fontsize=14, fontweight='bold')
        plt.xlabel('Ranking (1 = mejor)')
        plt.ylabel('Característica')
        
        # Añadir leyenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Seleccionado'),
                         Patch(facecolor='red', label='No seleccionado')]
        plt.legend(handles=legend_elements)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

# --------------------------
# EVALUACIÓN DE CARACTERÍSTICAS
# --------------------------

def evaluate_features(X: np.ndarray, 
                     y: np.ndarray, 
                     feature_names: List[str],
                     cv: int = 5,
                     scoring: str = 'roc_auc') -> pd.DataFrame:
    """
    Evalúa diferentes conjuntos de características usando validación cruzada.
    
    Args:
        X: Matriz de características
        y: Vector objetivo
        feature_names: Nombres de las características
        cv: Número de folds para validación cruzada
        scoring: Métrica de evaluación
        
    Returns:
        DataFrame con resultados de evaluación
    """
    results = []
    
    # Evaluar con todas las características
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    results.append({
        'n_features': len(feature_names),
        'features': 'all',
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'features_list': feature_names
    })
    
    # Evaluar con diferentes números de características
    for n in range(10, len(feature_names), 10):
        if n >= len(feature_names):
            continue
            
        # Usar RFE para seleccionar top n características
        selector = XGBoostRFESelector(cv=3, scoring=scoring)
        selector.fit(X, y, feature_names)
        
        top_features, mask = selector.get_top_features(n)
        X_subset = X[:, mask]
        
        cv_scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring)
        
        results.append({
            'n_features': n,
            'features': f'top_{n}',
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'features_list': top_features
        })
    
    return pd.DataFrame(results)

def plot_feature_evaluation(results_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """
    Grafica los resultados de evaluación de características.
    
    Args:
        results_df: DataFrame con resultados de evaluación
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Gráfico de líneas con barras de error
    plt.errorbar(results_df['n_features'], 
                 results_df['mean_score'], 
                 yerr=results_df['std_score'],
                 marker='o', 
                 linewidth=2, 
                 markersize=8,
                 capsize=5,
                 capthick=2)
    
    plt.xlabel('Número de Características')
    plt.ylabel(f'Puntuación ({results_df.iloc[0]["features_list"][0] if "all" in results_df.iloc[0]["features"] else "ROC AUC"})')
    plt.title('Evaluación de Características vs Número de Features', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

# --------------------------
# FUNCIÓN PRINCIPAL DE SELECCIÓN
# --------------------------

def select_features_with_xgboost(X: np.ndarray,
                                y: np.ndarray,
                                feature_names: List[str],
                                method: str = 'importance',
                                top_n: int = 20,
                                cv: int = 5,
                                plot_results: bool = True) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Función principal para selección de características con XGBoost.
    
    Args:
        X: Matriz de características
        y: Vector objetivo
        feature_names: Nombres de las características
        method: Método de selección ('importance' o 'rfe')
        top_n: Número de características a seleccionar
        cv: Número de folds para validación cruzada
        plot_results: Si True, genera gráficos
        
    Returns:
        Tuple con (nombres seleccionados, máscara, resultados)
    """
    print(f"🔧 Iniciando selección de características con XGBoost...")
    print(f"📊 Método: {method}")
    print(f"📊 Total características: {len(feature_names)}")
    print(f"📊 Objetivo: Top {top_n} características")
    
    results = {
        'method': method,
        'total_features': len(feature_names),
        'selected_count': top_n,
        'feature_names': feature_names
    }
    
    if method == 'importance':
        # Usar importancia de características
        selector = XGBoostFeatureSelector()
        selector.fit(X, y, feature_names)
        
        # Obtener top características
        top_df = selector.get_feature_importance(top_n)
        selected_features = top_df['feature'].tolist()
        
        # Crear máscara
        mask = np.zeros(len(feature_names), dtype=bool)
        for feature in selected_features:
            idx = feature_names.index(feature)
            mask[idx] = True
        
        results['importance_df'] = top_df
        results['selector'] = selector
        
        if plot_results:
            selector.plot_feature_importance(top_n)
            plt.show()
    
    elif method == 'rfe':
        # Usar RFE
        selector = XGBoostRFESelector(cv=cv)
        selector.fit(X, y, feature_names)
        
        selected_features, mask = selector.get_top_features(top_n)
        
        results['ranking_df'] = selector.ranking_df_
        results['selector'] = selector
        
        if plot_results:
            selector.plot_rfe_results(top_n)
            plt.show()
    
    else:
        raise ValueError("Método no soportado. Usa 'importance' o 'rfe'.")
    
    # Evaluar el conjunto seleccionado
    print(f"🔄 Evaluando características seleccionadas...")
    X_selected = X[:, mask]
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='roc_auc')
    
    results['cv_scores'] = cv_scores
    results['mean_cv_score'] = cv_scores.mean()
    results['std_cv_score'] = cv_scores.std()
    results['selected_features'] = selected_features
    results['mask'] = mask
    
    print(f"✅ Selección completada")
    print(f"📊 Características seleccionadas: {len(selected_features)}")
    print(f"📊 Score CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"📊 Top 5 características: {selected_features[:5]}")
    
    return selected_features, mask, results
