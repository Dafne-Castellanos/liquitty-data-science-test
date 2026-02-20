# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# --------------------------
# TRANSFORMADORES PERSONALIZADOS
# --------------------------

class NumericImputerWithFlag(BaseEstimator, TransformerMixin):
    """Numericas -> mediana + flag"""
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        
    def fit(self, X, y=None):
        # Convertir a numpy si es DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        self.imputer.fit(X_array)
        return self
    
    def transform(self, X):
        # Convertir a numpy si es DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
            
        flags = np.isnan(X_array).astype(int)
        X_imputed = self.imputer.transform(X_array)
        return np.hstack([X_imputed, flags])

class BinaryImputerWithFlag(BaseEstimator, TransformerMixin):
    """Binarias -> 0 + flag"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convertir a numpy si es DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
            
        flags = np.isnan(X_array).astype(int)
        X_filled = np.nan_to_num(X_array, nan=0.0)
        return np.hstack([X_filled, flags])

class CategoricalImputerWithFlag(BaseEstimator, TransformerMixin):
    """Categoricas -> 'missing' + flag"""
    def __init__(self, fill_value='missing'):
        self.fill_value = fill_value
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convertir a numpy si es DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
            
        # Para categoricos, usar pandas para detectar NaNs
        if hasattr(X, 'isna'):
            flags = X.isna().astype(int).values
        else:
            # Para arrays, usar una forma segura de detectar NaNs
            flags = np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in X_array.flatten()]).reshape(X_array.shape).astype(int)
        
        # Llenar valores faltantes
        X_filled = np.where(flags == 1, self.fill_value, X_array)
        
        return np.hstack([X_filled, flags])

# --------------------------
# FUNCION PARA CREAR EL PREPROCESSOR
# --------------------------

def create_preprocessor(numeric_features, binary_features, categorical_features):
    """
    Crea un preprocesador con validacion de columnas.
    
    Args:
        numeric_features: Lista de columnas numericas
        binary_features: Lista de columnas binarias  
        categorical_features: Lista de columnas categoricas
    """
    
    def validate_columns(features, feature_type):
        """Valida que las columnas existan y las filtra si es necesario"""
        if features is None:
            return []
        
        # Si es un string, convertir a lista
        if isinstance(features, str):
            features = [features]
            
        return features
    
    # Validar y limpiar listas de columnas
    numeric_features = validate_columns(numeric_features, 'numeric')
    binary_features = validate_columns(binary_features, 'binary')
    categorical_features = validate_columns(categorical_features, 'categorical')
    
    print(f"📊 Columnas numericas: {numeric_features}")
    print(f"📊 Columnas binarias: {binary_features}")
    print(f"📊 Columnas categoricas: {categorical_features}")
    
    # Pipeline numericas continuas
    numeric_transformer = Pipeline([
        ('imputer_flag', NumericImputerWithFlag(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline binarias 0/1 (solo imputacion + flags, sin escalado)
    binary_transformer = Pipeline([
        ('imputer_flag', BinaryImputerWithFlag())
    ])
    
    # Pipeline categoricas
    categorical_transformer = Pipeline([
        ('imputer_flag', CategoricalImputerWithFlag(fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ColumnTransformer con validacion
    transformers = []
    
    if numeric_features:
        print(f"✅ Agregando pipeline numerico para: {numeric_features}")
        transformers.append(('num', numeric_transformer, numeric_features))
        
    if binary_features:
        print(f"✅ Agregando pipeline binario para: {binary_features}")
        transformers.append(('bin', binary_transformer, binary_features))
        
    if categorical_features:
        print(f"✅ Agregando pipeline categorico para: {categorical_features}")
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    if not transformers:
        raise ValueError("No se especificaron columnas validas para el preprocesamiento")
    
    print(f"🔧 Creando ColumnTransformer con {len(transformers)} transformadores...")
    
    try:
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        print("✅ ColumnTransformer creado exitosamente")
        return preprocessor
    except Exception as e:
        print(f"❌ Error creando ColumnTransformer: {e}")
        raise

def manual_preprocessing(X_train, X_test, numeric_features, binary_features, categorical_features):
    """
    Realiza el preprocesamiento manualmente para evitar problemas de sklearn.
    """
    print("🔧 Iniciando preprocesamiento manual...")
    
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    feature_names = []
    
    # 1. Procesar variables numéricas
    if numeric_features:
        print(f"🔄 Procesando numéricas: {numeric_features}")
        
        for col in numeric_features:
            # Crear imputer y flag
            imputer = SimpleImputer(strategy='median')
            
            # Fit en train
            X_train_col = X_train[col].values.reshape(-1, 1)
            imputer.fit(X_train_col)
            
            # Transform train y test
            X_train_imputed = imputer.transform(X_train_col)
            X_test_imputed = imputer.transform(X_test[col].values.reshape(-1, 1))
            
            # Crear flags
            train_flags = np.isnan(X_train[col].values).astype(int).reshape(-1, 1)
            test_flags = np.isnan(X_test[col].values).astype(int).reshape(-1, 1)
            
            # Escalar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Combinar datos y flags
            train_combined = np.hstack([X_train_scaled, train_flags])
            test_combined = np.hstack([X_test_scaled, test_flags])
            
            # Agregar al DataFrame
            X_train_processed[f'{col}_scaled'] = X_train_scaled.flatten()
            X_train_processed[f'{col}_flag'] = train_flags.flatten()
            X_test_processed[f'{col}_scaled'] = X_test_scaled.flatten()
            X_test_processed[f'{col}_flag'] = test_flags.flatten()
            
            feature_names.extend([f'{col}_scaled', f'{col}_flag'])
            
        print(f"✅ Numéricas procesadas")
    
    # 2. Procesar variables binarias
    if binary_features:
        print(f"🔄 Procesando binarias: {binary_features}")
        
        for col in binary_features:
            # Imputar con 0 y crear flags
            train_values = X_train[col].fillna(0).values
            test_values = X_test[col].fillna(0).values
            
            train_flags = X_train[col].isna().astype(int).values
            test_flags = X_test[col].isna().astype(int).values
            
            # Agregar al DataFrame
            X_train_processed[f'{col}_imputed'] = train_values
            X_train_processed[f'{col}_flag'] = train_flags
            X_test_processed[f'{col}_imputed'] = test_values
            X_test_processed[f'{col}_flag'] = test_flags
            
            feature_names.extend([f'{col}_imputed', f'{col}_flag'])
            
        print(f"✅ Binarias procesadas")
    
    # 3. Procesar variables categóricas
    if categorical_features:
        print(f"🔄 Procesando categóricas: {categorical_features}")
        
        for col in categorical_features:
            # Imputar con 'missing'
            train_values = X_train[col].fillna('missing').values
            test_values = X_test[col].fillna('missing').values
            
            # Crear flags
            train_flags = X_train[col].isna().astype(int).values
            test_flags = X_test[col].isna().astype(int).values
            
            # One-hot encoding
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            # Fit en train
            train_encoded = encoder.fit_transform(train_values.reshape(-1, 1))
            test_encoded = encoder.transform(test_values.reshape(-1, 1))
            
            # Obtener nombres de categorías
            categories = encoder.categories_[0]
            encoded_names = [f'{col}_{cat}' for cat in categories]
            
            # Agregar al DataFrame
            for i, name in enumerate(encoded_names):
                X_train_processed[name] = train_encoded[:, i]
                X_test_processed[name] = test_encoded[:, i]
            
            # Agregar flags
            X_train_processed[f'{col}_flag'] = train_flags
            X_test_processed[f'{col}_flag'] = test_flags
            
            feature_names.extend(encoded_names + [f'{col}_flag'])
            
        print(f"✅ Categóricas procesadas")
    
    # Seleccionar solo las columnas procesadas
    X_train_final = X_train_processed[feature_names]
    X_test_final = X_test_processed[feature_names]
    
    print(f"✅ Preprocesamiento completado")
    print(f"📊 Shape final - Train: {X_train_final.shape}, Test: {X_test_final.shape}")
    
    return X_train_final.values, X_test_final.values, feature_names

def debug_preprocessing(X, numeric_features=None, binary_features=None, categorical_features=None):
    """
    Funcion de debugging para el preprocesamiento.
    """
    print("🔍 DEBUG: Iniciando preprocesamiento")
    print(f"📊 Shape de X: {X.shape}")
    print(f"📊 Columnas de X: {list(X.columns)}")
    
    # Validar columnas
    all_features = []
    if numeric_features:
        all_features.extend(numeric_features)
    if binary_features:
        all_features.extend(binary_features)
    if categorical_features:
        all_features.extend(categorical_features)
    
    missing_cols = [col for col in all_features if col not in X.columns]
    if missing_cols:
        print(f"❌ Columnas faltantes: {missing_cols}")
        return None
    
    print(f"✅ Todas las columnas existen")
    
    # Crear preprocesador
    preprocessor = create_preprocessor(numeric_features, binary_features, categorical_features)
    
    try:
        print("🔄 Ajustando preprocesador...")
        preprocessor.fit(X)
        print("✅ Preprocesador ajustado")
        
        print("🔄 Transformando datos...")
        X_transformed = preprocessor.transform(X)
        print(f"✅ Transformacion exitosa: {X_transformed.shape}")
        
        return preprocessor, X_transformed
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        import traceback
        traceback.print_exc()
        return None
