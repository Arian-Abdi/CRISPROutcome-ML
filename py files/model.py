import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

def create_base_models():
    """Create base LightGBM models with optimal parameters."""
    return {
        'Fraction_Insertions': lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            random_state=42,
            verbose=-1
        ),
        'Avg_Deletion_Length': lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            random_state=42,
            verbose=-1
        ),
        'Indel_Diversity': lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            random_state=42,
            verbose=-1
        ),
        'Fraction_Frameshifts': lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
    }

def train_models(X_train, y_train, X_val=None, y_val=None):
    """Train bagged models for all targets."""
    base_models = create_base_models()
    bagged_models = {}
    results = {}
    
    for target, base_model in base_models.items():
        print(f"\nTraining model for {target}")
        
        # Create bagging wrapper
        bagged_model = BaggingRegressor(
            base_estimator=base_model,
            n_estimators=15,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Cross validation
        cv_scores = cross_val_score(bagged_model, X_train, y_train[target], cv=5)
        print(f"CV R² scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train model
        bagged_model.fit(X_train, y_train[target])
        bagged_models[target] = bagged_model
        
        # Calculate metrics
        train_pred = bagged_model.predict(X_train)
        train_r2 = r2_score(y_train[target], train_pred)
        
        metrics = {
            'train_r2': train_r2,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = bagged_model.predict(X_val)
            metrics['val_r2'] = r2_score(y_val[target], val_pred)
            
        results[target] = metrics
        
        print(f"Training R²: {train_r2:.4f}")
        if 'val_r2' in metrics:
            print(f"Validation R²: {metrics['val_r2']:.4f}")
    
    return bagged_models, results

def calculate_feature_importance(bagged_models, feature_names):
    """Calculate feature importance for all models."""
    importance_dict = {}
    
    for target, model in bagged_models.items():
        n_features = len(feature_names)
        importance_scores = np.zeros(n_features)
        feature_counts = np.zeros(n_features)
        
        for estimator in model.estimators_:
            feature_indices = model.estimators_features_[
                model.estimators_.index(estimator)
            ]
            importance_scores[feature_indices] += estimator.feature_importances_
            feature_counts[feature_indices] += 1
        
        feature_counts = np.maximum(feature_counts, 1)
        importance_scores /= feature_counts
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        importance_dict[target] = importance_df
    
    return importance_dict

def save_models(bagged_models, filename='trained_bagged_models.joblib'):
    """Save trained models."""
    import joblib
    joblib.dump(bagged_models, filename)
    print(f"Models saved to {filename}")

def load_models(filename='trained_bagged_models.joblib'):
    """Load trained models."""
    import joblib
    return joblib.load(filename)