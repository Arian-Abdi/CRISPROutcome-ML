import pandas as pd
from sklearn.model_selection import train_test_split
from feature_extractor import CRISPRFeaturePipeline
from model import train_models, save_models, calculate_feature_importance

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    
    # Process features
    print("Processing features...")
    pipeline = CRISPRFeaturePipeline()
    processed_df = pipeline.process_data(train_df, is_training=True)
    
    # Define targets
    targets = ['Fraction_Insertions', 'Avg_Deletion_Length', 
               'Indel_Diversity', 'Fraction_Frameshifts']
    
    # Split features and targets
    feature_cols = [col for col in processed_df.columns 
                   if col not in targets + ['Id']]
    X = processed_df[feature_cols]
    y = processed_df[targets]
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\nTraining models...")
    bagged_models, results = train_models(X_train, y_train, X_val, y_val)
    
    # Save models
    save_models(bagged_models, 'trained_bagged_models.joblib')
    
    # Calculate and save feature importance
    importance_dict = calculate_feature_importance(bagged_models, feature_cols)
    
    # Save feature importances
    with pd.ExcelWriter('feature_importance.xlsx') as writer:
        for target, importance_df in importance_dict.items():
            importance_df.to_excel(writer, sheet_name=target, index=False)
    
    print("\nTraining complete!")
    print("\nModel Performance Summary:")
    for target, metrics in results.items():
        print(f"\n{target}:")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        if 'val_r2' in metrics:
            print(f"Validation R²: {metrics['val_r2']:.4f}")
        print(f"CV R²: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

if __name__ == "__main__":
    main()