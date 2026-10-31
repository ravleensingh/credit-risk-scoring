from utils.runtime import configure_runtime_environment

configure_runtime_environment()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from utils.modeling import (
    MODEL_MAX_DEPTH,
    MODEL_N_ESTIMATORS,
    MODEL_RANDOM_STATE,
    compute_permutation_feature_importance,
    train_random_forest_pipeline,
)
from utils.preprocessing import load_data
from utils.runtime import PROJECT_ROOT

VISUALIZATIONS_DIR = PROJECT_ROOT / 'visualizations'


def plot_confusion_matrix(y_test, y_pred, accuracy):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/final_confusion_matrix.png")

def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'final_roc_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/final_roc_curve.png")

def plot_feature_importance(model, feature_names):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    importances = model.values
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'final_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/final_feature_importance.png")

def train_model():
    print("Loading data...")
    df = load_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['loan_paid_back'].value_counts()}")
    
    print("\nTraining Random Forest model...")
    artifacts = train_random_forest_pipeline(save_artifact=True)
    model = artifacts.pipeline.named_steps['model']
    
    print("Evaluating model...")
    y_test = artifacts.y_test
    y_pred = artifacts.y_pred
    y_pred_proba = artifacts.y_pred_proba
    accuracy = artifacts.accuracy
    roc_auc = artifacts.roc_auc
    
    print(f"\nModel Accuracy: {accuracy*100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Default', 'Paid Back']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred, accuracy)
    plot_roc_curve(y_test, y_pred_proba, roc_auc)
    feature_contributions = compute_permutation_feature_importance(
        artifacts.pipeline,
        artifacts.X_test,
        artifacts.y_test,
    )
    plot_feature_importance(feature_contributions, feature_contributions.index.tolist())
    
    print("\nSaving model parameters...")
    top_features = feature_contributions.head(10)
    with (PROJECT_ROOT / 'models' / 'model_params.py').open('w', encoding='utf-8') as f:
        f.write(f'model_type = "RandomForest"\n')
        f.write(f'n_estimators = {MODEL_N_ESTIMATORS}\n')
        f.write(f'max_depth = {MODEL_MAX_DEPTH}\n')
        f.write(f'random_state = {MODEL_RANDOM_STATE}\n\n')
        f.write(f'feature_names = {feature_contributions.index.tolist()!r}\n')
        f.write('feature_importances = {\n')
        for feature_name, value in feature_contributions.items():
            f.write(f'    "{feature_name}": {value},\n')
        f.write('}\n\n')
        f.write(f'top_features = {list(top_features.items())!r}\n')
        f.write(f'accuracy = {accuracy}\n')
        f.write(f'roc_auc = {roc_auc}\n')
    
    print("Training complete!")
    return accuracy, roc_auc

if __name__ == "__main__":
    train_model()
