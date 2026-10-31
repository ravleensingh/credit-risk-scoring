from utils.runtime import configure_runtime_environment

configure_runtime_environment()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline

from utils.modeling import build_preprocessor, compute_permutation_feature_importance, get_train_test_data
from utils.runtime import PROJECT_ROOT

VISUALIZATIONS_DIR = PROJECT_ROOT / 'visualizations'

def plot_confusion_matrices(models_results, y_test):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, result) in enumerate(models_results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/confusion_matrices.png")

def plot_roc_curves(models_results, y_test):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for name, result in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(VISUALIZATIONS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/roc_curves.png")

def plot_metrics_comparison(results_df):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    for idx, model in enumerate(results_df['Model']):
        values = [
            float(results_df.iloc[idx]['Accuracy'].strip('%')) / 100,
            float(results_df.iloc[idx]['Precision']),
            float(results_df.iloc[idx]['Recall']),
            float(results_df.iloc[idx]['F1-Score']),
            float(results_df.iloc[idx]['ROC-AUC'])
        ]
        ax.bar(x + idx * width, values, width, label=model)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/metrics_comparison.png")

def plot_feature_importance(model, feature_names):
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    importances = model.values
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/feature_importance.png")

def compare_models():
    print("="*80)
    print("CREDIT RISK SCORING - MODEL COMPARISON")
    print("="*80)
    
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    }
    
    results = []
    models_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        models_results[name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'model': pipeline
        }
        
        results.append({
            'Model': name,
            'Accuracy': f'{accuracy*100:.2f}%',
            'Precision': f'{precision:.4f}',
            'Recall': f'{recall:.4f}',
            'F1-Score': f'{f1:.4f}',
            'ROC-AUC': f'{roc_auc:.4f}'
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    print("\nGenerating visualizations...")
    plot_confusion_matrices(models_results, y_test)
    plot_roc_curves(models_results, y_test)
    plot_metrics_comparison(results_df)
    
    best_model_name = max(models_results.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_model = models_results[best_model_name]['model']
    feature_contributions = compute_permutation_feature_importance(best_model, X_test, y_test)
    plot_feature_importance(feature_contributions, feature_contributions.index.tolist())
    
    print(f"\n Best Model: {best_model_name}")
    print(f" All visualizations saved to 'visualizations/' folder")
    print("\nComparison complete!")

if __name__ == "__main__":
    compare_models()
