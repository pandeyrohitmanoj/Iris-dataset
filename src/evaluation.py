import pandas as pd
from .preprocessing import X_test, y_test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from .training import svc_model as model #, knn_model, knn_model, voting_model, linearsvc, svc_model, voting_model
from sklearn.inspection import permutation_importance
from .data_exploration import copy_df
def evaluate_classification():
    """
    Input: grid_search (fitted GridSearchCV object), X_test, y_test, class_names (list)
    Action: Evaluates classification model performance with multiple metrics
    Output: Dictionary with evaluation metrics and plots
    """
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cv_score = model.best_score_
    
    class_report = classification_report(y_test, y_pred,  
                                       output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        if len(set(y_test)) == 2: 
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else: 
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        roc_auc = None
    
    print("=== Model Evaluation Results ===")
    print(f"Best CV Score: {cv_score:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Best Parameters: {model.best_params_}")
    
    if roc_auc:
        print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print(f"Overfitting: {cv_score - accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    results = {
        'cv_score': cv_score,
        'test_accuracy': accuracy,
        'roc_auc': roc_auc,
        'best_params': model.best_params_,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'overfitting': cv_score - accuracy
    }
    
    return results



def feature_importance_svc():
    best_model= model.best_estimator_
    perm_importance = permutation_importance(
        best_model, X_test, y_test, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    # feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_names = copy_df.drop('Species',axis=1).columns
    fig,ax = plt.subplots(figsize=(8,10))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    ax.bar(importance_df['feature'], importance_df['importance'],color='skyblue')
    ax.set_title('Feature importance')
    plt.savefig('importance.png', dpi=300, bbox_inches='tight')
    plt.show()    
    