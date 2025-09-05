Iris Species: https://www.kaggle.com/datasets/uciml/iris/data

Command to install dependency:
source venv/bin/activate  # Activate env first
pip install -r requirements.txt




Conclusion for Multi-class Classification Model:


1. VotingClassifier achieved best performance - 96.7% test accuracy with lowest overfitting (0.8%), demonstrating ensemble methods' superiority over individual models.
2. All models show minimal overfitting - Difference between CV and test scores under 5% indicates good generalization, suggesting proper model selection and adequate training data.
3. Individual models performed identically - LinearSVC, SVC, and KNN all achieved 93.3% test accuracy with nearly identical classification reports, showing dataset is well-suited for multiple algorithms.
4. Class imbalance handled effectively - All models achieved perfect precision/recall (1.00) for class 0, with classes 1 and 2 showing consistent 89-90% performance across metrics.
5. Polynomial kernel optimal for SVM variants - Both LinearSVC and SVC selected polynomial kernel with degree=3, C=10, and balanced class weights, indicating non-linear decision boundaries in the dataset.



About Dataset:
The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

Id
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
Species



Results:

LinearSVC:

=== Model Evaluation Results ===
Best CV Score: 0.9826
Test Accuracy: 0.9333
Best Parameters: {'model__C': 10, 'model__class_weight': 'balanced', 'model__degree': 3, 'model__gamma': 'scale', 'model__kernel': 'poly'}
Overfitting: 0.0493

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.90      0.90      0.90        10
           2       0.89      0.89      0.89         9

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30



SVC:

=== Model Evaluation Results ===
Best CV Score: 0.9826
Test Accuracy: 0.9333
Best Parameters: {'model__C': 10, 'model__class_weight': 'balanced', 'model__degree': 3, 'model__gamma': 'scale', 'model__kernel': 'poly'}
Overfitting: 0.0493

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.90      0.90      0.90        10
           2       0.89      0.89      0.89         9

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30



KNN:

=== Model Evaluation Results ===
Best CV Score: 0.9830
Test Accuracy: 0.9333
Best Parameters: {'model__metric': 'euclidean', 'model__n_neighbors': 9, 'model__weights': 'uniform'}
ROC AUC Score: 0.9949
Overfitting: 0.0496

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.90      0.90      0.90        10
           2       0.89      0.89      0.89         9

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30

VotingClassifier:

=== Model Evaluation Results ===
Best CV Score: 0.9743
Test Accuracy: 0.9667
Best Parameters: {'model__lr__C': 10, 'model__rf__n_estimators': 100, 'model__svc__C': 10, 'model__voting': 'soft'}
ROC AUC Score: 0.9897
Overfitting: 0.0076

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      0.90      0.95        10
           2       0.90      1.00      0.95         9

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.96        30
weighted avg       0.97      0.97      0.97        30