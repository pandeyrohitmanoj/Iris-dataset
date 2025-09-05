
from .preprocessing import create_model, X_train,y_train
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

numerical_columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

# Linear SVC classification: 0.958 (CV score)
param_grid_linearsvc = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l2'],
    'model__loss': ['hinge'],  # hinge works with dual=True
    'model__dual': [True],
    'model__class_weight': ['balanced', None],
}

# linearsvc = create_model(LinearSVC(),param_grid_linearsvc,numerical_columns)
# linearsvc.fit(X_train,y_train)



# knn classification: 0.987 (CV score)
param_grid_knn = {
    'model__n_neighbors': [5, 7, 9, 11, 13], 
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan'],    
}
# knn_model = create_model(KNeighborsClassifier(),param_grid_knn,numerical_columns)
# knn_model.fit(X_train,y_train)


param_grid_svc = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],           
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], 
    'model__degree': [2, 3, 4, 5],                         
    'model__class_weight': ['balanced', None],     
}
svc_model = create_model(SVC(),param_grid_svc,numerical_columns)
svc_model.fit(X_train,y_train)



# SVC: 0.967 (CV score)
param_grid_voting = {
   'model__voting': ['hard', 'soft'],
   'model__rf__n_estimators': [50, 100],
   'model__svc__C': [0.1, 1, 10],
   'model__lr__C': [0.1, 1, 10],
}
# voting_model = create_model(VotingClassifier(
#     estimators=[
#         ('rf', RandomForestClassifier()),
#         ('svc', SVC(probability=True)), 
#         ('lr', LogisticRegression())
#     ],
#     voting='hard'  
# ),param_grid_voting,numerical_columns)
# voting_model.fit(X_train,y_train)