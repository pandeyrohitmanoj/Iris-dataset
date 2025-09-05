from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .data_exploration import copy_df

def create_model(model,paramgrid, numerical_feature_array=[], category_feature_array=[]):
    ct_array = []
    if len(numerical_feature_array):
        ct_array.append(('numeric',StandardScaler(),numerical_feature_array))
    if len(category_feature_array):
        ct_array.append(('category',OneHotEncoder(),category_feature_array))
    preprocessor = ColumnTransformer(ct_array)
    pipeline = Pipeline([
        ('prepocessor',preprocessor),
        ('model',model)
    ])
    grid_search = GridSearchCV(
        pipeline,
        param_grid=paramgrid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    return grid_search      
    
one_encoder = OneHotEncoder()
scaler = StandardScaler()
robust_scaler = RobustScaler()
ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()

ct = ColumnTransformer([
    ('category',scaler,['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
])


copy_df['Species'] = label_encoder.fit_transform(copy_df['Species'])
X_transformed = copy_df.drop(columns=['Species','Id'],axis=1)
y_encoded = copy_df['Species']


X_train, X_test, y_train, y_test = train_test_split(X_transformed,y_encoded,test_size=0.2,random_state=42)




