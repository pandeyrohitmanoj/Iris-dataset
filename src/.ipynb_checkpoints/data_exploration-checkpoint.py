import pandas as pd


train_data = pd.read_csv('../data/Iris.csv')

def get_null_columns_name(pandas_df):
    return pandas_df.columns[train_data.isnull().sum() !=0]

def get_datatype_information(df):
    print(f"Dataset shape: {df.shape}")
    print(f"\n\n\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n\n\nData types:\n{df.dtypes}")
    print(f"\n\n\nColumn names: {list(df.columns)}")   
    print(f"\n\n\nNull columns: {get_null_columns_name(df)}")    


get_datatype_information(train_data)
copy_df = train_data.drop_duplicates().drop('Id').copy()#copying the training data, and 

get_null_columns_name(copy_df)
print(copy_df.head(15))