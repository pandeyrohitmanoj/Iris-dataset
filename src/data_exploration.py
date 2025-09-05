import pandas as pd
from pathlib import Path
training_csv_path = Path(__file__).parent.parent / 'data' / 'Iris.csv'
print('somthing',training_csv_path)

train_data = pd.read_csv(training_csv_path)

def get_null_columns_name(pandas_df):
    return pandas_df.columns[pandas_df.isnull().sum() !=0]

def get_datatype_information(df):
    print(f"Dataset shape: {df.shape}")
    print(f"\n\n\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n\n\nData types:\n{df.dtypes}")
    print(f"\n\n\nColumn names: {list(df.columns)}")   
    print(f"\n\n\nNull columns: {get_null_columns_name(df)}")    


get_datatype_information(train_data)
copy_df = train_data.drop('Id',axis=1).drop_duplicates().copy() # copying the training data, and removing the duplicates, and Id column

get_null_columns_name(copy_df)
print(copy_df.head(15))
