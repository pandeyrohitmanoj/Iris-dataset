import kaggle
from pathlib import Path

dir_path = Path(__file__).parent

def install_dataset():
    data_path = dir_path / 'data'
    data_path.mkdir(exist_ok=True)
        
    if any(data_path.iterdir()):
        print('dataset is already downloaded')
        return
    kaggle.api.dataset_download_files("uciml/iris",path='./data', unzip=True)
    print('dataset is downloaded from kaggle')
        

install_dataset()

from src.evaluation import evaluate_classification,feature_importance_svc
evaluate_classification()   #run training

feature_importance_svc()