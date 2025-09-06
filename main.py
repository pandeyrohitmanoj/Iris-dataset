from src.evaluation import evaluate_classification,feature_importance_svc
import kagglehub
from pathlib import Path
# Download latest version
def install_dataset():
    data_path = Path(__file__).parent / 'data'
    print(data_path, data_path.iterdir())
    if any(data_path.iterdir()):
        return
    kagglehub.dataset_download("uciml/iris", path=data_path)

# evaluate_classification()   #run training
install_dataset()
feature_importance_svc()