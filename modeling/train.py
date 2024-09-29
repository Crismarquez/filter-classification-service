import pickle
import xgboost as xgb
import mlflow
import mlflow.xgboost

from modeling.utils import load_config
from config.config import get_logger


logger = get_logger(__name__)
config = load_config()

def train_model(X_train, y_train):
    params = config['model']['parameters']
    with mlflow.start_run():
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        # Registrar el modelo y parámetros en MLflow
        mlflow.log_params(params)
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")
    # Guardar el modelo localmente
    model_output_path = config['training']['model_output_path']
    with open(f"{model_output_path}/spam_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    return model
