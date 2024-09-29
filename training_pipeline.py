import pandas as pd
from sklearn.model_selection import train_test_split

from modeling.data import ingest_data
from modeling.data import preprocess_text
from modeling.feature_engineering import create_features
from modeling.train import train_model
from modeling.eval import evaluate_model
from modeling.utils import load_config

from config.config import get_logger, DATA_DIR

logger = get_logger(__name__)
config = load_config()

def run_training_pipeline():

    df = ingest_data()

    df['cleaned_message'] = df['message'].apply(preprocess_text)

    X = create_features(df['cleaned_message'])
    y = pd.get_dummies(df['label'], drop_first=True).values.ravel()

    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run_training_pipeline()
