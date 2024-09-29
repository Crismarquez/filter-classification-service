from sklearn.metrics import classification_report
from config.config import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluando el modelo...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Reporte de evaluación:\n{report}")
    return report