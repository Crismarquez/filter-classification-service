import pickle
from inference.xgboost import XGBoostPredictor
from inference.genai.chains import AssistantClassificator
from config.config import get_logger

logger = get_logger(__name__)

class ModelManager:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.model = None
    
    def load_model(self):
        """
        Load the model based on the model type.
        """
        if self.model_type == "xgboost":
            logger.info("Loading XGBoost model for inference.")
            self.model = XGBoostPredictor()
        elif self.model_type == "gpt-4o":
            logger.info("Loading GPT-4o model for inference.")
            self.model = AssistantClassificator(model_name="gpt-4o")
        elif self.model_type == "gpt-4o-mini":
            logger.info("Loading GPT-4o-mini model for inference.")
            self.model = AssistantClassificator(model_name="gpt-4o-mini")
        else:
            raise ValueError(f"The model type {self.model_type} is not supported.")
    
    def predict(self, input_text):
        """
        Generate the predicion.
        """
        if not self.model:
            raise ValueError("El modelo no ha sido cargado. Llama a load_model() primero.")
        
        return self.model.predict(input_text)
    
    async def apredict(self, input_text):
        """
        Generate the predicion.
        """
        if not self.model:
            raise ValueError("The model is not loaded. Call load_model() first.")

        return await self.model.apredict(input_text)
