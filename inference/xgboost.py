import pickle
import re
import uuid

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from langchain_openai import OpenAIEmbeddings

from config.config import ENV_VARIABLES, get_logger

logger = get_logger(__name__)

class XGBoostPredictor:
    def __init__(self, model_path='models/spam_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self._load_model_and_vectorizer()
        
    def _load_model_and_vectorizer(self):
        """
        Load model XGBoost and embedding openai client.
        """
        try:
            logger.info(f"Cargando modelo XGBoost desde {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.vectorizer = OpenAIEmbeddings(
        openai_api_key=ENV_VARIABLES["OPENAI_KEY"],
         model="text-embedding-3-small"
    )
        except Exception as e:
            logger.error(f"Error al cargar el modelo o vectorizador: {e}")
            raise e
    
    def preprocess_text(self, text):
        """
        Preprocesa el texto de entrada para eliminar ruido y normalizar.
        """
        logger.info(f"Preprocesando texto: {text}")
        ps = PorterStemmer()
        sms = re.sub('[^a-zA-Z]', ' ', text)
        sms = sms.lower()
        sms = sms.split()
        sms = [ps.stem(word) for word in sms if word not in stopwords.words('english')]
        sms = ' '.join(sms)
        return sms
    
    async def apredict(self, input_text):
        """
        Prediction model by XGBoost.
        """
        id_prediction = str(uuid.uuid4())
        processed_text = self.preprocess_text(input_text)
        text_vector = self.vectorizer.embed_query(processed_text)
        text_vector = np.array(text_vector)
        text_vector = text_vector.reshape(1, -1)

        prediction = self.model.predict(text_vector)
        result = 'spam' if prediction[0] == 1 else 'ham'
        
        logger.info(f"Prediction result: {result}")
        return {"id_pred": id_prediction, "result": result, "metadata": {"input_text": input_text}}

    def predict(self, input_text):
        """
        Prediction model by XGBoost.
        """
        id_prediction = str(uuid.uuid4())

        processed_text = self.preprocess_text(input_text)
        
        text_vector = self.vectorizer.embed_query(processed_text)

        prediction = self.model.predict(text_vector)
        result = 'spam' if prediction[0] == 1 else 'ham'
        
        logger.info(f"Prediction result: {result}")
        return {"id_pred": id_prediction, "result": result, "metadata": {"input_text": input_text}}
