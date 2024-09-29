import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk

nltk.download('stopwords')

from modeling.utils import load_config
from config.config import get_logger

logger = get_logger(__name__)
config = load_config()

def ingest_data():
    raw_data_path = config['data']['raw_data_path']
    logger.info(f"Cargando datos desde {raw_data_path}")
    # df = pd.read_csv(raw_data_path, sep='\t', names=['label', 'message'])
    df = pd.read_csv(raw_data_path)
    return df

def preprocess_text(text):
    ps = PorterStemmer()
    sms = re.sub('[^a-zA-Z]', ' ', text)
    sms = sms.lower()
    sms = sms.split()
    sms = [ps.stem(word) for word in sms if word not in stopwords.words('english')]
    sms = ' '.join(sms)
    return sms