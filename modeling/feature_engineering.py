
from langchain_openai import OpenAIEmbeddings

from config.config import ENV_VARIABLES


def create_features(corpus):
    embedding_model = OpenAIEmbeddings(
        openai_api_key=ENV_VARIABLES["OPENAI_KEY"],
        model="text-embedding-3-small",
    )
    corpus_embeddings = embedding_model.embed_documents(corpus)
    return corpus_embeddings