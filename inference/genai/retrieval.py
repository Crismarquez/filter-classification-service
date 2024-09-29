from typing import Dict, Optional, List
import os

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizedQuery
)
from langchain_openai import OpenAIEmbeddings

from config.config import ENV_VARIABLES

class CognitiveSearch:
    def __init__(self) -> None:
        
        self.embedding_model = OpenAIEmbeddings(
        openai_api_key=ENV_VARIABLES["OPENAI_KEY"],
        model="text-embedding-3-small",
    )

    async def generate_embeddings(self, text):
        embedding = await self.embedding_model.aembed_query(text)
        return embedding
        
    async def search(
        self,
        semantic_query: str, 
        top: int=5,
        use_hybrid: bool = True,
        **kwargs: Optional[Dict]
    ):
        
        return await self._search(semantic_query, top, use_hybrid, **kwargs)

    async def _search(self, semantic_query: str, 
        top: int=5,
        use_hybrid: bool = True,
        **kwargs: Optional[Dict]):

        self.search_client = SearchClient(
            endpoint=f"https://{ENV_VARIABLES['AZURE_SEARCH_SERVICE']}.search.windows.net",
            index_name=ENV_VARIABLES['AZURE_SEARCH_INDEX'],
            credential=AzureKeyCredential(ENV_VARIABLES['AZURE_SEARCH_KEY']))

        vector = await self.generate_embeddings(semantic_query)

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top,
            fields="main_vector",
)
        
        if use_hybrid:
            documents = self.search_client.search(
                    search_text=semantic_query,
                    vector_queries=[vector_query],
                    top=top,
                    #select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
            )
                    #filter=f"type eq '{type_case}'")
                    
        else:
            documents = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=top,
                #select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                ) #filter=filter_instruction)

        documents_related = []
        for doc in documents:
            documents_related.append({
                "message": doc["message"],
                "label": doc["label"],
                "score": doc["@search.score"]
            })

        return documents_related
   