from typing import Dict, Optional, List
import os

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizedQuery
)
from langchain_openai import AzureOpenAIEmbeddings

from config.config import ENV_VARIABLES, TYPE_INDEX_ALLOWED

class CognitiveSearch:
    def __init__(self) -> None:
        
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com",
            openai_api_key=ENV_VARIABLES["AZURE_OPENAI_API_KEY"],
            azure_deployment=ENV_VARIABLES["AZURE_OPENAI_EMBEDDING"],
            openai_api_version="2023-05-15",
)

    async def generate_embeddings(self, text):
        embedding = await self.embedding_model.aembed_query(text)
        return embedding
        
    async def search(
        self,
        type_index: str,
        semantic_query: str, 
        top: int=5,
        use_hybrid: bool = True,
        refine_by_service: bool = False,
        **kwargs: Optional[Dict]
    ):
        
        if type_index not in TYPE_INDEX_ALLOWED:
            raise ValueError(f"Type index {type_index} not allowed. Allowed types: {TYPE_INDEX_ALLOWED}")

        if type_index == "classification":
            return await self._search_classification(semantic_query, top, use_hybrid, refine_by_service, **kwargs)
        elif type_index == "calification":
            return await self._search_calification(semantic_query, top, use_hybrid)


    async def _search_classification(self, semantic_query: str, 
        top: int=5,
        use_hybrid: bool = True,
        refine_by_service: bool = False,
        **kwargs: Optional[Dict]):

        self.search_client = SearchClient(
            endpoint=f"https://{ENV_VARIABLES['AZURE_SEARCH_SERVICE']}.search.windows.net",
            index_name=ENV_VARIABLES['AZURE_SEARCH_INDEX_CLASSIFICATION'],
            credential=AzureKeyCredential(ENV_VARIABLES['AZURE_SEARCH_KEY']))

        vector = await self.generate_embeddings(semantic_query)

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top,
            fields="content_vector",
)
        
        type_case = kwargs.get("type")
        if use_hybrid:
            if type_case:
                if kwargs.get("select_service"):
                    select_service = " or ".join([f"service eq '{service}'" for service in kwargs.get("select_service")])
                    filter_instruction = f"type eq '{type_case}' and ({select_service})" # f"type_dataset eq 'train' and type eq '{type_case}' and ({select_service})"
                else:
                    filter_instruction = f"type eq '{type_case}'" # f"type_dataset eq 'train' and type eq '{type_case}'"
                documents = self.search_client.search(
                    search_text=semantic_query,
                    vector_queries=[vector_query],
                    top=top,
                    select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                    filter=filter_instruction)
                    #filter=f"type eq '{type_case}'")
            else:
                if kwargs.get("select_service"):
                    select_service = " or ".join([f"service eq '{service}'" for service in kwargs.get("select_service")])
                    filter_instruction = f"{select_service}" # f"type_dataset eq 'train' and ({select_service})"
                else:
                    filter_instruction = "type_dataset eq 'train'"
                documents = self.search_client.search(
                    search_text=semantic_query,
                    vector_queries=[vector_query],
                    top=top,
                    select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                    )#filter=filter_instruction) # only search in train data
                #)
                    
        else:
            if kwargs.get("select_service"):
                select_service = " or ".join([f"service eq '{service}'" for service in kwargs.get("select_service")])
                filter_instruction = f"type_dataset eq 'train' and ({select_service})"
            else:
                filter_instruction = "type_dataset eq 'train'"
            documents = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=top,
                select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                filter=filter_instruction) # only search in train data

        documents_related = []
        for doc in documents:
            documents_related.append({
                "record_id": doc["record_id"],
                "type_dataset": doc["type_dataset"],
                "title": doc["title"],
                "description": doc["description"],
                "type": doc["type"],
                "service": doc["service"],
                "category": doc["category"],
                "subcategory": doc["subcategory"],
                "score": doc["@search.score"]
            })

        if refine_by_service:
            services_related = [doc["service"] for doc in documents_related]
            unique_services = list(set(services_related))

            type_case = kwargs.get("type")
            if use_hybrid:
                filter_by_services = " or ".join([f"service eq '{service}'" for service in unique_services])
                if type_case:
                    documents = self.search_client.search(
                        search_text=semantic_query,
                        vector_queries=[vector_query],
                        top=top,
                        select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                        #filter=f"type eq '{type_case}' and ({filter_by_services})")
                        filter=f"type_dataset eq 'train' and type eq '{type_case}' and ({filter_by_services})")
                else:
                    documents = self.search_client.search(
                        search_text=semantic_query,
                        vector_queries=[vector_query],
                        top=top,
                        select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                        filter=f"type_dataset eq 'train' and ({filter_by_services})") # only search in train data
                        
            else:
                documents = self.search_client.search(
                    search_text=None,
                    vector_queries=[vector_query],
                    top=top,
                    select=["record_id", "type_dataset", "title", "description", "type", "service", "category", "subcategory"],
                    filter=f"type_dataset eq 'train' and ({filter_by_services})") # only search in train data

            documents_related = []
            for doc in documents:
                documents_related.append({
                    "record_id": doc["record_id"],
                    "type_dataset": doc["type_dataset"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "type": doc["type"],
                    "service": doc["service"],
                    "category": doc["category"],
                    "subcategory": doc["subcategory"],
                    "score": doc["@search.score"]
                })
            return documents_related
        else:
            return documents_related
    
    async def _search_calification(self, semantic_query: str, 
        top: int=5,
        use_hybrid: bool = True):

        self.search_client = SearchClient(
            endpoint=f"https://{os.environ['AZURE_SEARCH_SERVICE']}.search.windows.net",
            index_name=os.environ['AZURE_SEARCH_INDEX_CALIFICATION'],
            credential=AzureKeyCredential(os.environ['AZURE_SEARCH_KEY']))

        vector = await self.generate_embeddings(semantic_query)
        if use_hybrid:
            documents = self.search_client.search(
                search_text=semantic_query,
                vectors= [vector],
                top=top,
                filter="type_data eq 'train'") # only search in train data
                    
        else:
            documents = self.search_client.search(
                search_text=None,
                vectors= [vector],
                top=top,
                filter="type_data eq 'train'") # only search in train data

        return documents