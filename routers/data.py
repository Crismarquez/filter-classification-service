from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import uuid

from langchain_openai import OpenAIEmbeddings
from azure.cosmos import exceptions
from azure.cosmos.aio import CosmosClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient  

import logging
from schemas.schema import NewKnowledge
from config.config import get_logger, ENV_VARIABLES

logger = get_logger(__name__)

router_data = APIRouter()

# Create the async Cosmos DB client
client = CosmosClient(
    f"https://{ENV_VARIABLES['AZURE_COSMOSDB_ACCOUNT']}.documents.azure.com:443/",
    credential=ENV_VARIABLES["AZURE_COSMOSDB_ACCOUNT_KEY"]
)

# Reference the database and container
database_name = ENV_VARIABLES["AZURE_COSMOSDB_DATABASE"]
container_name = ENV_VARIABLES["AZURE_COSMOSDB_EVALUATIONS_CONTAINER"]


async def get_all_items():
    try:
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        # Query to retrieve all items
        query = "SELECT * FROM c"
        items = [item async for item in container.query_items(query=query)]
        return items

    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"Error fetching items from Cosmos DB: {e}")
        raise


@router_data.get("/metrics", response_model=List[dict])
async def read_items():
    try:
        items = await get_all_items()
        if not items:
            raise HTTPException(status_code=404, detail="No items found.")
        return items
    except Exception as e:
        logger.error(f"Failed to fetch items: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving items from Cosmos DB")

@router_data.post("/continous_training")
async def continous_training(new_knowledge: NewKnowledge):

    search_client = SearchClient(
        endpoint=f"https://{ENV_VARIABLES['AZURE_SEARCH_SERVICE']}.search.windows.net/",
        index_name=ENV_VARIABLES['AZURE_SEARCH_INDEX'],
        credential=AzureKeyCredential(ENV_VARIABLES['AZURE_SEARCH_KEY'])
    )

    embedding_model = OpenAIEmbeddings(
        openai_api_key=ENV_VARIABLES["OPENAI_KEY"],
        model="text-embedding-3-small",
    )

    try:
        record = [
            {
                "id": str(uuid.uuid4()),
                "message": new_knowledge.text,
                "label": new_knowledge.label,
                "source": "continous_training",
                "main_vector": embedding_model.embed_query(new_knowledge.text),
            }
        ]
        result = search_client.upload_documents(documents=record)
        succeeded = sum([1 for r in result if r.succeeded])
        logger.info(f"Uploaded {succeeded} of {len(record)} documents")
    
    except Exception as e:
        logger.error(f"Error uploading documents to Azure Cognitive Search: {e}")
        raise HTTPException(status_code=500, detail="Error uploading documents to Azure Cognitive Search")