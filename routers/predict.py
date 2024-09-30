from typing import Optional, Dict
from datetime import datetime
import uuid
import asyncio
import base64

from fastapi import  APIRouter, Request, BackgroundTasks, HTTPException
from azure.cosmos import CosmosClient

from inference.models   import ModelManager
from inference.multimodal import ImageAnalyser
from config.config     import ENV_VARIABLES
from schemas.schema    import TextInput, PredictInputModel

from config.config import get_logger

logger = get_logger(__name__)


router = APIRouter()

xgboost_manager = ModelManager(model_type="xgboost")
xgboost_manager.load_model()

gpt_4o_manager = ModelManager(model_type="gpt-4o")
gpt_4o_manager.load_model()

gpt_4o_mini_manager = ModelManager(model_type="gpt-4o-mini")
gpt_4o_mini_manager.load_model()

multimodal_analyser = ImageAnalyser()

async def send_data_to_cosmos(data: Dict):
    client = CosmosClient(f"https://{ENV_VARIABLES['AZURE_COSMOSDB_ACCOUNT']}.documents.azure.com:443/", credential=ENV_VARIABLES["AZURE_COSMOSDB_ACCOUNT_KEY"])
    database = client.get_database_client(ENV_VARIABLES["AZURE_COSMOSDB_DATABASE"])
    container = database.get_container_client(ENV_VARIABLES["AZURE_COSMOSDB_MONITORING_CONTAINER"])
    container.create_item(body=data)
    print("Item saved successfully")

@router.post("/xgboost/predict")
async def predict_xgboost(request: Request, input_text: TextInput):
    """
    Predict using XGBoost model.
    """
    try:
        result = await xgboost_manager.apredict(input_text.text)
        return result
    except Exception as e:
        logger.error(f"Error en predicción de spam: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de spam")
    
@router.post("/generative/gpt-4o")
async def predict_gpt_4o(request: Request, input_text: TextInput):
    """
    Predict using GPT-4o model.
    """
    try:
        result = await gpt_4o_manager.apredict(input_text.text)
        return result
    except Exception as e:
        logger.error(f"Error en predicción de spam: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de spam")
    
@router.post("/generative/gpt-4o-mini")
async def predict_gpt_4o_mini(request: Request, input_text: TextInput):
    """
    Predict using GPT-4o-mini model.
    """
    try:
        result = await gpt_4o_mini_manager.apredict(input_text.text)
        return result
    except Exception as e:
        logger.error(f"Error en predicción de spam: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de spam")
    
# predict with all models
@router.post("/predict")
async def predict(request: Request, background_tasks: BackgroundTasks):
    """
    Predict using all models.
    """
    pred_id = str(uuid.uuid4()) #predId
    data = await request.json()
    try:
        
        text = data.get('text', None)
        image_base64 = data.get('image', None) # This can be None if not provided

        # If an image is provided, validate that it is a valid Base64 string
        if image_base64:
            try:
                # Decode to verify the image is a valid base64 string
                base64.b64decode(image_base64)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid image encoding")

        xg_result, gpt_4o_result, gpt_4o_mini_result, image_result = await asyncio.gather(
            xgboost_manager.apredict(text),
            gpt_4o_manager.apredict(text),
            gpt_4o_mini_manager.apredict(text),
            multimodal_analyser.apredict(image_base64)
        )


        pred_results = {
                "id": str(uuid.uuid4()),
                "predId": pred_id,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "xgboost": xg_result,
                "gpt-4o": gpt_4o_result,
                "gpt-4o-mini": gpt_4o_mini_result,
                "image_analysis": image_result,
            }
        
        background_tasks.add_task(send_data_to_cosmos, pred_results)
        return pred_results
    
    except Exception as e:
        logger.error(f"Error en predicción de spam: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de spam")