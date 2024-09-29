from typing import Optional
import asyncio
import base64

from fastapi import  APIRouter, Request, File, UploadFile, Form, HTTPException

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
async def predict(request: Request):
    """
    Predict using all models.
    """
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
        return {
                "xgboost": xg_result,
                "gpt-4o": gpt_4o_result,
                "gpt-4o-mini": gpt_4o_mini_result,
                "image_analysis": image_result,
            }
    except Exception as e:
        logger.error(f"Error en predicción de spam: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción de spam")