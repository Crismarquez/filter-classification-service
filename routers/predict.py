import time
from typing            import Dict
from fastapi           import APIRouter, Request, HTTPException
from fastapi.responses import  JSONResponse
from fastapi.encoders  import jsonable_encoder

from inference.models   import ModelManager
from config.config     import ENV_VARIABLES
from schemas.schema    import TextInput

from config.config import get_logger

logger = get_logger(__name__)


router = APIRouter()

xgboost_manager = ModelManager(model_type="xgboost")
xgboost_manager.load_model()

gpt_4o_manager = ModelManager(model_type="gpt-4o")
gpt_4o_manager.load_model()

gpt_4o_mini_manager = ModelManager(model_type="gpt-4o-mini")
gpt_4o_mini_manager.load_model()

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