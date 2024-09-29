import time
import uuid
import asyncio
import pandas as pd
from sklearn.metrics import classification_report
from azure.cosmos import CosmosClient, exceptions

from inference.models import ModelManager

from config.config import get_logger, DATA_DIR, ENV_VARIABLES

logger = get_logger(__name__)

# Crear el cliente de Cosmos DB
client = CosmosClient(f"https://{ENV_VARIABLES['AZURE_COSMOSDB_ACCOUNT']}.documents.azure.com:443/", credential=ENV_VARIABLES["AZURE_COSMOSDB_ACCOUNT_KEY"])

# Referenciar la base de datos y el contenedor
database = client.get_database_client(ENV_VARIABLES["AZURE_COSMOSDB_DATABASE"])
container = database.get_container_client(ENV_VARIABLES["AZURE_COSMOSDB_EVALUATIONS_CONTAINER"])

xgboost_manager = ModelManager(model_type="xgboost")
xgboost_manager.load_model()

gpt_4o_manager = ModelManager(model_type="gpt-4o")
gpt_4o_manager.load_model()

gpt_4o_mini_manager = ModelManager(model_type="gpt-4o-mini")
gpt_4o_mini_manager.load_model()

def load_valid_dataset():
    df_valid = pd.read_csv(DATA_DIR / "valid"/ "sample.csv")
    return df_valid

async def evaluate_models():
    df_valid = load_valid_dataset()
    df_valid = df_valid.sample(n=20, random_state=42)
    time_stamp = time.time()
    logger.info(f"Evaluating models...")
    results = []
    for _, row in df_valid.iterrows():
        input_text = row["message"]
        label = row["label"]

        xg_result, gpt_4o_result, gpt_4o_mini_result = await asyncio.gather(
            xgboost_manager.apredict(input_text),
            gpt_4o_manager.apredict(input_text),
            gpt_4o_mini_manager.apredict(input_text)
        )

        results.append({
            "input_text": input_text,
            "label": label,
            "xgboost_pred": xg_result["result"],
            "gpt_4o_pred": gpt_4o_result["result"],
            "gpt_4o_mini_pred": gpt_4o_mini_result["result"],
            "xgboost_results": xg_result,
            "gpt_4o_results": gpt_4o_result,
            "gpt_4o_mini_results": gpt_4o_mini_result,
            "timestamp": time_stamp
        })

    df_results = pd.DataFrame(results)

    xboos_metrics = classification_report(df_results["label"], df_results["xgboost_pred"], output_dict=True)
    gpt_4o_metrics = classification_report(df_results["label"], df_results["gpt_4o_pred"], output_dict=True)
    gpt_4o_mini_metrics = classification_report(df_results["label"], df_results["gpt_4o_mini_pred"], output_dict=True)

    metrics = [
        {"run_time": time_stamp, "model": "xgboost", "metrics": xboos_metrics},
        {"run_time": time_stamp, "model": "gpt-4o", "metrics": gpt_4o_metrics},
        {"run_time": time_stamp, "model": "gpt-4o-mini", "metrics": gpt_4o_mini_metrics}
    ]

    return metrics
    
async def store_db(results):
    try:
        for result in results:
            result['id'] = str(uuid.uuid4())
            # Intentar insertar el documento en Cosmos DB
            container.create_item(body=result)
        logger.info("Result stored in Cosmos DB.")
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"Error storing result in Cosmos DB: {e}")

async def evaluate_and_store():
    metrics = await evaluate_models()
    await store_db(metrics)

if __name__ == "__main__":
    asyncio.run(evaluate_and_store())