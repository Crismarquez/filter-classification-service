from typing import Dict, Tuple
from pathlib import Path
import json
from datetime import datetime
import random
import asyncio
import time

from assistant.agents import AssistantClassificator

class DataLoader:
    def __init__(self, file_dir: Path) -> None:
        self.file_dir = file_dir

    def load_data_jsonl(self) -> list:
        with open(self.file_dir, 'r') as archivo:
            dataset = [json.loads(line) for line in archivo]

        return dataset

    def load_data(self) -> list:
        with open(self.file_dir, 'r') as archivo:
            dataset = json.load(archivo)

        return dataset
    
    def _conversation_format(self, session: dict) -> list:
        return [{"role": "user", "content": session["question"]}]
    

class Evaluator:
    def __init__(self, dataloader, model_pipeline: AssistantClassificator):

        self.dataloader = dataloader
        self.assistant = model_pipeline

    async def run_evaluation(self, size_sample: int = 100) -> None:
        dataset = self.dataloader.load_data_jsonl()
        if len(dataset) >= size_sample:
            sample_size = size_sample
        else:
            sample_size = len(dataset)
            print(f"La lista tiene menos de {size_sample} elementos. Seleccionando todos los {sample_size} elementos.")

        sampled_data = random.sample(dataset, sample_size)
        # sampled_data = dataset[:10]

        async def evaluate_batch(batch):
            tasks = [self.assistant.a_run(session) for session in batch]
            results = await asyncio.gather(*tasks)
            for session, result in zip(batch, results):
                session["result"] = result
                
        # # Dividir sampled_data en lotes de 4
        # batches = [sampled_data[i:i + 4] for i in range(0, len(sampled_data), 4)]
        
        # # Evaluar cada lote de forma secuencial (puedes también hacer esto concurrente si es necesario)
        # for batch in batches:
        #     #time.sleep(1)  # for limit quota
        #     try:
        #         await evaluate_batch(batch)
        #     except Exception as e:
        #         print(f"Error evaluating batch: {e}")
        #         print("Assuming quota limit reached. Sleeping for 1 minute.")
        #         time.sleep(60)
        #         await evaluate_batch(batch)

        for session in sampled_data:
            try:
                session["result"] = await self.assistant.a_run(session)
            except Exception as e:
                print(f"Error evaluating session: {e}")
                print("parsing error")
                continue
        # for session in sampled_data:
        #     try:
        #         session["result"] = await self.assistant.a_run(session)
        #     except Exception as e:
        #         print(f"Error evaluating session: {e}")
        #         print("Assuming quota limit reached. Sleeping for 1 minute.")
        #         time.sleep(20)
        #         session["result"] = await self.assistant.a_run(session)

        return sampled_data
