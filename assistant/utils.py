from typing import Optional, List, Dict, TypedDict, Tuple
from itertools import groupby
import uuid

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

async def add_group(classifications, decision_tree):
    for classification in classifications:
        service = classification["service"]
        category = classification["category"]
        subcategory = classification["subcategory"]
        group = decision_tree.get(f"{service}-{category}-{subcategory}", None)
        if not group:
            group = "Not found"
        else:
            group = group["Grupo"]
        classification["group"] = group

    return classifications

def sort_tuples_by_scores(tuples, scores):
    # Combinamos las tuplas y los scores en una lista de tuplas
    combined = list(zip(tuples, scores))
    
    # Ordenamos la lista combinada por score de mayor a menor
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
    # Extraemos las tuplas ya ordenadas
    sorted_tuples = [item[0] for item in sorted_combined]
    
    return sorted_tuples

def filter_documents_by_combinations(documents, target_combinations: Tuple):

    # Filtra los documentos que tienen una combinación que está en el conjunto objetivo
    filtered_documents = [doc for doc in documents if (doc['service'], doc['category'], doc['subcategory']) in target_combinations]
    
    return filtered_documents

def unique_combinations_with_highest_score(documents):
    # Diccionario para almacenar la combinación con el score más alto
    labels = []
    scores = []

    
    # Recorremos cada documento para actualizar el score máximo por combinación
    for doc in documents:
        combination = (doc["service"], doc["category"], doc["subcategory"])
        score = doc["score"]  # Asumimos que el score está en 'doc["score"]'

        # Si la combinación no está en el diccionario o si encontramos un score más alto, actualizamos
        if combination not in labels:
            labels.append(combination)
            scores.append(score)
    
    return labels, scores

def unique_combinations(documents):
    unique_combinations = set()
    
    # Recorremos cada documento y extraemos las combinaciones de service, category y subcategory
    for doc in documents:
        # Creamos una tupla con las tres variables
        combination = (doc["service"], doc["category"], doc["subcategory"])
        # Añadimos la combinación al conjunto, que automáticamente maneja la unicidad
        unique_combinations.add(combination)
    
    # El tamaño del conjunto es el número de combinaciones únicas
    return unique_combinations

def normalize_by_service(context_data, n_max=10):
    context_data.sort(key=lambda x: (x['service'], -x['score']))
    final_documents = []
    for key, group in groupby(context_data, key=lambda x: x['service']):
        final_documents.extend(list(group)[:n_max])
    return final_documents

def normalize_labels(context_data, n_max=10):
    context_data.sort(key=lambda x: (x['service'], x['category'], x['subcategory'], -x['score']))

    # Agrupamos por 'service', 'category' y 'subcategory', y limitamos a 10 entradas por grupo
    final_documents = []
    group_key = lambda x: (x['service'], x['category'], x['subcategory'])
    for key, group in groupby(context_data, key=group_key):
        final_documents.extend(list(group)[:n_max])
    return final_documents

def get_classification_examples(classification_context_data):

    examples = []
    for doc in classification_context_data:
        text = f"Título: {doc['title']}\nDescripción: {doc['description']}"
        examples.append(
            (
                text,
                ClassificationOutput(
                    service=doc["service"],
                    category=doc["category"],
                    subcategory=doc["subcategory"],
                ),
            )
        )
    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
        )
    
    return messages


def get_classification_service_examples(classification_context_data):

    examples = []
    for doc in classification_context_data:
        text = f"Título: {doc['title']}\nDescripción: {doc['description']}"
        examples.append(
            (
                text,
                ClassificationServiceOutput(
                    service=doc["service"]
                ),
            )
        )
    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
        )
    
    return messages

class Calification(BaseModel):
    """Calification options"""
    # Creates a model so that we can extract multiple entities.
    calification: int = Field(..., description="Calification of the case information")
    argument: str = Field(..., description="Argument of the calification")
    user_suggestion: str = Field(..., description="User suggestion of the input text, friendly message motivating the user to provide additional information in order to improve clasification.")

class ClassificationServiceOutput(BaseModel):
    service: str = Field(..., description="Service of the input text")

class ClassificationOutput(BaseModel):
    service: str = Field(..., description="Service of the input text")
    category: str = Field(..., description="Category of the input text")
    subcategory: str = Field(..., description="Subcategory of the input text")

class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted

class Data(BaseModel):
    """Classification options"""

    # Creates a model so that we can extract multiple entities.
    classifications: List[ClassificationOutput]

class DataService(BaseModel):
    """Classification options"""

    # Creates a model so that we can extract multiple entities.
    classifications: List[ClassificationServiceOutput]

def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages