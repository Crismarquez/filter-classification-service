from typing import Optional, List, Dict, TypedDict, Tuple
import uuid
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

contextualize_q_system_prompt ="""
Your task is twofold:

1. **Contextualize User Queries**: Given a chat history and the most recent user question, which might refer to previous context, formulate a standalone question that can be fully understood without requiring any additional context from the chat history. Do NOT answer the question. If reformulation is needed, do so; otherwise, return the question as is.

2. **Identify the Topic**: Analyze the provided chat history and determine the primary topic discussed. The topic can only be one of the following: "autos" (related to car insurance), "salud" (related to health insurance), or "Null" if the topic cannot be clearly determined as either "autos" or "salud". Return only one of these three values: "autos", "salud", or "Null".

Respond always in Spanish and ensure that your outputs for each task are clear and precise."""

systemprompt_multiquery = """
You are an expert at converting user questions into vector store queries. \
Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.
You can also write a more generic question that needs to be answered in order to answer the specific question.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Return at least 3 versions of the question in Spanish.
"""

systemprompt_followup_q = """"
You are a smart assistant designed to suggest follow-up questions based on a given context and an initial user question. Your task is to generate two follow-up questions that would likely interest the user and can be answered using the provided context.

Instructions:

1. Analyze the context and the user's question.
2. Suggest three follow-up questions that are relevant, interesting, and answerable using the context.
"""

systemprompt_synthesis = """
Eres un asistente ....
"""

        



def get_selfquery_examples():

    examples = [
        (
                "¿Puedes comparar la Region-Andina con la Region-Sur?, -¿Cuál es la comparación entre la Region-Andina y la Region-Sur?, -¿Qué diferencias y similitudes existen entre la Region-Andina y la Region-Sur?",
                [
                    Query(query="Puedes comparar los logros de la Region-Andina", year= 'Null', country="Null", region="Region-Andina"),
                    Query(query="Puedes comparar los beneficios de la Region-Andina", year= 'Null', country="Null", region="Region-Andina"),
                    Query(query="Puedes comparar los logros de la Region-Sur", year= 'Null', country="Null", region="Region-Sur"),
                    Query(query="Puedes comparar los beneficios de la Region-Sur", year= 'Null', country="Null", region="Region-Sur"),
                ],
            ),
        (
                "¿Cuáles son las principales iniciativas del IICA en Haití?, -¿Qué instrumentos de cooperación ha implementado el IICA en Haití?, -¿Qué proyectos de cooperación del IICA están activos en Haití?",
                [
                    Query(query="¿Cuáles son las principales iniciativas del IICA en Haití?", year= 'Null', country="Haití", region="Null"),
                    Query(query="¿Qué instrumentos de cooperación ha implementado el IICA en Haití?", year= 'Null', country="Haití", region="Null"),
                    Query(query="¿Qué proyectos de cooperación del IICA están activos en Haití?", year= 'Null', country="Haití", region="Null"),
                ],
            ),
        (
                "¿Qué logros de los países de la Region-Andina están relacionados con el trabajo del IICA?, -¿Cuáles son los logros de las naciones de la Region-Andina que se pueden atribuir a la colaboración con el IICA?",
                [
                    Query(query="¿Qué logros de los países de la Region-Andina están relacionados con el trabajo del IICA?", year= 'Null', country="Null", region="Region-Andina"),
                    Query(query="¿Cuáles son los logros de las naciones de la Region-Andina que se pueden atribuir a la colaboración con el IICA?", year= 'Null', country="Null", region="Region-Andina")
                ],
            ),
        (
                "¿Con qué organizaciones trabajó conjuntamente el Instituto Interamericano de Cooperación para la Agricultura (IICA) en 2022?, -¿Qué entidades colaboraron con el IICA en el año 2022?",
                [
                    Query(query="¿Con qué organizaciones trabajó conjuntamente el Instituto Interamericano de Cooperación para la Agricultura (IICA) en 2022?", year= 'Null', country="Null", region="Null"),
                    Query(query="¿Qué entidades colaboraron con el IICA en el año 2022?", year= 'Null', country="Null", region="Null")
                ],
            ),
        (
                "Con qué logros cuenta Brasil en 2022?, -Qué logros destacables tiene Brasil en 2022?, -Con qué logros cuenta Colombia en 2023?, -Qué logros destacables tiene Colombia en 2023?",
                [
                    Query(query="Con qué logros cuenta Brasil en 2022?", year= '2022', country="Brasil", region="Null"),
                    Query(query="Qué logros destacables tiene Brasil en 2022?", year= '2022', country="Brasil", region="Null"),
                    Query(query="Con qué logros cuenta Colombia en 2023?", year= '2023', country="Colombia", region="Null"),
                    Query(query="Qué logros destacables tiene Colombia en 2023?", year= '2023', country="Colombia", region="Null"),
                ],
            ),
        (
                "¿Cuáles fueron los logros de Brasil en 2022?, -¿Qué metas alcanzó Brasil en 2022?, -¿Cuáles fueron los logros de la Region-Norte?, -¿Qué metas alcanzó la Region-Norte?",
                [
                    Query(query="¿Cuáles fueron los logros de Brasil en 2022?", year= '2022', country="Brasil", region="Null"),
                    Query(query="¿Qué metas alcanzó Brasil en 2022?", year= '2022', country="Brasil", region="Null"),
                    Query(query="¿Cuáles fueron los logros de la Region-Norte?", year= 'Null', country="Null", region="Region-Norte"),
                    Query(query="¿Qué metas alcanzó la Region-Norte?", year= 'Null', country="Null", region="Region-Norte")
                ],
            ),
        (
                "¿Qué éxitos ha alcanzado el campo femenino?, -¿Cuáles son los principales logros obtenidos en el campo femenino?, -¿Qué hitos importantes se han logrado en el campo femenino?",
                [
                    Query(query="¿Qué éxitos ha alcanzado el campo femenino?", year= 'Null', country="Null", region="Null"),
                    Query(query="¿Cuáles son los principales logros obtenidos en el campo femenino?", year= 'Null', country="Null", region="Null"),
                    Query(query="¿Qué hitos importantes se han logrado en el campo femenino?", year= 'Null', country="Null", region="Null"),
                ],
            ),
        (
                "¿Puedes comparar la agricultura de Colombia con la de la Region-Norte?, -¿Cómo se compara la agricultura de Colombia con la de la Region-Norte?, -¿Cuáles son las diferencias y similitudes entre la agricultura de Colombia y la de la Region-Norte?",
                [
                    Query(query="¿Puedes comparar la agricultura de Colombia con la de la Region-Norte?", year= 'Null', country="Colombia", region="Null"),
                    Query(query="¿Puedes comparar la agricultura de Colombia con la de la Region-Norte?", year= 'Null', country="Null", region="Region-Norte"),
                    Query(query="¿Cómo se compara la agricultura de Colombia con la de la Region-Norte?", year= 'Null', country="Colombia", region="Null"),
                    Query(query="¿Cómo se compara la agricultura de Colombia con la de la Region-Norte?", year= 'Null', country="Null", region="Region-Norte"),
                ],
            ),
        (
                "Cuáles son las diferencias y similitudes entre la Region-Andina y la Region-Sur?, -Qué características comparten la Region-Andina y la Region-Sur?",
                [
                    Query(query="Cuáles son las diferencias y similitudes entre la Region-Andina y la Region-Sur?", year= 'Null', country="Null", region="Region-Andina"),
                    Query(query="Cuáles son las diferencias y similitudes entre la Region-Andina y la Region-Sur?", year= 'Null', country="Null", region="Region-Sur"),
                    Query(query="Qué características comparten la Region-Andina y la Region-Sur?", year= 'Null', country="Null", region="Region-Andina"),
                    Query(query="Qué características comparten la Region-Andina y la Region-Sur?", year= 'Null', country="Null", region="Region-Sur")
                ],
            ),
        # (
        #         "",
        #         [
        #             Query(query="", year= 'Null', country="Null", region="Null"),
        #             Query(query="", year= 'Null', country="Null", region="Null"),
        #             Query(query="", year= 'Null', country="Null", region="Null"),
        #         ],
        #     ),
    ]
    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": tool_call})
        )
    
    return messages

class Query(BaseModel):
    """ Clase representando una pregunta individual """

    query : str = Field(description="Pregunta o consulta")
    country:  str = Field(description="Lista de nombres de  países válidos y que existan", default="Null")
    region:  str = Field(description="Lista de regiones válidas que existan", default="Null")
    year : str = Field(description="Lista de año o años identificados en la pregunta. Si el país está en inglés, tradúcelo al español. Ejemplo USA = Estados Unidos. El año actual es 2024", default="Null")
    

class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted

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