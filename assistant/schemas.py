
from typing import Optional, List  
from langchain_core.pydantic_v1 import BaseModel, Field


class Condensor(BaseModel):
    "Reformulate the original question ensuring the context and detect the language"

    language : str = Field(description="language detected. Spanish, English, etc")
    condensor : str = Field(description="reformulate it if needed, always in spanish")

class Multiquery(BaseModel):
    """ Genera una lista de preguntas formuladas """
    multiquery_list : str = Field(description="lista de preguntas formuladas")


class Query(BaseModel):
    """ Clase representando una pregunta individual """

    query : str = Field(description="Pregunta o consulta")
    country:  str = Field(description="Lista de nombres de  países válidos y que existan", default="Null")
    region:  str = Field(description="Lista de regiones válidas que existan", default="Null")
    year : str = Field(description="Lista de año o años identificados en la pregunta. Si el país está en inglés, tradúcelo al español. Ejemplo USA = Estados Unidos. El año actual es 2024", default="Null")
    


class Queries(BaseModel):
    """ Clase representando una lista de preguntas """

    queries: List[Query]

class FollowupQuestion(BaseModel):
    """ Class representing a follow-up question """
    followup_question: str = Field(description="The next suggested question to ask")
    followup_question_summary: str = Field(description="A very concise version of the follow-up question")


class Questions(BaseModel):
    """ Class representing a list of questions """
    followup_questions: List[FollowupQuestion]
