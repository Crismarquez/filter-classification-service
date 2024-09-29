from pydantic import BaseModel, Field, validator 
from typing   import Text, Optional

class TextInput(BaseModel):
    text: str

class PredictInputModel(BaseModel):
    text: Optional[str] = Field(None, description="The text input for analysis")
    image: Optional[str] = Field(None, description="Base64 encoded image for analysis (optional)")

class NewKnowledge(BaseModel):
    text: str
    # label shpuldbe a string with values: 'spam' or ham'
    label: str = Field(..., description="The label of the knowledge")