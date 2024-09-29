from pydantic import BaseModel, Field, validator 
from typing   import Text, Optional

class TextInput(BaseModel):
    text: str