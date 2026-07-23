"""this file is for the types validation using
                pydantic BaseModel"""

from typing import Any
# pyrefly: ignore [missing-import]
from pydantic import BaseModel


#  "parameters": {"a": {"type": "number"},
class parameterdef(BaseModel):
    type: str


# then now the func def
class functiondef(BaseModel):
    name: str
    description: str
    parameters: dict[str, parameterdef]
    returns: parameterdef


# then now function calling test "prompt"
class prompt(BaseModel):
    prompt: str


# FunctionCallResult
class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
