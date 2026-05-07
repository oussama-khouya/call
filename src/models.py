# ABOUTME: Pydantic data models for function definitions, test prompts,
# ABOUTME: and results. All validation is handled by pydantic.

from typing import Any
from pydantic import BaseModel, field_validator


class ParameterDefinition(BaseModel):
    """Definition of a single function parameter.

    Attributes:
        type: The type of the parameter ('number', 'string', 'boolean').
    """

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure the parameter type is one of the allowed types."""
        allowed = {"number", "string", "boolean"}
        if v not in allowed:
            raise ValueError(
                f"Invalid parameter type: {v}. Must be one of {allowed}"
            )
        return v


class ReturnDefinition(BaseModel):
    """Definition of a function's return type.

    Attributes:
        type: The return type of the function.
    """

    type: str


class FunctionDefinition(BaseModel):
    """Schema for a callable function exposed to the LLM.

    Attributes:
        name: Unique function identifier (e.g. 'fn_add_numbers').
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their type definitions.
        returns: The return type definition.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class TestPrompt(BaseModel):
    """A single test prompt from the input file.

    Attributes:
        prompt: The natural language request to process.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """The output for a single prompt: function to call and with what args.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function to call.
        parameters: The arguments to pass, with correct types.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
