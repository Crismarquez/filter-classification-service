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


class Calification(BaseModel):
    """Calification options"""
    # Creates a model so that we can extract multiple entities.
    calification: int = Field(..., description="Calification of the case information")
    argument: str = Field(..., description="Argument of the calification")
    user_suggestion: str = Field(..., description="User suggestion of the input text, friendly message motivating the user to provide additional information in order to improve clasification.")


class ClassificationOutput(BaseModel):
    Classification: str = Field(..., description="Determine if the input text is spam or not, sould be either 'spam' or 'ham'")
    Explanation: str = Field(..., description="Explain why the input text is spam or ham")

class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def get_classification_examples(classification_context_data):

    examples = []
    for doc in classification_context_data:
        text = f"message: {doc['message']}"
        examples.append(
            (
                text,
                ClassificationOutput(
                    Classification=doc["label"],
                    Explanation="[include the explanation]",
                ),
            )
        )
    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
        )
    
    return messages


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