from pydantic import BaseModel, Field

from jarvis.llm.base import (
    AssistantMessage,
    ImageContent,
    LLMError,
    ToolCall,
    ToolDefinition,
    Usage,
)
from jarvis.utils.errors import JarvisError


class OpenAppInput(BaseModel):
    name: str = Field(description="Application name")
    maximised: bool = False


def test_tool_definition_from_pydantic_model_carries_the_json_schema() -> None:
    tool = ToolDefinition.from_model("open_app", "Open an application", OpenAppInput)

    assert tool.name == "open_app"
    assert tool.input_schema["type"] == "object"
    assert set(tool.input_schema["properties"]) == {"name", "maximised"}
    assert tool.input_schema["required"] == ["name"]


def test_image_content_encodes_standard_base64() -> None:
    image = ImageContent(data=b"\x89PNG\r\n", media_type="image/png")

    assert image.as_base64() == "iVBORw0K"


def test_usage_totals_input_and_output() -> None:
    assert Usage(input_tokens=10, output_tokens=5, cache_read_tokens=100).total_tokens == 15


def test_assistant_message_defaults_to_an_empty_turn() -> None:
    message = AssistantMessage()

    assert message.text == ""
    assert message.tool_calls == ()
    assert message.provider_state == ()


def test_tool_call_arguments_default_to_empty() -> None:
    assert ToolCall(id="c1", name="noop").arguments == {}


def test_llm_error_is_a_jarvis_error() -> None:
    assert issubclass(LLMError, JarvisError)
