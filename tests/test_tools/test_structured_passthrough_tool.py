from typing import List

import pytest
from pydantic import BaseModel, ValidationError

from motleycrew.tools.structured_passthrough import StructuredPassthroughTool


class SampleSchema(BaseModel):
    name: str
    age: int


class B(BaseModel):
    id: int
    value: str
    active: bool = True


class C(BaseModel):
    id: int
    value: str
    active: bool = True


class A(BaseModel):
    member: List[B]


@pytest.fixture
def sample_schema():
    return SampleSchema


@pytest.fixture
def structured_passthrough_tool(sample_schema):
    return StructuredPassthroughTool(schema=sample_schema)


def test_structured_passthrough_tool_initialization(structured_passthrough_tool, sample_schema):
    assert structured_passthrough_tool.schema == sample_schema
    assert structured_passthrough_tool.name == "structured_passthrough_tool"
    assert structured_passthrough_tool.description == "A tool that checks output validity."


def test_structured_passthrough_tool_run_valid_input(structured_passthrough_tool):
    input_data = {"name": "John Doe", "age": 30}
    result = structured_passthrough_tool.run(**input_data)
    assert result.name == "John Doe"
    assert result.age == 30


def test_structured_passthrough_tool_run_invalid_input(structured_passthrough_tool):
    input_data = {"name": "John Doe", "age": "thirty"}
    with pytest.raises(ValidationError):
        structured_passthrough_tool.run(**input_data)


def test_structured_passthrough_tool_post_process(structured_passthrough_tool):
    def post_process(data):
        data.name = data.name.upper()
        return data

    tool_with_post_process = StructuredPassthroughTool(
        schema=structured_passthrough_tool.schema, post_process=post_process
    )

    input_data = {"name": "John Doe", "age": 30}
    result = tool_with_post_process.run(**input_data)
    assert result.name == "JOHN DOE"
    assert result.age == 30


def test_structured_passthrough_tool_post_process_noop(structured_passthrough_tool):
    def post_process(data):
        return data

    tool_with_post_process = StructuredPassthroughTool(
        schema=structured_passthrough_tool.schema, post_process=post_process
    )

    input_data = {"name": "John Doe", "age": 30}
    result = tool_with_post_process.run(**input_data)
    assert result.name == "John Doe"
    assert result.age == 30


def test_structured_passthrough_tool_with_nested_model_instances():
    """Test that StructuredPassthroughTool handles lists of child class instances correctly."""
    tool = StructuredPassthroughTool(schema=A)

    # Create actual instances of BChild (child class of B) instead of dictionaries
    b1 = C(id=1, value="first")
    b2 = C(id=2, value="second", active=False)
    b3 = C(id=3, value="third")  # Uses default extra_field

    # Call run with a list of actual BaseModel child class instances
    result = tool.run(member=[b1, b2, b3])

    # Verify the result
    assert isinstance(result, A)
    assert len(result.member) == 3

    # Verify each member was properly validated
    assert result.member[0].id == 1
    assert result.member[0].value == "first"
    assert result.member[0].active is True

    assert result.member[1].id == 2
    assert result.member[1].value == "second"
    assert result.member[1].active is False

    assert result.member[2].id == 3
    assert result.member[2].value == "third"
    assert result.member[2].active is True
