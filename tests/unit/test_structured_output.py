"""Unit tests for StructuredOutputConfig class.

Tests cover schema serialization and field exclusion behavior.
"""

import pytest
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dhenara.ai.types.genai.dhenara.request import StructuredOutputConfig


class SampleOutputModel(PydanticBaseModel):
    """Sample Pydantic model for testing structured output"""

    name: str = Field(..., description="Name of the entity")
    age: int = Field(..., description="Age in years")
    active: bool = Field(default=True, description="Whether active")


@pytest.mark.unit
@pytest.mark.case_id("DAI-061")
def test_from_model_and_model_dump_excludes_flags():
    """
    GIVEN a StructuredOutputConfig created from a Pydantic model
    WHEN model_dump() is called
    THEN the output should include output_schema
    AND should exclude model_class_reference (marked with exclude=True)
    AND should exclude allow_post_process_on_error (internal flag)
    """
    # Create config from a Pydantic model class
    config = StructuredOutputConfig.from_model(SampleOutputModel)

    # Verify the model class reference is stored internally
    assert config.get_model_class() == SampleOutputModel

    # Verify the schema is available
    schema = config.get_schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]

    # Test model_dump excludes the fields as expected
    dumped = config.model_dump()

    # Should have output_schema
    assert "output_schema" in dumped
    assert dumped["output_schema"] == schema

    # Should NOT have model_class_reference (excluded)
    assert "model_class_reference" not in dumped

    # Should NOT have allow_post_process_on_error (custom exclusion in model_dump)
    assert "allow_post_process_on_error" not in dumped

    # Test with allow_post_process_on_error explicitly set
    config_with_flag = StructuredOutputConfig.from_model(SampleOutputModel)
    config_with_flag.allow_post_process_on_error = True
    dumped_with_flag = config_with_flag.model_dump()

    # Flag should still not be in the dump
    assert "allow_post_process_on_error" not in dumped_with_flag
