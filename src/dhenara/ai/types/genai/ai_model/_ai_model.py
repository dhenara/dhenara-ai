import logging
from typing import Any

from pydantic import Field, model_validator

from dhenara.ai.types.external_api._providers import AIModelFunctionalTypeEnum, AIModelProviderEnum
from dhenara.ai.types.genai.dhenara import ChatResponseUsage, ImageResponseUsage, UsageCharge
from dhenara.ai.types.shared.base import BaseModel

logger = logging.getLogger(__name__)


class ValidOptionValue(BaseModel):
    """
    Represents a valid option configuration for an AI model parameter.
    """

    allowed_values: list[Any] = Field(
        ...,
        description="List of allowed values for this option",
    )
    default_value: Any = Field(
        ...,
        description="Default value for this option",
    )
    cost_sensitive: bool = Field(
        ...,
        description="Will this option affect api-cost or not",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what this option controls",
    )
    display_only: bool = Field(
        default=False,
        description=(
            "Whether this option is only for display purpose."
            "When set, this won't be send in API calls. Useful for deriving additional options"
        ),
    )

    @model_validator(mode="after")
    def validate_default_in_allowed_values(self) -> "ValidOptionValue":
        """Ensures the default value is among allowed values."""
        if self.default_value not in self.allowed_values:
            raise ValueError(
                f"Default value {self.default_value} must be one of {self.allowed_values}",
            )
        return self


class BaseCostData(BaseModel):
    # NOTE: Default should be None to avoid wrong cost calculation
    # without proper overrides of standard foundation models in the package
    cost_multiplier_percentage: float | None = Field(
        default=None,
        description=(
            "Cost multiplication percentage f any."
            "Use this field to offset orgianl cost you paid to API provider with your additional expences/margin"
        ),
    )

    def calculate_usage_charge(self, usage) -> UsageCharge:
        raise NotImplementedError("calculate_usage_charge() not implemented")

    def get_charge(self, cost: float):
        if self.cost_multiplier_percentage:
            charge = round(
                cost * (1 + (self.cost_multiplier_percentage / 100)),
                6,
            )
        else:
            charge = None
        return UsageCharge(cost=cost, charge=charge)


class ChatModelCostData(BaseCostData):
    input_token_cost_per_million: float = Field(
        ...,
        description="",
    )
    output_token_cost_per_million: float = Field(
        ...,
        description="",
    )

    def calculate_usage_charge(
        self,
        usage: ChatResponseUsage,
    ) -> UsageCharge:
        try:
            input_per_token_cost = self.input_token_cost_per_million / 1000000
            output_per_token_cost = self.output_token_cost_per_million / 1000000

            cost = round(
                usage.prompt_tokens * input_per_token_cost + usage.completion_tokens * output_per_token_cost,
                6,
            )

            return self.get_charge(cost)
        except Exception as e:
            raise ValueError(f"calculate_usage_charge: Error: {e}")


class ImageModelCostData(BaseCostData):
    flat_cost_per_image: float | None = Field(
        default=None,
        description="Flat per image cost",
    )

    image_options_cost_data: list[dict] | None = Field(  # TODO: rename var
        default=None,
        description="Image options cost data",
    )

    @model_validator(mode="after")
    def _validate_cost_factores(self) -> "ImageModelCostData":
        if not (self.flat_cost_per_image or self.image_options_cost_data):
            raise ValueError("Either of flat_cost_per_image / image_options_cost_data must be set")
        if self.flat_cost_per_image and self.image_options_cost_data:
            raise ValueError("Set only one of flat_cost_per_image / image_options_cost_data is allowed")

        return self

    def calculate_usage_charge(
        self,
        usage: ImageResponseUsage,
    ) -> UsageCharge:
        try:
            cost_per_image = None
            if self.flat_cost_per_image:
                cost_per_image = self.flat_cost_per_image
            elif self.image_options_cost_data:
                cost_per_image = self.get_image_cost_with_options(
                    used_options=usage.options,
                )
            else:
                raise ValueError("calculate_image_charges: cost_per_image or cost options mapping is not set ")

            if cost_per_image is None:
                raise ValueError("calculate_image_charges: Failed to fix cost_per_image")

            cost = round(cost_per_image * usage.number_of_images, 6)
            return self.get_charge(cost)

        except Exception as e:
            raise ValueError(f"calculate_usage_charge: Error: {e}")

    # -------------------------------------------------------------------------
    def get_image_cost_with_options(self, used_options):
        if not self.image_options_cost_data:
            return None

        # Create a copy of used_options with standardized keys
        standardized_options = used_options.copy()

        # Remove keys that aren't used in cost data (like 'quality' )
        valid_keys = {key for data in self.image_options_cost_data for key in data.keys() if key != "cost_per_image"}
        standardized_options = {k: v for k, v in standardized_options.items() if k in valid_keys}

        for cost_data in self.image_options_cost_data:
            matches = True
            for key, value in standardized_options.items():
                if key not in cost_data or value not in cost_data[key]:
                    matches = False
                    break
            if matches:
                return cost_data["cost_per_image"]

        raise ValueError(
            f"get_image_cost_with_options: Failed to get price. "
            f"used_options={used_options}, image_options_cost_data={self.image_options_cost_data})"
        )


class ChatModelSettings(BaseModel):
    max_context_window_tokens: int | None = Field(
        default=None,
        description="Maximum context window size in tokens",
    )
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens allowed",
    )
    max_output_tokens: int = Field(
        ...,
        description="Maximum output tokens allowed",
    )
    # Reasoning settings
    supports_reasoning: bool = Field(
        default=False,
        description="If reasoning is supported or not",
    )
    max_reasoning_tokens: int | None = Field(
        default=None,
        description="Maximum reasoning tokens allowed",
    )
    max_output_tokens_reasoning_mode: int | None = Field(
        default=None,
        description="Maximum output tokens (including reasoning) in reasoning mode",
    )

    @model_validator(mode="after")
    def _set_token_limits(self) -> "ChatModelSettings":
        # Store original values
        input_tokens = self.max_input_tokens
        context_tokens = self.max_context_window_tokens
        output_tokens = self.max_output_tokens

        if not output_tokens:
            raise ValueError("set_token_limits: max_output_tokens must be specified")

        if not (input_tokens or context_tokens):
            raise ValueError("set_token_limits: max_input_tokens or max_context_window_tokens must be specified")

        # Calculate values without direct assignment
        values_to_update = {}
        if input_tokens is not None:
            values_to_update["max_context_window_tokens"] = input_tokens + output_tokens
        elif context_tokens is not None:
            values_to_update["max_input_tokens"] = context_tokens - output_tokens

        # fmt: off
        if self.supports_reasoning:
            if (
                self.max_reasoning_tokens
                and self.max_output_tokens_reasoning_mode
                and self.max_reasoning_tokens > self.max_output_tokens_reasoning_mode
            ):
                raise ValueError(
                    "set_token_limits: max_reasoning_tokens must be less than max_output_tokens_reasoning_mode"
                )
        # fmt: on

        # Update the model's dict directly
        for key, value in values_to_update.items():
            self.__dict__[key] = value

        return self


class ImageModelSettings(BaseModel):
    max_words: int | None = Field(
        None,
        description="Maximum word count, if applicable",
    )


class BaseAIModel(BaseModel):
    """
    Pydantic model representing an AI model configuration with options validation.
    """

    provider: AIModelProviderEnum = Field(
        ...,
        description="The AI model provider",
    )
    functional_type: AIModelFunctionalTypeEnum = Field(
        ...,
        description="Type of AI model functionality",
    )
    model_name: str = Field(
        ...,
        max_length=300,
        description="Model name used in API calls",
    )
    display_name: str = Field(
        ...,
        max_length=300,
        description="Display name for the model",
    )

    order: int = Field(
        0,
        description="Order for display purposes",
    )
    enabled: bool = Field(
        True,  # noqa: FBT003
        description="Whether the model is enabled",
    )
    beta: bool = Field(
        False,  # noqa: FBT003
        description="Whether the model is in beta",
    )

    settings: ChatModelSettings | ImageModelSettings | None = Field(
        default=None,
        description="Settings",
    )
    valid_options: dict[str, ValidOptionValue] | None = Field(
        default=None,
        description="Configured valid options and their allowed values",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata if needed",
    )

    cost_data: ChatModelCostData | ImageModelCostData | None = Field(
        None,
        description="Optional Cost data",
    )

    reference_number: str | None = Field(
        None,
        description="Optional unique reference number",
    )

    @property
    def model_name_with_version_suffix(self):
        version_suffix = self.metadata.get("version_suffix", None)
        if version_suffix:
            return f"{self.model_name}{version_suffix}"
        else:
            return self.model_name

    @property
    def is_foundation_model(self) -> bool:
        return isinstance(self, FoundationModel)

    @model_validator(mode="after")
    def _validate_model_options(self) -> "BaseAIModel":
        """Validates model options against foundation model if present."""
        # Option validation is not done for foundation models
        if self.is_foundation_model:
            return self

        # If, not foundation model, ideally, options should be set, at least as an empty dict
        if self.foundation_model is None or self.valid_options is None:
            raise ValueError(
                "Either set the foundation model, or set the `valid_options` for this model.\
 If there are no options required for the models, set `valid_options` to an empty dict, but not as None"
            )

        # Validate that all options are present in foundation model
        invalid_options = set(self.valid_options.keys()) - set(
            self.foundation_model.valid_options.keys(),
        )
        if invalid_options:
            raise ValueError(
                f"Invalid options found: {invalid_options}. Must be subset of foundation model options.",
            )

        # Validate option values against foundation model
        for option_name, option_config in self.valid_options.items():
            foundation_config = self.foundation_model.valid_options[option_name]
            invalid_values = set(option_config.allowed_values) - set(
                foundation_config.allowed_values,
            )
            if invalid_values:
                raise ValueError(
                    f"Invalid values for option {option_name}: {invalid_values}",
                )

        return self

    @model_validator(mode="after")
    def _validate_settings(self) -> "BaseAIModel":
        if not self.settings:
            return self

        (setting_model, cost_model) = AIModel.get_pydantic_model_classes(self.functional_type)
        if not isinstance(self.settings, setting_model):
            raise ValueError(f"Settings should be instance of {setting_model} for {self.functional_type} models")
        if self.is_foundation_model and not isinstance(self.cost_data, cost_model):
            raise ValueError(f"For {self.functional_type} models, cost data must be set and of type {cost_model}")

        return self

    @model_validator(mode="after")
    def _validate_names(self) -> "BaseAIModel":
        if not self.display_name:
            self.display_name = self.model_name

        return self

    def validate_options(self, options: dict[str, Any]) -> bool:
        """
        Validates if the provided options conform to the model's valid options.

        Args:
            options: Dictionary of option name to value mappings

        Returns:
            bool: True if options are valid, False otherwise
        """
        try:
            self._validate_options_strict(options)
            return True
        except ValueError as e:
            logger.error(f"validate_options Fails: {e}")
            return False

    def _validate_options_strict(self, options: dict[str, Any]) -> None:
        """
        Strictly validates options and raises ValueError for invalid options.

        Args:
            options: Dictionary of option name to value mappings

        Raises:
            ValueError: If any option is invalid
        """
        _valid_options = self.get_valid_options()

        invalid_options = set(options.keys()) - set(_valid_options.keys())
        if invalid_options:
            raise ValueError(f"Unknown options: {invalid_options}")

        for option_name, value in options.items():
            valid_values = _valid_options[option_name].allowed_values
            if value not in valid_values:
                raise ValueError(
                    f"Invalid value for {option_name}: {value}. Must be one of {valid_values}",
                )

    def get_options_with_defaults(self, options: dict[str, Any]) -> dict[str, Any]:
        """
        Returns a complete options dictionary with defaults for missing values.

        Args:
            options: Partial options dictionary

        Returns:
            dict: Complete options dictionary with defaults
        """
        _valid_options = self.get_valid_options()

        self._validate_options_strict(options)

        complete_options = {}
        for option_name, option_config in _valid_options.items():
            complete_options[option_name] = options.get(
                option_name,
                option_config.default_value,
            )

        return complete_options

    def _get_attribute(self, attr_name: str):
        """Generic method to get attributes with foundation model fallback

        Args:
            attr_name: Name of the attribute to get
        Returns:
            Value of the attribute from this model or foundation model
        """
        if self.is_foundation_model:
            return getattr(self, attr_name)

        value = getattr(self, attr_name)
        if not value:
            return getattr(self.foundation_model, f"get_{attr_name}")()
        return value

    def get_settings(self):
        return self._get_attribute("settings")

    def _get_valid_options(self):
        return self._get_attribute("valid_options")

    def get_valid_options(self, include_display_only: bool = False) -> dict[str, ValidOptionValue]:
        """
        Get valid options excluding display-only options.

        Returns:
            dict: Dictionary of valid options without display-only options
        """
        options = self._get_valid_options()
        if include_display_only:
            return options

        if options:
            for key in list(options.keys()):
                if options[key].display_only:
                    del options[key]
        return options

    def get_cost_data(self):
        return self._get_attribute("cost_data")

    # -------------------------------------------------------------------------
    @classmethod
    def get_pydantic_model_classes(cls, functional_type):
        if functional_type == AIModelFunctionalTypeEnum.TEXT_GENERATION:
            setting_model = ChatModelSettings
            cost_model = ChatModelCostData
        elif functional_type == AIModelFunctionalTypeEnum.IMAGE_GENERATION:
            setting_model = ImageModelSettings
            cost_model = ImageModelCostData
        else:
            raise ValueError(f"Functional type {functional_type} is not implemented ")

        return (setting_model, cost_model)


class FoundationModel(BaseAIModel):
    pass


class AIModel(BaseAIModel):
    foundation_model: FoundationModel | None = Field(
        None,
        description="Matching foundation model for parameter preloading",
    )
