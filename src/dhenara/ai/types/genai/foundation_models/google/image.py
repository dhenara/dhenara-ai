from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    FoundationModel,
    ImageModelCostData,
    ImageModelSettings,
    ValidOptionValue,
)

_imagen_valid_options = {
    "aspect_ratio": ValidOptionValue(
        allowed_values=["1:1", "9:16", "16:9", "4:3", "3:4"],
        default_value="1:1",
        cost_sensitive=True,
        description="Aspect ratio",
    ),
    "number_of_images": ValidOptionValue(
        allowed_values=list(range(1, 4)),
        default_value=1,
        cost_sensitive=True,
        description="Number of images",
    ),
    "add_watermark": ValidOptionValue(
        allowed_values=[True, False],
        default_value=False,
        cost_sensitive=False,
        description="Add watermark or not",
    ),
    "safety_filter_level": ValidOptionValue(
        allowed_values=[None, "block_low_and_above", "block_medium_and_above", "block_only_high"],
        default_value="block_only_high",
        cost_sensitive=False,
        description="Safety Filter Level",
    ),
    "person_generation": ValidOptionValue(
        allowed_values=[None, "dont_allow", "allow_adult", "allow_all"],
        default_value="allow_all",
        cost_sensitive=False,
        description="Person generation settings",
    ),
}


Imagen3 = FoundationModel(
    model_name="imagen-3.0-generate",
    display_name="imagen-3.0",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.IMAGE_GENERATION,
    settings=ImageModelSettings(),
    valid_options=_imagen_valid_options,
    metadata={
        "details": "Imagen 3.0",
        "version_suffix": "-002",  # NOTE: This is required for google models
    },
    order=1,
    cost_data=ImageModelCostData(
        flat_cost_per_image=0.04,
        image_options_cost_data=None,
    ),
)


Imagen3Fast = FoundationModel(
    model_name="imagen-3.0-fast-generate",
    display_name="imagen-3.0 Fast",
    provider=AIModelProviderEnum.GOOGLE_AI,
    functional_type=AIModelFunctionalTypeEnum.IMAGE_GENERATION,
    settings=ImageModelSettings(),
    valid_options=_imagen_valid_options,
    metadata={
        "details": "Imagen 3.0",
        "version_suffix": "-001",  # NOTE: This is required for google models
    },
    order=2,
    cost_data=ImageModelCostData(
        flat_cost_per_image=0.02,
        image_options_cost_data=None,
    ),
)
IMAGE_MODELS = [Imagen3, Imagen3Fast]
