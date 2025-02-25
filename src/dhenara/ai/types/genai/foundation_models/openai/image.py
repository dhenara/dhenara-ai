from dhenara.ai.types.genai.ai_model import (
    AIModelFunctionalTypeEnum,
    AIModelProviderEnum,
    FoundationModel,
    ImageModelCostData,
    ImageModelSettings,
    ValidOptionValue,
)

DallE2 = FoundationModel(
    model_name="dall-e-2",
    display_name="dall-e-2",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.IMAGE_GENERATION,
    settings=ImageModelSettings(max_words=1000),
    valid_options={
        "quality": ValidOptionValue(
            allowed_values=["standard"],
            default_value="standard",
            cost_sensitive=True,
            description="Image quality",
        ),
        "size": ValidOptionValue(
            allowed_values=["256x256", "512x512", "1024x1024"],
            default_value="1024x1024",
            cost_sensitive=True,
            description="Dimensions",
        ),
        "n": ValidOptionValue(
            allowed_values=list(range(1, 10)),
            default_value=1,
            cost_sensitive=True,
            description="Number of images",
        ),
        "response_format": ValidOptionValue(
            allowed_values=["b64_json", "url"],
            default_value="url",
            cost_sensitive=False,
            description="Response format",
        ),
    },
    metadata={
        "details": "DALL·E model released in Nov 2023.",
    },
    cost_data=ImageModelCostData(
        flat_cost_per_image=None,  # Pricing is within options
        image_options_cost_data=[
            {
                "size": ["256x256"],
                "cost_per_image": 0.016,
            },
            {
                "size": ["512x512"],
                "cost_per_image": 0.018,
            },
            {
                "size": ["1024x1024"],
                "cost_per_image": 0.020,
            },
        ],
    ),
)


DallE3 = FoundationModel(
    model_name="dall-e-3",
    display_name="dall-e-3",
    provider=AIModelProviderEnum.OPEN_AI,
    functional_type=AIModelFunctionalTypeEnum.IMAGE_GENERATION,
    settings=ImageModelSettings(max_words=4000),
    valid_options={
        "quality": ValidOptionValue(
            allowed_values=["standard", "hd"],
            default_value="standard",
            cost_sensitive=True,
            description="Image quality",
        ),
        "size": ValidOptionValue(
            allowed_values=["1024x1024", "1024x1792", "1792x1024"],
            default_value="1024x1024",
            cost_sensitive=True,
            description="Dimensions",
        ),
        "style": ValidOptionValue(
            allowed_values=["natural", "vivid"],
            default_value="natural",
            cost_sensitive=False,
            description="Style preference",
        ),
        "n": ValidOptionValue(
            allowed_values=[1],
            default_value=1,
            cost_sensitive=True,
            description="Number of images",
        ),
        "response_format": ValidOptionValue(
            allowed_values=["b64_json", "url"],
            default_value="url",
            cost_sensitive=False,
            description="Response format",
        ),
    },
    metadata={
        "details": "DALL·E model released in Nov 2023.",
    },
    cost_data=ImageModelCostData(
        flat_cost_per_image=None,  # Pricing is within options
        image_options_cost_data=[
            {
                "quality": ["standard"],
                "size": ["1024x1024"],
                "cost_per_image": 0.04,
            },
            {
                "quality": ["standard"],
                "size": ["1024x1792", "1792x1024"],
                "cost_per_image": 0.08,
            },
            {
                "quality": ["hd"],
                "size": ["1024x1024"],
                "cost_per_image": 0.08,
            },
            {
                "quality": ["hd"],
                "size": ["1024x1792", "1792x1024"],
                "cost_per_image": 0.12,
            },
        ],
    ),
)

IMAGE_MODELS = [DallE2, DallE3]
