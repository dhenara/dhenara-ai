import base64
import io

from PIL import Image  # NOTE: You need to install 'Pillow' # pip install Pillow

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.external_api import AIModelAPIProviderEnum
from dhenara.ai.types.genai import AIModelAPI, ImageContentFormat
from dhenara.ai.types.genai.foundation_models.openai.image import DallE3

# 1. Create an API
# This can be used to create multiple model endpoints for the same API provider
openai_api = AIModelAPI(
    provider=AIModelAPIProviderEnum.OPEN_AI,
    api_key="your_api_key",
)


# Create the model endpoint
model_endpoint = AIModelEndpoint(api=openai_api, ai_model=DallE3)

# Create the client
client = AIModelClient(
    model_endpoint=model_endpoint,
    config=AIModelCallConfig(
        options={
            "quality": "standard",
            "size": "1024x1024",
            "style": "natural",
            "n": 1,
            "response_format": "b64_json",
        },
        test_mode=False,
    ),
    is_async=False,  # Sync mode
)


user_query = "Elephant amigurumi walking in savanna, a professional photograph, blurry background"

response = client.generate(
    prompt=user_query,
    context=[],
    instructions=[],
)


if response.image_response:
    for choice in response.image_response.choices:
        for image_content in choice.contents:
            if image_content.content_format == ImageContentFormat.BASE64:
                # Convert base64 to image
                image_bytes = base64.b64decode(image_content.content_b64_json)
                image = Image.open(io.BytesIO(image_bytes))

                # Save the image
                image.save("generated_image.png")
                print("Image saved as generated_image.png")
            elif image_content.content_format == ImageContentFormat.URL:
                print(f"URL: {image_content.content_url}")

    # Optionally get the usage and cost for this call.
    # Usage/Cost calculation is enabled by default, but can be disabled via setting
    print("-" * 80)
    print(f"Usage: {response.image_response.usage}")
    print(f"Usage Charge: {response.image_response.usage_charge}")
    print("-" * 80)
