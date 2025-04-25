import base64
import io
import logging

from PIL import Image  # NOTE: You need to install 'Pillow' # pip install Pillow

from dhenara.ai import AIModelClient
from dhenara.ai.types import (
    AIModelAPIProviderEnum,
    AIModelCallConfig,
    AIModelEndpoint,
    ImageContentFormat,
    ResourceConfig,
)
from dhenara.ai.types.genai.foundation_models.google.image import Imagen3Fast
from dhenara.ai.types.genai.foundation_models.openai.image import DallE3, GPTImage1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dhenara")
logger.setLevel(logging.INFO)


# Initialize all model enpoints and collect it into a ResourceConfig.
# Ideally, you will do once in your application when it boots, and make it global
resource_config = ResourceConfig()
resource_config.load_from_file(
    credentials_file="~/.env_keys/.dhenara_credentials.yaml",  # Path to your file
)

openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
vertext_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_VERTEX_AI)

# Create various model endpoints, and add them to resource config
gpt_ep = AIModelEndpoint(api=openai_api, ai_model=GPTImage1)
dalle_ep = AIModelEndpoint(api=openai_api, ai_model=DallE3)
imagen_ep = AIModelEndpoint(api=vertext_api, ai_model=Imagen3Fast)


def print_response(response):
    if response.image_response:
        for choice in response.image_response.choices:
            for image_content in choice.contents:
                if image_content.content_format == ImageContentFormat.BASE64:
                    # Convert base64 to image
                    image_bytes = base64.b64decode(image_content.content_b64_json)
                    image = Image.open(io.BytesIO(image_bytes))

                    # Save the image
                    image.save("generated_image_b64.png")
                    print("Image saved as generated_image.png")
                elif image_content.content_format == ImageContentFormat.BYTES:
                    # Directly use the bytes data
                    image = Image.open(io.BytesIO(image_content.content_bytes))

                    # Save the image
                    image.save("generated_image_bytes.png")
                    print("Image saved as generated_image_from_bytes.png")
                elif image_content.content_format == ImageContentFormat.URL:
                    print(f"URL: {image_content.content_url}")

        # Optionally get the usage and cost for this call.
        # Usage/Cost calculation is enabled by default, but can be disabled via setting
        print("-" * 80)
        print(f"Usage: {response.image_response.usage}")
        print(f"Usage Charge: {response.image_response.usage_charge}")
        print("-" * 80)


user_query = "Elephant amigurumi walking in savanna, a professional photograph, blurry background"

# OpenAI
# GPT-Image
client = AIModelClient(
    model_endpoint=gpt_ep,
    config=AIModelCallConfig(
        options={
            "quality": "low",
            "size": "1024x1024",
            "n": 1,
        },
    ),
    is_async=False,  # Sync mode
)


response = client.generate(
    prompt=user_query,
    context=[],
    instructions=[],
)
print_response(response)


# Dalle
client = AIModelClient(
    model_endpoint=dalle_ep,
    config=AIModelCallConfig(
        options={
            "quality": "standard",
            "size": "1024x1024",
            "style": "natural",
            "n": 1,
            "response_format": "b64_json",  # or "url"
        },
    ),
    is_async=False,  # Sync mode
)


response = client.generate(
    prompt=user_query,
    context=[],
    instructions=[],
)
print_response(response)

# Google
# Create the client
client = AIModelClient(
    model_endpoint=imagen_ep,
    config=AIModelCallConfig(
        options={
            "aspect_ratio": "1:1",
            "number_of_images": 1,
            "person_generation": "dont_allow",
        },
    ),
    is_async=False,  # Sync mode
)


response = client.generate(
    prompt=user_query,
    context=[],
    instructions=[],
)
print_response(response)
