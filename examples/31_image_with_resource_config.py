import base64
import io
import logging

from include.shared_config import create_artifact_config, generate_run_dirname
from PIL import Image  # NOTE: You need to install 'Pillow' # pip install Pillow

from dhenara.ai import AIModelClient
from dhenara.ai.types import (
    AIModelAPIProviderEnum,
    AIModelCallConfig,
    AIModelEndpoint,
    ImageContentFormat,
    ResourceConfig,
)
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

# Create OpenAI image model endpoints
gpt_ep = AIModelEndpoint(api=openai_api, ai_model=GPTImage1)
dalle_ep = AIModelEndpoint(api=openai_api, ai_model=DallE3)


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

# Generate run directory once for this session
run_dir = generate_run_dirname()

# OpenAI
# GPT-Image
artifact_config_gpt = create_artifact_config(f"31_image_rc/{run_dir}/gpt_image")

client = AIModelClient(
    model_endpoint=gpt_ep,
    config=AIModelCallConfig(
        options={
            "quality": "low",
            "size": "1024x1024",
            "n": 1,
        },
        artifact_config=artifact_config_gpt,
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
artifact_config_dalle = create_artifact_config(f"31_image_rc/{run_dir}/dalle3")

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
        artifact_config=artifact_config_dalle,
    ),
    is_async=False,  # Sync mode
)


response = client.generate(
    prompt=user_query,
    context=[],
    instructions=[],
)
print_response(response)

## Google Imagen example removed for now (OpenAI only in this turn)
