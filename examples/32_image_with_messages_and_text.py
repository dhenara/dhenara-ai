import base64
import io
import logging

from include.shared_config import create_artifact_config, generate_run_dirname
from PIL import Image  # NOTE: You need to install 'Pillow'

from dhenara.ai import AIModelClient
from dhenara.ai.types import (
    AIModelAPIProviderEnum,
    AIModelCallConfig,
    AIModelEndpoint,
    ImageContentFormat,
    ResourceConfig,
)
from dhenara.ai.types.genai.dhenara.request import Prompt
from dhenara.ai.types.genai.foundation_models.google.image import Imagen3Fast
from dhenara.ai.types.genai.foundation_models.openai.image import GPTImage15

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dhenara")
logger.setLevel(logging.INFO)


# Initialize all model endpoints and collect it into a ResourceConfig.
resource_config = ResourceConfig()
resource_config.load_from_file(
    # credentials_file="path/to/your/credentails_file"  # or None to use `DAI_CREDENTIALS_FILE`
)

openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
google_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_VERTEX_AI)

# Create an OpenAI image model endpoint
image_ep = AIModelEndpoint(api=openai_api, ai_model=GPTImage15)


def print_response(response):
    if response.image_response:
        for choice in response.image_response.choices:
            for image_content in choice.contents:
                if image_content.content_format == ImageContentFormat.BASE64:
                    image_bytes = base64.b64decode(image_content.content_b64_json)
                    image = Image.open(io.BytesIO(image_bytes))
                    image.save("generated_image_b64.png")
                    print("Image saved as generated_image_b64.png")
                elif image_content.content_format == ImageContentFormat.BYTES:
                    image = Image.open(io.BytesIO(image_content.content_bytes))
                    image.save("generated_image_bytes.png")
                    print("Image saved as generated_image_bytes.png")
                elif image_content.content_format == ImageContentFormat.URL:
                    print(f"URL: {image_content.content_url}")

        print("-" * 80)
        print(f"Usage: {response.image_response.usage}")
        print(f"Usage Charge: {response.image_response.usage_charge}")
        print("-" * 80)


# Build the image request using the new messages format (Prompt objects).
# This exercises the dhenara-ai image clients' messages support.
messages = [
    Prompt.with_text("Elephant amigurumi walking in savanna, a professional photograph, blurry background"),
    Prompt.with_text("Style: cinematic lighting, shallow depth of field"),
]

run_dir = generate_run_dirname()
artifact_config = create_artifact_config(f"32_image_messages/{run_dir}/gpt_image")

client = AIModelClient(
    model_endpoint=image_ep,
    config=AIModelCallConfig(
        options={
            "quality": "low",
            "size": "1024x1024",
            "n": 1,
        },
        artifact_config=artifact_config,
    ),
    is_async=False,
)

response = client.generate(
    prompt=None,
    context=[],
    instructions=[],
    messages=messages,
)

print_response(response)


# Imagen
image_ep = AIModelEndpoint(api=google_api, ai_model=Imagen3Fast)
artifact_config_dalle = create_artifact_config(f"31_image_rc/{run_dir}/imagen3")

client = AIModelClient(
    model_endpoint=image_ep,
    config=AIModelCallConfig(
        options={
            "aspect_ratio": "16:9",
            "number_of_images": 1,
        },
        artifact_config=artifact_config,
    ),
    is_async=False,
)

response = client.generate(
    prompt=None,
    context=[],
    instructions=[],
    messages=messages,
)

print_response(response)
