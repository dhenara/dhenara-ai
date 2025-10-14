import random

from include.shared_config import all_endpoints, load_resource_config

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig
from dhenara.ai.types.genai.dhenara import PromptMessageRoleEnum
from dhenara.ai.types.genai.dhenara.request.data import Content, Prompt, PromptText, SystemInstruction

# Initialize shared ResourceConfig and restrict to OpenAI endpoints
resource_config = load_resource_config()
resource_config.model_endpoints = all_endpoints(resource_config)


def demonstrate_multiple_formats():
    """Demonstrate different ways to call generate() with various input formats."""

    # Choose a model endpoint
    model_endpoint = random.choice(resource_config.model_endpoints)
    # OR choose if fixed order as
    # model_endpoint = resource_config.get_model_endpoint(model_name=Claude35Haiku.model_name)

    # Create client
    client = AIModelClient(
        model_endpoint=model_endpoint,
        config=AIModelCallConfig(
            max_output_tokens=1000,
            streaming=False,
        ),
        is_async=False,
    )

    user_query = "What are three ways to improve productivity? Respond in less than 300 words."

    print("🔄 Demonstrating multiple formats for generate() calls\n")

    # Format 1: Simple string prompt (most basic)
    print("📝 FORMAT 1: Simple string prompt")
    response1 = client.generate(
        prompt=user_query,
        instructions=["Be specific and actionable."],
    )
    print(f"Response: {response1.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 2: Dictionary prompt (original format)
    print("📝 FORMAT 2: Dictionary prompt (original format)")
    response2 = client.generate(
        prompt={"role": "user", "text": user_query},
        context=[],
        instructions=["Be specific and actionable."],
    )
    print(f"Response: {response2.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 3: Prompt object with Content
    print("📝 FORMAT 3: Using Prompt object with Content")
    content = Content(type="text", text=user_query)
    prompt_text = PromptText(content=content)
    prompt = Prompt(role=PromptMessageRoleEnum.USER, text=prompt_text)

    response3 = client.generate(
        prompt=prompt,
        instructions=["Be specific and actionable."],
    )
    print(f"Response: {response3.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 4: Using SystemInstruction
    print("📝 FORMAT 4: Using SystemInstruction")
    instructions = [SystemInstruction(text="Be specific and actionable.")]

    response4 = client.generate(
        prompt="What are three ways to improve productivity?",
        instructions=instructions,
    )
    print(f"Response: {response4.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 5: Using PromptText directly
    print("📝 FORMAT 5: Using PromptText directly")
    prompt_text = PromptText(content=Content(type="text", text=user_query))
    prompt = Prompt(role=PromptMessageRoleEnum.USER, text=prompt_text)

    response5 = client.generate(
        prompt=prompt,
        instructions=["Be specific and actionable."],
    )
    print(f"Response: {response5.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 6: Using response as context for next turn
    print("📝 FORMAT 6: Using response as context for next turn")
    context_message = response5.chat_response.to_prompt()

    follow_up_response = client.generate(
        prompt="Tell me more about the first suggestion.",
        context=[context_message],
        instructions=["Be detailed."],
    )
    print(f"Response: {follow_up_response.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 7: Using PromptContext for multi-turn conversation
    print("📝 FORMAT 7: Using PromptContext for multi-turn conversation")
    # Create a PromptContext with multiple messages
    prompt_context = [
        Prompt(
            role=PromptMessageRoleEnum.USER,
            text="What are the benefits of regular exercise?",
        ),
        Prompt(
            role=PromptMessageRoleEnum.ASSISTANT,
            text="Regular exercise has numerous benefits including improved cardiovascular health, weight management, and enhanced mood.",  # noqa: E501
        ),
    ]

    # New prompt
    new_prompt = Prompt(
        role=PromptMessageRoleEnum.USER,
        text="How can I incorporate exercise into a busy schedule?",
    )

    response7 = client.generate(prompt=new_prompt, context=prompt_context, instructions=["Be practical and concise."])
    print(f"Response: {response7.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    # Format 8: Mix of formats in context
    print("📝 FORMAT 8: Mix of formats in context")
    mixed_context = [
        {"role": "user", "text": "What are good habits for productivity?"},
        Prompt(
            role=PromptMessageRoleEnum.ASSISTANT,
            text="Good productivity habits include time blocking, regular breaks, and prioritization.",
        ),
        "Can you explain time blocking in more detail?",
    ]

    response8 = client.generate(
        prompt="How does time blocking compare to other time management techniques?",
        context=mixed_context,
        instructions=["Compare and contrast different approaches."],
    )
    print(f"Response: {response8.chat_response.choices[0].contents[0].get_text()[:100]}...\n")

    print("✅ Demonstration complete")


if __name__ == "__main__":
    demonstrate_multiple_formats()
