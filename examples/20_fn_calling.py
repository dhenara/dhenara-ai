import datetime
import logging
import random
from typing import Any

from dhenara.ai import AIModelClient
from dhenara.ai.types import (
    AIModelAPIProviderEnum,
    AIModelCallConfig,
    AIModelEndpoint,
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    ResourceConfig,
    ToolChoice,
    ToolDefinition,
)
from dhenara.ai.types.conversation import ConversationNode
from dhenara.ai.types.genai.foundation_models.anthropic.chat import Claude35Haiku
from dhenara.ai.types.genai.foundation_models.google.chat import Gemini20FlashLite
from dhenara.ai.types.genai.foundation_models.openai.chat import GPT4oMini

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

anthropic_api = resource_config.get_api(AIModelAPIProviderEnum.ANTHROPIC)
openai_api = resource_config.get_api(AIModelAPIProviderEnum.OPEN_AI)
google_api = resource_config.get_api(AIModelAPIProviderEnum.GOOGLE_AI)

# Create various model endpoints, and add them to resource config
resource_config.model_endpoints = [
    AIModelEndpoint(api=anthropic_api, ai_model=Claude35Haiku),
    AIModelEndpoint(api=openai_api, ai_model=GPT4oMini),
    AIModelEndpoint(api=google_api, ai_model=Gemini20FlashLite),
]


# Define a toll definition of type `FunctionDefinition`
# There are 2 options, you can directly define it using `get_weather_tool_`
get_weather_function = FunctionDefinition(
    name="get_weather",
    description="Get the current weather in a given location",
    parameters=FunctionParameters(
        type="object",
        properties={
            "location": FunctionParameter(
                type="string",
                description="The city and state, e.g. San Francisco, CA",
                required=True,
            ),
            "unit": FunctionParameter(
                type="string",
                description="The unit system to use: 'celsius' or 'fahrenheit'",
                allowed_values=["celsius", "fahrenheit"],
                default="celsius",
            ),
        },
        required=["location"],
    ),
)
# and Create tool definition from the function
get_weather_tool = ToolDefinition(function=get_weather_function)


# OR
#
# Define Python function  with proper typing and docstrings
def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    """Get the current weather in a given location.

    :param location: The city and state, e.g. San Francisco, CA
    :param unit: The unit system to use: 'celsius' or 'fahrenheit'
    :return: Weather information including temperature and conditions
    """
    # In a real implementation, this would call a weather API
    # For demo purposes, we'll just return mock data
    temp = random.randint(0, 30) if unit == "celsius" else random.randint(32, 86)
    conditions = random.choice(["sunny", "cloudy", "rainy", "snowy"])

    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "conditions": conditions,
        "forecast": f"It's {conditions} with a temperature of {temp}°{'C' if unit == 'celsius' else 'F'}",
    }


# and then use the `from_callable` in ToolDefinition to create the tool
# The advantage of here is, you can later use your function to call with LLM generated arguments
get_weather_tool = ToolDefinition.from_callable(get_weather)


# Create a second function for calendar management
def schedule_meeting(
    title: str,
    start_time: str,
    duration_minutes: int = 30,
    attendees: list[str] | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Schedule a meeting on the user's calendar.

    :param title: The title of the meeting
    :param start_time: When the meeting starts (ISO format)
    :param duration_minutes: How long the meeting lasts
    :param attendees: List of email addresses for attendees
    :param location: Physical or virtual location for the meeting
    :return: Details of the scheduled meeting
    """
    # Mock implementation
    meeting_id = f"meet_{random.randint(1000, 9999)}"
    return {
        "meeting_id": meeting_id,
        "title": title,
        "start_time": start_time,
        "end_time": f"calculated_from_{start_time}_plus_{duration_minutes}_minutes",
        "attendees": attendees or [],
        "location": location,
        "status": "scheduled",
    }


# Create a second tool from the meeting function
schedule_meeting_tool = ToolDefinition.from_callable(schedule_meeting)


def handle_conversation_turn(
    user_query: str,
    instructions: list[str],
    endpoint: AIModelEndpoint,
    conversation_nodes: list[ConversationNode],
) -> ConversationNode:
    """Process a single conversation turn with the specified model and query."""
    # Create config with function calling

    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=1000,
            streaming=False,
            tools=[get_weather_tool, schedule_meeting_tool],  # Add both tools
            tool_choice=ToolChoice(type="one_or_more"),
        ),
        is_async=False,
    )

    context = []
    for node in conversation_nodes:
        context += node.get_context()

    # Generate response
    response = client.generate(
        prompt=user_query,
        context=context,
        instructions=instructions,
    )

    # Create conversation node
    node = ConversationNode(
        user_query=user_query,
        input_files=[],
        response=response.chat_response,
        timestamp=datetime.datetime.now().isoformat(),
    )

    return node


def run_multi_turn_conversation():
    multi_turn_queries = [
        "What's the weather like in New York?",
        "Can you schedule a meeting with my team tomorrow at 2pm?",
        "What functions can you call for me?",
    ]

    # Instructions for each turn
    instructions_by_turn = ["", "", ""]

    # Store conversation history
    conversation_nodes = []

    # Process each turn
    for i, query in enumerate(multi_turn_queries):
        # Choose a random model endpoint
        model_endpoint = random.choice(resource_config.model_endpoints)
        # OR choose if fixed order as
        # model_endpoint = resource_config.get_model_endpoint(model_name=Claude35Haiku.model_name)

        print(f"🔄 Turn {i + 1} with {model_endpoint.ai_model.model_name} from {model_endpoint.api.provider}\n")

        node = handle_conversation_turn(
            user_query=query,
            instructions=instructions_by_turn[i],  # Only if you need to change instruction on each turn, else leave []
            endpoint=model_endpoint,
            conversation_nodes=conversation_nodes,
        )

        # Display the conversation
        print(f"User: {query}")
        print(f"Model: {model_endpoint.ai_model.model_name}\n")
        for content in node.response.choices[0].contents:
            print(f"Model Response Content {content.index}:\n{content.get_text()}\n")

            if content.type == "tool_call" and content.tool_call:
                tool_call = content.tool_call
                function_name = tool_call.name
                arguments = tool_call.arguments.arguments_dict
                print("\n📞 Function call detected!")
                print(f"Function: {function_name}")
                print(f"Arguments: {arguments}")

                # Call the fns
                if function_name == "get_weather":
                    result = get_weather(**arguments)
                elif function_name == "schedule_meeting":
                    result = schedule_meeting(**arguments)
                else:
                    print(f"Unknown function: {function_name}")

                # Add a debug message to know the fn was called
                print(f"function result: {result}")
                # In a real implementation, you would process this result
                # and potentially send it back to the model

        print("-" * 80)
        # Append to nodes, so that next turn will have the context generated
        conversation_nodes.append(node)


if __name__ == "__main__":
    ## Print the generated tool definitions to see how they were created
    # print("📚 Tool Definition from get_weather function:")
    # print(get_weather_tool.model_dump_json(indent=2))

    # print("\n📚 Tool Definition from schedule_meeting function:")
    # print(schedule_meeting_tool.model_dump_json(indent=2))

    # print("\n🔧 Generated OpenAI format for get_weather:")
    # print(get_weather_tool.to_openai_format())

    print("\n🚀 Starting conversation...\n")
    run_multi_turn_conversation()
