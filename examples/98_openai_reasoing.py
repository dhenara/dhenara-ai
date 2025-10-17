import time

from openai import OpenAI

client = OpenAI()


# Start conversation: user asks for a horoscope. Request a short "Reasoning" section.
input_list = [{"role": "user", "content": "Tell me a short story under 2000 chars."}]

first_resp = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    reasoning={
        "effort": "low",
        "summary": "auto",
    },
)


print("\nTurn 1 - assistant output:")
print(first_resp.output)

input_list += first_resp.output

time.sleep(0.5)  # small gap to emulate turn-based chat

# Follow-up user turn: ask for career-focused details. Again request explicit reasoning.
input_list.append(
    {
        "role": "user",
        "content": ("Thanks — Please add a twist to this."),
    }
)
print("\n Turn 2 input", input_list)

second_resp = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    reasoning={
        "effort": "low",
        "summary": "auto",
    },
)

print("\nTurn 2 - assistant raw JSON:")
print(second_resp.model_dump_json(indent=2))
print("\nTurn 2 - assistant text:")
print(second_resp.output)


input_list += second_resp.output
print("\n Turn 3 input", input_list)


time.sleep(0.5)  # small gap to emulate turn-based chat
# Final user turn: ask for a happy ending. Again request explicit reasoning.
input_list.append(
    {
        "role": "user",
        "content": ("Great! Now please conclude the story with an inspiring ending."),
    }
)
third_resp = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    reasoning={
        "effort": "low",
        "summary": "auto",
    },
)
print("\nTurn 3 - assistant raw JSON:")
print(third_resp.model_dump_json(indent=2))
print("\nTurn 3 - assistant text:")
print(third_resp.output)
input_list += third_resp.output
