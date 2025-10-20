"""Streaming multi-turn conversation with structured output + thinking (no tools).

This example runs up to 5 dependent turns. Each turn:
- Streams model output with reasoning enabled
- Requests structured JSON via Pydantic schema
- Validates the returned structure and flags mismatches
- Feeds validated values into the next turn

Exit code is non-zero if any turn fails validation.

Usage:
  # From repo root (activates venv and PYTHONPATH helpers)
  #   source sourceme_dev
  # Then:
  #   cd packages/dhenara_ai/examples
  #   python 19_streaming_multi_turn_structured_thinking.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from include.console_renderer import StreamingRenderer, render_usage
from include.shared_config import all_endpoints, create_artifact_config, generate_run_dirname, load_resource_config
from pydantic import BaseModel, Field

from dhenara.ai import AIModelClient
from dhenara.ai.types import AIModelCallConfig, AIModelEndpoint
from dhenara.ai.types.genai.dhenara.request import MessageItem, Prompt


# ------------------------
# Pydantic schemas per turn
# ------------------------


class TravelPlan(BaseModel):
    destination: str
    days: int = Field(ge=1, le=14)
    interests: List[str] = Field(min_length=1)


class ItineraryDay(BaseModel):
    day: int = Field(ge=1)
    activities: List[str] = Field(min_length=1)


class Itinerary(BaseModel):
    days: List[ItineraryDay] = Field(min_length=1)


class BudgetItem(BaseModel):
    name: str
    cost: float = Field(ge=0)


class Budget(BaseModel):
    currency: str
    items: List[BudgetItem] = Field(min_length=1)
    total: float = Field(ge=0)


class PackingList(BaseModel):
    items: List[str] = Field(min_length=1)
    assumptions: str


class RiskItem(BaseModel):
    risk: str
    mitigation: str
    severity: Literal["low", "medium", "high"]


class RiskAssessment(BaseModel):
    risks: List[RiskItem] = Field(min_length=1)


# ------------------------
# State & helpers
# ------------------------


@dataclass
class TestState:
    plan: Optional[TravelPlan] = None
    itinerary: Optional[Itinerary] = None
    budget: Optional[Budget] = None
    packing: Optional[PackingList] = None
    risks: Optional[RiskAssessment] = None
    errors: list[str] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return len(self.errors) == 0


def expect_structured(final, model_type: type[BaseModel], state: TestState, turn_name: str):
    """Validate structured output for a turn; record error on mismatch.

    Returns parsed model instance or None.
    """
    if not final:
        state.errors.append(f"{turn_name}: no final response from model")
        return None
    data = final.structured() or {}
    try:
        obj = model_type(**data)
        return obj
    except Exception as e:
        # Save raw for debugging
        try:
            raw = json.dumps(data, indent=2)
        except Exception:
            raw = str(data)
        state.errors.append(f"{turn_name}: structured validation failed: {e}\nRaw: {raw}")
        return None


def run_turn(
    *,
    endpoint: AIModelEndpoint,
    messages: list[MessageItem],
    prompt_text: str,
    struct_model: type[BaseModel],
    art_subdir: str,
):
    artifact_config = create_artifact_config(art_subdir)
    client = AIModelClient(
        model_endpoint=endpoint,
        config=AIModelCallConfig(
            max_output_tokens=2000,
            max_reasoning_tokens=1024,
            reasoning_effort="low",
            reasoning=True,
            streaming=True,
            artifact_config=artifact_config,
            # No tools in this example
            structured_output=struct_model,
        ),
        is_async=False,
    )

    streaming_renderer = StreamingRenderer()
    prompt = Prompt(role="user", text=prompt_text)
    # Persist the user's prompt into the messages history so subsequent turns include it
    messages = [*messages, prompt]
    stream = client.generate(messages=messages)
    final = streaming_renderer.process_stream(stream)

    # Append the assistant message (contains thinking + structured block)
    if final:
        assistant_msg = final.to_message_item()
        if assistant_msg:
            messages.append(assistant_msg)

    return final, messages


def close_enough(a: float, b: float, rel_tol: float = 0.05, abs_tol: float = 1.0) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def run_streaming_multi_turn_structured_thinking():
    print("=" * 80)
    print("Streaming Multi-Turn: Structured Output + Thinking (no tools)")
    print("=" * 80)

    # Prepare resource config and model endpoint (use a single endpoint for all turns)
    resource_config = load_resource_config()
    resource_config.model_endpoints = all_endpoints(resource_config)
    endpoints = resource_config.model_endpoints
    if not endpoints:
        print("No model endpoints configured. Please update your resource config.")
        sys.exit(2)

    endpoint = endpoints[0]
    print(f"Using endpoint: {endpoint.ai_model.model_name} ({endpoint.api.provider})")

    messages: list[MessageItem] = []
    run_dir = generate_run_dirname()
    state = TestState()

    def safe_render_usage(final):
        try:
            if final:
                render_usage(final)
        except Exception as e:
            print(f"Usage unavailable: {e}")

    # Turn 1: Get a TravelPlan
    print("\n🔄 Turn 1: TravelPlan\n")
    t1_prompt = (
        "Create a TravelPlan JSON for a 3-day trip to Tokyo focused on food and culture. "
        "Only output JSON matching the schema exactly—no extra keys or markdown."
    )
    final1, messages = run_turn(
        endpoint=endpoint,
        messages=messages,
        prompt_text=t1_prompt,
        struct_model=TravelPlan,
        art_subdir=f"19_struct/{run_dir}/turn_1",
    )
    safe_render_usage(final1)
    plan = expect_structured(final1, TravelPlan, state, "Turn1:TravelPlan")
    if plan is None:
        print("Turn 1 failed: Could not parse TravelPlan.")
        print("\n".join(state.errors))
        sys.exit(1)
    state.plan = plan
    print("Parsed TravelPlan:")
    print(json.dumps(plan.model_dump(), indent=2))

    # Turn 2: Build an Itinerary consistent with TravelPlan
    print("\n🔄 Turn 2: Itinerary\n")
    t2_prompt = (
        "Based on the previous TravelPlan, create an Itinerary with exactly "
        f"{plan.days} days for {plan.destination}. Each day must include 2-4 activities that align with "
        f"these interests: {', '.join(plan.interests)}. Number days starting at 1. "
        "Only output JSON matching the schema."
    )
    final2, messages = run_turn(
        endpoint=endpoint,
        messages=messages,
        prompt_text=t2_prompt,
        struct_model=Itinerary,
        art_subdir=f"19_struct/{run_dir}/turn_2",
    )
    safe_render_usage(final2)
    itinerary = expect_structured(final2, Itinerary, state, "Turn2:Itinerary")
    if itinerary is None:
        print("Turn 2 failed: Could not parse Itinerary.")
        print("\n".join(state.errors))
        sys.exit(1)
    # Extra validation: day count must match plan.days
    if len(itinerary.days) != plan.days:
        state.errors.append(f"Turn2:Itinerary day count mismatch. expected={plan.days}, got={len(itinerary.days)}")
    state.itinerary = itinerary
    print("Parsed Itinerary:")
    print(json.dumps(itinerary.model_dump(), indent=2))

    # Turn 3: Budget for the itinerary
    print("\n🔄 Turn 3: Budget\n")
    t3_prompt = (
        f"Estimate a realistic budget in USD for the {plan.days}-day itinerary in {plan.destination}. "
        "Include 3-6 items (e.g., lodging, food, transport, activities). Set total to the sum of item costs. "
        "Only output JSON matching the schema."
    )
    final3, messages = run_turn(
        endpoint=endpoint,
        messages=messages,
        prompt_text=t3_prompt,
        struct_model=Budget,
        art_subdir=f"19_struct/{run_dir}/turn_3",
    )
    safe_render_usage(final3)
    budget = expect_structured(final3, Budget, state, "Turn3:Budget")
    if budget is None:
        print("Turn 3 failed: Could not parse Budget.")
        print("\n".join(state.errors))
        sys.exit(1)
    # Validate total ~= sum(items)
    items_total = sum(i.cost for i in budget.items)
    if not close_enough(budget.total, items_total):
        state.errors.append(f"Turn3:Budget total mismatch. total={budget.total}, sum(items)={items_total}")
    state.budget = budget
    print("Parsed Budget:")
    print(json.dumps(budget.model_dump(), indent=2))

    # Turn 4: Packing list with assumptions
    print("\n🔄 Turn 4: PackingList\n")
    t4_prompt = (
        f"Create a PackingList for a {plan.days}-day trip to {plan.destination} focused on {', '.join(plan.interests)}. "
        "Assume mild spring weather. Include weather-related assumptions in the 'assumptions' field. "
        "Only output JSON matching the schema."
    )
    final4, messages = run_turn(
        endpoint=endpoint,
        messages=messages,
        prompt_text=t4_prompt,
        struct_model=PackingList,
        art_subdir=f"19_struct/{run_dir}/turn_4",
    )
    safe_render_usage(final4)
    packing = expect_structured(final4, PackingList, state, "Turn4:PackingList")
    if packing is None:
        print("Turn 4 failed: Could not parse PackingList.")
        print("\n".join(state.errors))
        sys.exit(1)
    state.packing = packing
    print("Parsed PackingList:")
    print(json.dumps(packing.model_dump(), indent=2))

    # Turn 5: Risks and mitigations
    print("\n🔄 Turn 5: RiskAssessment\n")
    t5_prompt = (
        f"Identify key trip risks for {plan.destination} and provide mitigations. "
        "Include 3-5 risks with severity levels (low/medium/high). "
        "Only output JSON matching the schema."
    )
    final5, messages = run_turn(
        endpoint=endpoint,
        messages=messages,
        prompt_text=t5_prompt,
        struct_model=RiskAssessment,
        art_subdir=f"19_struct/{run_dir}/turn_5",
    )
    safe_render_usage(final5)
    risks = expect_structured(final5, RiskAssessment, state, "Turn5:RiskAssessment")
    if risks is None:
        print("Turn 5 failed: Could not parse RiskAssessment.")
        print("\n".join(state.errors))
        sys.exit(1)
    state.risks = risks
    print("Parsed RiskAssessment:")
    print(json.dumps(risks.model_dump(), indent=2))

    # Final result
    print("\n" + "-" * 80)
    if state.all_valid:
        print("RESULT: PASS — All turns produced valid structured outputs and dependencies were respected.")
        sys.exit(0)
    else:
        print("RESULT: FAIL — Issues encountered:")
        for e in state.errors:
            print(" - " + e)
        sys.exit(1)


if __name__ == "__main__":
    run_streaming_multi_turn_structured_thinking()
