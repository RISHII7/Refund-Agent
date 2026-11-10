import sys
from pathlib import Path
from uuid import uuid4

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import CacheConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall, ToolCallParams

from evaluations.dataset import dataset
from evaluations.model import eval_model




SUCCESSFUL_TASK_DESCRIPTION = """
Evaluate whether the customer support refund agent successfully handled the user's specific request according to these policy rules:

Scenario A - User asks for a refund for specific item(s):
- The agent must only consider the item(s) the user explicitly named in their message (not all items in the user's order history). For example, if a user bought both a book and a pen, but only asks for a refund for the book, only the book should be refunded.
- If the requested item was never purchased (e.g. sneakers): The agent must deny the refund because the item was not bought, not issue any refund ticket, and set conversation=False. Doing so completes the task successfully.
- If the requested item was purchased outside the 30-day policy (e.g. purchased in July when today is November): The agent must deny the refund for that item, explaining it exceeds the 30-day limit, not issue a refund ticket for it, and set conversation=False. Doing so completes the task successfully.
- If all requested items are eligible: The agent must issue a refund ticket for the total amount, inform the user of the issued ticket with the dollar amount, and set conversation=False.
- If some requested items are eligible and some are not (mixed): The agent must issue a refund ticket for the eligible item(s) with the correct amount, explain why the ineligible item was denied, and set conversation=False.

Scenario B - User explicitly asks for information only (e.g. "I would like no refund, I just want to know what I purchased"):
- The agent must retrieve their order history, list what they bought, confirm that no refund ticket was issued, and set conversation=False. Doing so completes the task successfully.

Scenario C - User provides an invalid/non-existent username:
- The agent must recognize that the username was not found, tell the user the username is invalid, ask for a valid username, and set conversation=True to allow follow-up. Doing so completes the task successfully.
"""


def build_metrics():
    """Keep the thresholds and evaluator model used by the individual scripts."""
    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
        ),
        StepEfficiencyMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
        PlanQualityMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
        PlanAdherenceMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
    ]
    single_turn_metrics = [
        ArgumentCorrectnessMetric(
            threshold=None, 
            model=eval_model,
            include_reason=True,
        ),
        ToolCorrectnessMetric(
            threshold=0.9, 
            model=eval_model,
            include_reason=True,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Check that the actual output and expected output have the same "
                "logical conclusion about the user's request.",
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics


def main():
    load_dotenv()
    from agent.agent import agent


    trace_metrics, single_turn_metrics = build_metrics()
    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    test_cases = []

    # The callback captures the trace required by the four agentic metrics.
    for golden in dataset.evals_iterator(
        metrics=trace_metrics,
        cache_config=cache_config,
        display_config=display_config,
    ):
        response = agent.invoke(
            {"messages": [{"role": "user", "content": golden.input}]},
            config={
                "configurable": {"thread_id": str(uuid4())},
                "recursion_limit": 10,
                "callbacks": [CallbackHandler()],
            },
        )

        tools_called = []
        for message in response["messages"]:
            for tool_call in getattr(message, "tool_calls", []) or []:
                tools_called.append(
                    ToolCall(
                        name=tool_call["name"],
                        input_parameters=tool_call["args"],
                    )
                )

        # Reuse the same response for the three remaining metrics. Keep these
        # cases local so repeated runs do not accumulate dataset.test_cases.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=response["structured_response"].answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=single_turn_metrics,
        cache_config=cache_config,
        display_config=display_config,
    )
    print(f"\nCompleted all 7 metrics for {len(test_cases)} test cases.")


if __name__ == "__main__":
    main()