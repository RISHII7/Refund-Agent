import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from langchain.agents import create_agent
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

if os.getenv("OPENAI_API_KEY"):
    agent_model = "openai:gpt-4"
    eval_model = "gpt-4o"
else:
    from agent.models import model as agent_model
    from evaluations.model import eval_model


# Step 1: Define the tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Step 2: Create LangChain agent
agent = create_agent(
    model=agent_model,
    tools=[multiply],
    system_prompt="Be concise. Use tools for any math questions.",
)


# Step 3: Define goldens
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What is 8123 multiplied by 62323?"
        ),
        Golden(
            input="What is 7 multiplied by 92323?"
        ),
    ]
)


# Step 4: Execute agent and create test cases
for golden in dataset.goldens:

    print(f"\nEvaluating: {golden.input}")

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": golden.input}
            ]
        },
        config={"recursion_limit": 10},
    )

    actual_output = result["messages"][-1].content

    # Extract actual tool calls
    tools_called = []

    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    input_parameters=tool_call["args"],
                )
            )

    # Create LLMTestCase
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        tools_called=tools_called,
    )

    dataset.add_test_case(test_case)

    print("Agent Response:", actual_output)
    print("Tools Called:", tools_called)


# Step 5: Configure Argument Correctness metric
metric = ArgumentCorrectnessMetric(
    threshold=1.0,
    model=eval_model,
    include_reason=True,
    verbose_mode=True,
)


# Step 6: Run evaluation
evaluate(
    test_cases=dataset.test_cases,
    metrics=[metric],
)
