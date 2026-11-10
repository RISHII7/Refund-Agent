import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from langchain.agents import create_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric

if os.getenv("OPENAI_API_KEY"):
    agent_model = "openai:gpt-4o-mini"
    eval_model = "gpt-4o"
else:
    from agent.models import model as agent_model
    from evaluations.model import eval_model

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = create_agent(
    model=agent_model,
    tools=[multiply],
    system_prompt="Be concise. Use tools for any math questions.",
)

# Goldens are the inputs you want to evaluate.
dataset = EvaluationDataset(goldens=[
    Golden(input="What is 8 multiplied by 6?"),
    Golden(input="What is 7 multiplied by 9?"),
])

metrics = [TaskCompletionMetric(model=eval_model)]


# The `TaskCompletionMetric` is passed into the `evals_iterator`.
for golden in dataset.evals_iterator(metrics=metrics):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "recursion_limit": 10,
            "callbacks": [CallbackHandler()],
        },
    )
