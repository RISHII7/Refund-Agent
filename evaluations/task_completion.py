import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from agent.agent import agent
from evaluations.dataset import dataset
from evaluations.model import eval_model

from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler

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

metrics = [TaskCompletionMetric(threshold=0.8, model=eval_model, task=SUCCESSFUL_TASK_DESCRIPTION, include_reason=True)]


for golden in dataset.evals_iterator(metrics=metrics):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": str(uuid4())},
            "recursion_limit": 10,
            "callbacks": [CallbackHandler()]
        }
    )
