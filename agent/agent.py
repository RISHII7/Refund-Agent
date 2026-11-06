import sys
from pathlib import Path

# Add project root to sys.path so 'agent' package is always resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uuid import uuid4
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.models import model
from agent.tools import (
    calculate_right_refund,
    get_today_date,
    issue_refund_ticket,
    get_user_purchase_history,
)
from agent.prompts import agent_system_prompt
from agent.objects import Answer


load_dotenv()

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[calculate_right_refund,get_user_purchase_history,issue_refund_ticket,get_today_date],
    response_format=Answer,
    checkpointer=InMemorySaver()
)

def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}

    text = "I would like to request a refund"
    print(text)
    
    question = HumanMessage(content=text)
    conversation = True

    while conversation:

        response = agent.invoke({"messages":question}, config=config)
        structured_response = response["structured_response"]

        print(structured_response.answer)

        conversation = structured_response.conversation

        if conversation:
            user_input = input("reply: \n")
            question = HumanMessage(content=user_input)

    print("the job is done!")

run_conversation()