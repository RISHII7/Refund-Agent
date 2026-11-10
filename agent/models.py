import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

model = ChatNVIDIA(
    model="nvidia/nemotron-3.5-bolt-e4b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.0,
    max_tokens=2048,
    model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
)
