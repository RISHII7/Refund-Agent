import os
from pathlib import Path
from dotenv import load_dotenv
from deepeval.models import OpenAIModel

# Load .env from project root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# NVIDIA NIM is OpenAI-compatible, so we use OpenAIModel with custom base_url
eval_model = OpenAIModel(
    model="nvidia/nemotron-3.5-bolt-e4b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)
