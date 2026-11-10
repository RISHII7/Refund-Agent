import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
import json_repair
import deepeval.models.llms.utils as deepeval_utils
from deepeval.models import OpenAIModel

# Monkeypatch DeepEval's trim_and_load_json with json_repair to handle unescaped quotes in LLM responses
_orig_trim_and_load = deepeval_utils.trim_and_load_json

def _safe_trim_and_load_json(input_string: str):
    start = input_string.find("{")
    end = input_string.rfind("}") + 1
    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)
    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
    try:
        return json.loads(jsonStr)
    except Exception:
        try:
            return json_repair.loads(jsonStr)
        except Exception:
            return _orig_trim_and_load(input_string)

deepeval_utils.trim_and_load_json = _safe_trim_and_load_json

# Load .env from project root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# NVIDIA NIM is OpenAI-compatible, so we use OpenAIModel with custom base_url
eval_model = OpenAIModel(
    model="nvidia/nemotron-3.5-bolt-e4b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)
