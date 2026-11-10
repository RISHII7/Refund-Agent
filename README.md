# 💳 Refund Agent

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.4+-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agents-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![DeepEval](https://img.shields.io/badge/DeepEval-4.3+-purple.svg)](https://github.com/confident-ai/deepeval)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Nemotron--3.5--bolt--e4b-76B900.svg)](https://build.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, autonomous customer refund support agent built with **LangChain**, **LangGraph**, and **NVIDIA NIM** (`nvidia/nemotron-3.5-bolt-e4b`). The system enforces business refund policies, validates customer identity and purchase histories, calculates partial or full refunds, issues support tickets, and features a comprehensive end-to-end evaluation suite powered by **DeepEval**.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Agent Workflow & Policy Rules](#-agent-workflow--policy-rules)
- [Core Tools & Schema](#-core-tools--schema)
- [NVIDIA NIM Integration](#-nvidia-nim-integration)
- [Evaluation Suite (`evaluations/`)](#-evaluation-suite-evaluations)
  - [Metrics Evaluated](#metrics-evaluated)
  - [Test Dataset Coverage](#test-dataset-coverage)
  - [Reliability & Resilience Enhancements](#reliability--resilience-enhancements)
- [Educational Metrics Suite (`understanding_metrics/`)](#-educational-metrics-suite-understanding_metrics)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Installation](#installation)
- [Running the Project](#-running-the-project)
  - [Interactive Agent CLI](#1-interactive-agent-cli)
  - [Running the Complete Evaluation Suite](#2-running-the-complete-evaluation-suite)
  - [Running Individual Metric Tests](#3-running-individual-metric-tests)
  - [Running Educational Metrics Examples](#4-running-educational-metrics-examples)
- [Troubleshooting & Windows Tips](#-troubleshooting--windows-tips)

---

## 🌟 Overview & Architecture

The **Refund Agent** is designed to handle multi-turn customer refund inquiries deterministically and safely. It uses a stateful LangGraph agent architecture with an in-memory checkpointer (`InMemorySaver`), allowing conversation continuity while adhering strictly to sequence constraints and refund guidelines.

```
                  +----------------------------------+
                  |         Customer Input           |
                  +-----------------+----------------+
                                    |
                                    v
                  +----------------------------------+
                  |         LangGraph Agent          |
                  |  (nvidia/nemotron-3.5-bolt-e4b)  |
                  |     (Thinking Mode Enabled)      |
                  +-----------------+----------------+
                                    |
         +--------------------------+--------------------------+
         |                          |                          |
         v                          v                          v
+------------------+       +------------------+       +------------------+
| User History Tool|       | Date Check Tool  |       | Refund Calc Tool |
|  (Verify User &  |       | (System Benchmark|       | (Filter items by |
|   Past Orders)   |       |   2026-10-25)    |       |   eligibility)   |
+------------------+       +------------------+       +------------------+
         |                          |                          |
         +--------------------------+--------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |      Issue Refund Ticket Tool    |
                  |  (Only if valid eligible amount) |
                  +-----------------+----------------+
                                    |
                                    v
                  +----------------------------------+
                  |    Structured Answer Schema      |
                  | - answer: Final customer response|
                  | - conversation: Boolean flag     |
                  +----------------------------------+
```

---

## 📋 Agent Workflow & Policy Rules

The agent adheres strictly to the following corporate refund policy guidelines:

1. **User Identity Validation**:
   - The agent first queries `get_user_purchase_history(user=<username>)`.
   - If the user does not exist in the database, the agent immediately halts refund processing and prompts:
     > *"The username was not valid. Please provide a valid username before we can give information about a refund."*
2. **Purchase Verification**:
   - If a requested item was never purchased by the user, the agent politely rejects the refund without processing it.
3. **30-Day Policy Evaluation**:
   - For all requested items present in the purchase history, the agent fetches the current date using `get_today_date()`.
   - Purchases older than **30 days** relative to the benchmark date are strictly ineligible for refunds.
4. **Item-Level Partial Refunds**:
   - In mixed requests (e.g., one item within 30 days, one item outside 30 days), the agent calculates eligible amounts using `calculate_right_refund`, issues a ticket for eligible items via `issue_refund_ticket`, and provides explicit itemized justifications for non-eligible items.
5. **Sequential Execution Guardrails**:
   - The agent is forbidden from calling the final `Answer` structured response prematurely or as an intermediate status update. All necessary tool actions must execute first.

---

## 🛠️ Core Tools & Schema

The agent interacts with four deterministic mock tools located in [`agent/tools.py`](agent/tools.py):

| Tool | Parameters | Purpose |
| :--- | :--- | :--- |
| `get_user_purchase_history` | `user: str` | Retrieves user purchase records (item names, prices, and purchase dates) from mock database. Returns an error message if the user does not exist. |
| `get_today_date` | *None* | Returns the benchmark system date (`2026-10-25`) to assess the 30-day window. |
| `calculate_right_refund` | `user: str, items: list[dict]` | Computes the exact monetary refund total for the specified eligible item list. |
| `issue_refund_ticket` | `user: str, refund_amount: int` | Records and issues an official ticket ID confirming the refund has been processed. |

The agent returns structured responses matching the Pydantic schema in [`agent/objects.py`](agent/objects.py):

```python
class Answer(BaseModel):
    answer: str          # Final human-readable response to the user
    conversation: bool   # False when request is resolved, True if more info needed
```

---

## 🚀 NVIDIA NIM Integration

Both agent execution and evaluation LLM judging use NVIDIA's high-performance **NIM** API with the **Nemotron-3.5-Bolt** reasoning model (`nvidia/nemotron-3.5-bolt-e4b`):

- **Agent Engine (`agent/models.py`)**:
  - Leverages `langchain_nvidia_ai_endpoints.ChatNVIDIA`.
  - Configured with `temperature=0.0` for deterministic execution and `enable_thinking: True` for robust chain-of-thought planning and tool calling.
- **Evaluation Judge (`evaluations/model.py`)**:
  - NVIDIA NIM exposes an OpenAI-compatible API endpoint (`https://integrate.api.nvidia.com/v1`).
  - Integrated into DeepEval via `deepeval.models.OpenAIModel`, eliminating external OpenAI or Gemini dependencies.

---

## 🧪 Evaluation Suite (`evaluations/`)

The repository includes a comprehensive evaluation framework powered by **DeepEval** to validate agent intelligence across both component-level and conversational dimensions.

### Metrics Evaluated

1. **Task Completion Metric** (`task_completion.py`):
   - Measures whether the agent completely fulfilled the user's explicit objective based on the execution trace and final output.
2. **Step Efficiency Metric** (`step_efficiency.py`):
   - Assesses execution path conciseness, penalizing redundant, circular, or extraneous tool calls.
3. **Plan Quality Metric** (`plan_quality.py`):
   - Evaluates the decomposition and logical order of steps planned by the model prior to execution.
4. **Plan Adherence Metric** (`plan_adherence.py`):
   - Verifies whether the agent faithfully executed the sequence of steps outlined in its internal plan.
5. **Tool Correctness Metric** (`tool_correctness.py`):
   - Compares the tools invoked by the agent against golden tool definitions, verifying that required tools were executed and unneeded tools were avoided.
6. **Argument Correctness Metric** (`argument_correctness.py`):
   - Validates that parameters passed to each tool (e.g., username, item names, refund amounts) match expected parameters exactly.
7. **Answer Correctness (GEval)** (`answer_correctness.py`):
   - LLM-as-a-judge metric using a formal multi-step rubric to evaluate factual accuracy, policy explanation clarity, and currency formatting tolerance.

### Test Dataset Coverage

The evaluation dataset defined in [`evaluations/dataset.py`](evaluations/dataset.py) encompasses 7 diverse scenarios:

1. **Multi-item eligible refund**: Valid user requesting refund for two recent items ($30 total).
2. **Single-item eligible refund**: Valid user requesting refund for one recent item ($23 total).
3. **Informational lookup only**: Valid user asking what they bought (no refund, only `get_user_purchase_history` called).
4. **Invalid username handling**: Non-existent user `user999` (rejects request, asks for valid username).
5. **Policy expired item**: Valid user requesting refund for an item purchased > 30 days ago (clean rejection).
6. **Mixed eligibility order**: Valid user with one item inside policy ($45) and one outside (partial refund approved with explanation).
7. **Unpurchased item inquiry**: Valid user requesting refund for an item not in purchase records (notifies user and halts).

### Reliability & Resilience Enhancements

- **DeepEval JSON Parsing Fix (`evaluations/model.py`)**:
  - Automatically monkeypatches DeepEval's internal `trim_and_load_json` using `json-repair` (`json_repair.loads`). This prevents `JSONDecodeError` when evaluation models emit unescaped quotes or markdown formatting in evaluation reasoning.
- **Structured Response Filter**:
  - Automatically filters out LangChain's internal `Answer` tool call from `tools_called` lists, ensuring clean comparison of domain tools against test golden definitions.
- **Windows File Lock Bypass**:
  - Uses `CacheConfig(use_cache=False, write_cache=False)` to prevent `portalocker` file lock conflicts and cache corruption on Windows environments.

---

## 📚 Educational Metrics Suite (`understanding_metrics/`)

Located in [`understanding_metrics/`](understanding_metrics/), this directory is an interactive, educational playground derived from the LangChain Agent Evaluation System. It uses a simplified Math Tutor agent with **intentional, controlled design flaws** to demonstrate precisely how each DeepEval metric reacts:

- `all_metrics.py`: Runs all educational evaluations in one script.
- `step_efficiency.py`: Shows how extraneous calculations degrade step efficiency scores.
- `plan_adherence.py`: Demonstrates penalty when an agent deviates from its declared plan.
- `plan_quality.py`: Tests scoring of poor vs. high quality multi-step mathematical plans.
- `tool_correctness.py`: Illustrates detection of hallucinated or missing tools.
- `argument_correctness.py`: Highlights detection of parameter mismatch errors.
- `answer_correctness.py`: Compares semantic correctness of answers against goldens using GEval.
- `task_completion.py`: Demonstrates scoring when a goal is partially vs. fully achieved.

---

## 📂 Project Structure

```
.
├── agent/                                # Refund Agent Implementation
│   ├── __init__.py
│   ├── agent.py                          # LangChain create_agent & CLI runner
│   ├── data.py                           # Mock customer purchase database
│   ├── models.py                         # ChatNVIDIA model configuration
│   ├── objects.py                        # Pydantic structured output models
│   ├── prompts.py                        # System instructions and policy guardrails
│   └── tools.py                          # Refund verification and calculation tools
├── evaluations/                          # DeepEval Test & Benchmark Suite
│   ├── __init__.py
│   ├── answer_correctness.py             # GEval Answer Correctness test runner
│   ├── argument_correctness.py           # Tool argument correctness test runner
│   ├── dataset.py                        # 7 Golden test cases & expected tool calls
│   ├── eval_all.py                       # Master async runner for all 7 metrics
│   ├── model.py                          # NVIDIA NIM evaluation judge & json-repair patch
│   ├── plan_adherence.py                 # Plan adherence evaluation runner
│   ├── plan_quality.py                   # Plan quality evaluation runner
│   ├── step_efficiency.py                # Step efficiency evaluation runner
│   ├── task_completion.py                # Task completion evaluation runner
│   └── tool_correctness.py               # Tool correctness evaluation runner
├── src/
│   └── refund_agent/
│       └── __init__.py                   # Application package entrypoint
├── understanding_metrics/                # Pedagogical Metric Demonstration Suite
│   ├── all_metrics.py                    # Combined educational test runner
│   ├── answer_correctness.py             # GEval metric demo
│   ├── argument_correctness.py           # Argument correctness metric demo
│   ├── plan_adherence.py                 # Plan adherence metric demo
│   ├── plan_quality.py                   # Plan quality metric demo
│   ├── step_efficiency.py                # Step efficiency metric demo
│   ├── task_completion.py                # Task completion metric demo
│   └── tool_correctness.py               # Tool correctness metric demo
├── .env.example                          # Template environment variables
├── .python-version                       # Python version pin (3.13)
├── pyproject.toml                        # Project dependencies and packaging
├── README.md                             # Comprehensive project documentation
└── uv.lock                               # Resolved dependency lockfile
```

---

## 🏁 Getting Started

### Prerequisites

- **Python**: Version `>= 3.13`
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- **NVIDIA API Key**: Obtain a free API key from [NVIDIA Build](https://build.nvidia.com/)

### Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Add your NVIDIA API key:

```ini
NVIDIA_API_KEY="nvapi-your-key-here"
```

### Installation

Install all project dependencies into a managed virtual environment using `uv`:

```bash
uv sync
```

Alternatively, with traditional `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## 💻 Running the Project

### 1. Interactive Agent CLI

Run the interactive multi-turn refund agent in your terminal:

```bash
uv run refund-agent
```

*Example interaction:*
```
I would like to request a refund
reply:
I am user123, I would like a refund for the book and the pen
A refund ticket for 30$ has been issued successfully.
the job is done!
```

### 2. Running the Complete Evaluation Suite

Execute the master evaluation suite across all 7 test cases and 7 metrics simultaneously:

```bash
python evaluations/eval_all.py
```

### 3. Running Individual Metric Tests

You can also run any specific evaluation metric independently:

```bash
# Task Completion
python evaluations/task_completion.py

# Step Efficiency
python evaluations/step_efficiency.py

# Plan Quality
python evaluations/plan_quality.py

# Plan Adherence
python evaluations/plan_adherence.py

# Tool Correctness
python evaluations/tool_correctness.py

# Argument Correctness
python evaluations/argument_correctness.py

# Answer Correctness (GEval)
python evaluations/answer_correctness.py
```

### 4. Running Educational Metrics Examples

To explore how DeepEval scores agents on deliberate flaws, run the understanding metrics suite:

```bash
# Run all educational metric demos
python understanding_metrics/all_metrics.py

# Run individual educational demos
python understanding_metrics/tool_correctness.py
python understanding_metrics/step_efficiency.py
python understanding_metrics/plan_adherence.py
```

---

## 🔧 Troubleshooting & Windows Tips

- **Windows Cache Locks**:
  If you encounter `portalocker` or `test_cases_lookup_map` errors when running evaluations on Windows, ensure test runs use `CacheConfig(use_cache=False, write_cache=False)` as configured in the scripts.
- **Malformed JSON from LLM Judges**:
  If DeepEval throws `JSONDecodeError` during metric evaluation due to quotes in generated reasoning, ensure `evaluations/model.py` is imported first. It patches `deepeval.models.llms.utils.trim_and_load_json` with `json-repair`.
- **NVIDIA Rate Limits**:
  If requests fail with `429 Too Many Requests`, introduce slight delay or run evaluations sequentially rather than with high concurrency.

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
