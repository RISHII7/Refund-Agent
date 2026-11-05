# Refund Agent

An AI-powered agent designed to automate and streamline refund request processing, policy validation, and customer resolution.

## Tech Stack & Requirements

- **Python**: >= 3.13
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

## Project Structure

```
.
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore specification
├── .python-version       # Pinned Python version
├── pyproject.toml        # Project configuration & dependencies
├── README.md             # Project documentation
└── src/
    └── refund_agent/
        └── __init__.py   # Application entrypoint
```

## Getting Started

### 1. Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/RISHII7/Refund-Agent.git
cd Refund-Agent
```

### 2. Environment Configuration

Copy the template environment file and supply your API keys:

```bash
cp .env.example .env
```

### 3. Dependency Management

Install dependencies and set up the virtual environment using `uv`:

```bash
uv sync
```

### 4. Running the Application

Execute the application script via `uv`:

```bash
uv run refund-agent
```
