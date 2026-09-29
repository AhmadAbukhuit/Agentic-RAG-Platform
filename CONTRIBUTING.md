# Contributing to Agentic RAG Platform

Thank you for your interest in contributing to the **Agentic RAG Development Platform**! We welcome bug fixes, architecture improvements, new agent tools, documentation updates, and benchmark enhancements.

---

## 📋 Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please treat everyone with respect and empathy.

---

## 🛠️ Local Development Setup

### 1. Prerequisites

* **Python 3.12+**
* **Docker & Docker Compose** (recommended for running Qdrant, Redis, and MongoDB)
* **Git**

### 2. Fork & Clone

```bash
git clone https://github.com/AhmadAbukhuit/LangChain-LangGraph-LangFlow-LangSmith.git
cd LangChain-LangGraph-LangFlow-LangSmith
```

### 3. Environment Setup

```bash
cp .env.example .env
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install ruff bandit[toml] pytest pytest-asyncio
```

---

## 🏗️ Architectural & Coding Standards

To maintain this repository as a high-quality production template, please adhere to these design principles:

1. **State Isolation**:
   * All state schemas must reside in [`app/state/`](./app/state).
   * Use `TypedDict(..., total=False)` so nodes only return partial dictionary updates.
   * Prefer replacement semantics for candidate passages (`documents: List[str]`) and reducers only where accumulation is required (`messages: Annotated[List[BaseMessage], add_messages]`).

2. **Decoupled Prompts**:
   * Never hardcode prompts inside graph nodes.
   * Place all prompt templates inside [`app/prompts/`](./app/prompts) as `ChatPromptTemplate` instances with clear system and human roles.

3. **Resilient Local Fallbacks**:
   * When integrating external services (databases, vector stores, search APIs), wrap them with graceful local fallbacks (see [`app/core/cache.py`](./app/core/cache.py) and [`app/tools/memory_store.py`](./app/tools/memory_store.py)) so that unit tests can run offline without external infrastructure.

4. **Type Annotations & Docstrings**:
   * Every new function must include complete type annotations and docstrings explaining input arguments, return state slices, and architectural purpose.

---

## 🔍 Pre-Commit Checks & Tooling

Before opening a Pull Request, ensure all linters, security scanners, and tests pass:

### 1. Ruff Code Linting & Formatting

```bash
# Check and auto-fix linting issues
ruff check --fix .

# Verify formatting
ruff format --check .
```

### 2. Bandit AST Security Scan

```bash
# Run security analysis
bandit -r app/ -c pyproject.toml
```

### 3. Run Test Suite

```bash
# Execute unit tests
pytest tests/test_pipeline.py -v

# Run local evaluation benchmark
python tests/run_evaluation.py --local
```

---

## 🚀 Pull Request Process

1. **Create a Feature Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Commit Changes**: Write descriptive, imperative commit messages (e.g. `feat: add hybrid search reranker node` or `fix: handle empty context in crag grading`).
3. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request**:
   * Reference any relevant GitHub Issue (e.g. `Fixes #12`).
   * Provide a clear summary of what was added or changed.
   * Confirm that all CI checks (Ruff, Bandit, Pytest) are green.

Thank you for helping make this Agentic RAG platform better!
