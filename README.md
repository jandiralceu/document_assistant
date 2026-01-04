# Document Assistant

## Project Overview

This document assistant uses a multi-agent architecture with LangGraph to handle different types of user requests:

- **Q&A Agent**: Answers specific questions about document content
- **Summarization Agent**: Creates summaries and extracts key points from documents
- **Calculation Agent**: Performs mathematical operations on document data

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository:

```bash
cd <repository_path>
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Assistant

```bash
python main.py
```

## Project Structure

```
doc_assistant_project/
├── src/
│   ├── schemas.py        # Pydantic models
│   ├── retrieval.py      # Document retrieval
│   ├── tools.py          # Agent tools
│   ├── prompts.py        # Prompt templates
│   ├── agent.py          # LangGraph workflow
│   └── assistant.py      # Main agent
├── sessions/             # Saved conversation sessions
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## System Architecture & Implementation Decisions

The system is built using a **Multi-Agent Router Architecture** powered by LangGraph. This design was chosen to provide deterministic control over the conversation flow while allowing flexibility in how individual tasks are handled.

### Core Components

1.  **Multi-Agent Workflow**:
    - **Router (Classifier)**: The entry point (`classify_intent`) uses an LLM to analyze user input and route it to the specific specialist agent (QA, Summarization, or Calculation). This prevents "jack of all trades" hallucinations by narrowing the context.
    - **Specialist Agents**: Each agent (`qa_agent`, `summarization_agent`, `calculation_agent`) is a self-contained unit with access to specific tools and prompts.
    - **Memory Manager**: The `update_memory` node runs after every agent action to synthesize the conversation and update the persistent context.

2.  **Type Safety with Pydantic**:
    - We enforced strict structured outputs using Pydantic models (`UserIntent`, `AnswerResponse`, etc.) defined in `src/schemas.py`.
    - This ensures that the LLM always returns data in a predictable format, preventing parsing errors and allowing downstream logic to rely on specific fields (like `confidence` scores).

3.  **Tool Separation**:
    - Tools are isolated in `src/tools.py` and use the `@tool` decorator.
    - Critically, the **Calculator Tool** uses a safe evaluation method with regex validation to prevent code injection, satisfying security requirements.

## State Management & Memory

State is managed using `LangGraph`'s global state object (`AgentState`), which flows between nodes.

### Persistence Strategy

- **InMemorySaver**: We use an `InMemorySaver` checkpointer to persist the state of the graph. This allows the conversation to pause (e.g., waiting for user input) and resume seamlessly.
- **Session Management**: The `DocumentAssistant` class handles session loading and saving to JSON files in the `sessions/` directory. This ensures that even if the application restarts, the conversation history can be reloaded if the session ID is preserved.

### Context Preservation

The `update_memory` agent is critical for long-running conversations. Instead of feeding the entire raw chat log to every prompt (which wastes tokens), it maintains a `conversation_summary` and a list of `active_documents`. This "compressed" memory is injected into future prompts, keeping the assistant aware of context without exceeding context windows.

## Structured Output Enforcement

To ensure reliability, we do not rely on raw string parsing. Instead, we use `llm.with_structured_output(Schema)`:

- **UserIntent**: Forces the LLM to categorize input into `qa`, `summarization`, or `calculation`. If the confidence is low, the routing logic can handle it gracefully.
- **Agent Responses**: Each agent returns a specific schema (e.g., `AnswerResponse` containing `sources` and `confidence`), ensuring the UI always has the metadata it needs to display citations and trust scores.

## Example Conversations

Here are examples of how the system handles different intents:

### 1. Q&A (Information Retrieval)

**User**: "What is the total amount for invoice INV-001?"
**Assistant**:

- _Intent_: `qa`
- _Action_: Retrieves `INV-001`.
- _Response_: "The total amount for invoice INV-001 is $22,000.00."
- _Sources_: `['INV-001']`

### 2. Summarization

**User**: "Can you summarize the service agreement?"
**Assistant**:

- _Intent_: `summarization`
- _Action_: Retrieves `CON-001` (Service Agreement).
- _Response_: "The Service Agreement between DocDacity Solutions and Healthcare Partners LLC is for a 12-month term valued at $180,000. It covers document processing, support, analytics, and compliance."
- _Key Points_: `['12-month duration', '$180,000 total value', 'Includes analytics and support']`

### 3. Calculation

**User**: "Calculate 15% tax on the invoice total of $50,000."
**Assistant**:

- _Intent_: `calculation`
- _Tool Used_: `calculator` (Input: "50000 \* 0.15")
- _Response_: "The calculated tax is $7,500."
- _Explanation_: "15% of $50,000 is calculated by multiplying 50000 by 0.15."
