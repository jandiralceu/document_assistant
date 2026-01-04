# Document Assistant

> Project developed for the **AI Agentic Engineer with LangChain and LangGraph** Nanodegree at [Udacity](https://www.udacity.com/).

## Project Overview

This document assistant uses a multi-agent architecture with LangGraph to handle different types of user requests:

- **Q&A Agent**: Answers specific questions about document content
- **Summarization Agent**: Creates summaries and extracts key points from documents
- **Calculation Agent**: Performs mathematical operations on document data

### Prerequisites

- Python 3.10+
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

![Agent Architecture](./docs/langgraph_agent_architecture.png)

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

Here are authentic examples of how the system handles different intents:

### 1. Q&A (Information Retrieval)

**User**: "What's the total amount in invoice INV-001?"
**Assistant**:

> The total amount in invoice INV-001 is $22,000, which includes a subtotal of $20,000 and a tax of $2,000.

- **INTENT**: `calculation` (likely triggered by "amount", answered via retrieval)
- **TOOLS USED**: `document_reader`

### 2. Summarization

**User**: "Summarize the service agreement"
**Assistant**:

> ### Service Agreement Summary
>
> **Document ID:** CON-001
> **Date:** January 1, 2024
> **Parties Involved:**
>
> - **Provider:** DocDacity Solutions Inc.
> - **Client:** Healthcare Partners LLC
>
> **Services Provided:**
>
> 1. Access to Document Processing Platform
> 2. 24/7 Technical Support
> 3. Monthly Data Analytics Reports
> 4. Compliance Monitoring
>
> **Financial Details:**
>
> - **Monthly Fee:** $15,000
> - **Total Contract Value:** $180,000
>
> **Duration:**
>
> - 12 months
>
> **Termination Clause:**
>
> - Either party may terminate the agreement with a 60-day written notice.
>
> This agreement outlines the services and financial terms between DocDacity Solutions Inc. and Healthcare Partners LLC for a one-year period, with a focus on document processing and support services.

- **INTENT**: `summarization`
- **TOOLS USED**: `document_search`, `document_reader`

### 3. Calculation

**User**: "Calculate the sum of all invoice totals"
**Assistant**:

> The total sum of all invoice totals is $466,250.00.

- **INTENT**: `calculation`
- **TOOLS USED**: `document_search`, `document_reader`, `document_statistics`
