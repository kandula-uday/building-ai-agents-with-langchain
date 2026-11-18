# Module 3 — Study Notes

These notes summarize the Module 3 notebooks: middleware, message management, human approval, dynamic models, prompts and tools, and the email-agent example.

## 1. The big picture

**Middleware** adds behavior at specific points in an agent's workflow. It can inspect or change a model request, manage messages, or pause before a tool executes.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    middleware=[...],
)
```

Middleware changes how the application runs; it is not itself the language model. Use it when behavior needs to be applied consistently around the agent's work.

## 2. Managing messages

### Why manage messages?

Long conversations can consume more tokens, increase cost, and eventually exceed a model's context limit. Message management helps control what remains available to the agent.

| Approach | What happens? |
|---|---|
| Summarize | Compress older information into a shorter summary |
| Trim | Keep a selected portion, such as recent messages |
| Delete | Remove selected messages from the current state |

Summaries may omit details, and deleted messages are no longer available in the active state. Removing them from active state does not necessarily erase existing checkpoints or logs.

### Summarization syntax

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5-nano",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 100),
            keep=("messages", 1),
        )
    ],
)
```

- `model` is used to create the summary.
- `trigger` sets when summarization should happen.
- `keep` controls how much recent history is retained.
- The notebook's low threshold is useful for demonstration, not necessarily suitable for production.

### Remove selected messages

The [message-management notebook](./3.2_managing_messages.ipynb) uses custom middleware to remove tool messages:

```python
from typing import Any
from langchain.agents import AgentState
from langchain.agents.middleware import before_agent
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

@before_agent
def remove_tool_messages(state: AgentState, runtime: Runtime) -> dict[str, Any]:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}
```

Pass `middleware=[remove_tool_messages]` to `create_agent`. `@before_agent` runs this hook before the agent execution. Take care to preserve valid message sequences: deleting a tool result while retaining the corresponding tool call can cause provider errors. A production trimming strategy must preserve required tool-call/result pairs.

## 3. Human in the loop

### Subject

Human-in-the-loop (HITL) middleware pauses before configured tool calls so a person can review an action. This is useful for actions such as sending emails, making payments, or modifying data.

### Configure approval

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5-nano",
    tools=[read_email, send_email],  # Define these tools first.
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        )
    ],
)
```

`False` lets that tool execute without this approval step. `True` pauses before execution. These settings configure the middleware; they do not change the tool definition.

### Inspect and resume

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "email-demo"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Read my email and reply."}]},
    config=config,
)

# If the agent requests an approval-protected action:
print(response["__interrupt__"])

# After a human reviews the proposed action:
response = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
```

Other decisions include `{"type": "reject", "message": "Please revise."}` and `edit` with an `edited_action` containing the tool name and replacement arguments. Use the same thread ID when resuming. The interrupt exists only if a protected tool call actually occurs.

The [HITL notebook](./3.3_hitl.ipynb) prints the proposed email body and demonstrates separate approve, reject, and edit cells. It does **not** create an approval UI automatically, and its email tool simulates sending. In a real application, collect the decision from an authenticated reviewer and use a durable checkpointer if pause/resume must survive a restart.

## 4. Dynamic models

### Subject

Dynamic model selection chooses a model for a particular request, for example based on conversation length or task requirements.

### Imports and boilerplate

```python
from typing import Callable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model

standard_model = init_chat_model("gpt-5-nano")
large_model = init_chat_model("claude-sonnet-4-6")

@wrap_model_call
def choose_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    model = large_model if len(request.messages) > 10 else standard_model
    return handler(request.override(model=model))

agent = create_agent(
    model="gpt-5-nano",
    middleware=[choose_model],
)
```

- `@wrap_model_call` wraps a model call.
- `request.messages` gives access to the request's messages.
- `request.override(model=...)` creates a request with the selected model.
- `handler(...)` continues the model call.

Message count is a simple example, not a token count or a reliable measure of difficulty. Each provider needs the appropriate integration, credentials, and model access.

## 5. Dynamic prompts

### Subject

A dynamic prompt generates instructions based on the request or runtime context. The notebook uses the user's language preference.

### Imports and boilerplate

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class LanguageContext:
    user_language: str = "English"

@dynamic_prompt
def language_prompt(request: ModelRequest) -> str:
    language = request.runtime.context.user_language
    return f"You are a helpful assistant. Respond in {language}."

agent = create_agent(
    model="gpt-5-nano",
    context_schema=LanguageContext,
    middleware=[language_prompt],
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    context=LanguageContext(user_language="Spanish"),
)
```

The application supplies the context; the middleware reads it and generates the system prompt. Prompts guide model behavior but cannot guarantee compliance.

## 6. Dynamic tools

### Subject

Dynamic tool selection changes which tools are presented to the model. For example, an internal workflow might offer a database lookup, while an external workflow offers only public search.

### Imports and boilerplate

```python
from dataclasses import dataclass
from typing import Callable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool

@dataclass
class UserContext:
    role: str = "external"

@tool
def public_search(query: str) -> str:
    """Search public information."""
    return f"Public search requested: {query}"

@tool
def internal_lookup(query: str) -> str:
    """Look up internal information."""
    return f"Internal lookup requested: {query}"

@wrap_model_call
def choose_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime.context.role == "internal":
        available = [public_search, internal_lookup]
    else:
        available = [public_search]
    return handler(request.override(tools=available))

agent = create_agent(
    model="gpt-5-nano",
    tools=[public_search, internal_lookup],
    context_schema=UserContext,
    middleware=[choose_tools],
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Find order information."}]},
    context=UserContext(role="external"),
)
```

These tool bodies are placeholders. `request.override(tools=...)` controls which tools the model is offered for that call.

**Security:** Hiding a tool from the model is not authorization. Obtain identity and roles from trusted authentication and enforce permissions in the tool or backend. Do not rely on an LLM or a prompt to protect data.

## 7. Email-agent example

The [email-agent notebook](./3.5_email_agent.ipynb) combines several concepts:

1. **Context** carries example credentials.
2. **State** tracks whether the example considers the user authenticated.
3. **Dynamic tools and prompts** change the capabilities and instructions offered.
4. **Human-in-the-loop middleware** pauses before sending an email.
5. A **checkpointer** maintains the workflow across interactions.

This is a teaching demonstration with hardcoded sample credentials and simulated email behavior—not a production authentication system. Real applications should use established identity services, protect secrets, enforce permissions in the backend, and audit sensitive actions.

## 8. Quick revision

- **Middleware:** adds behavior around agent steps.
- **Summarize:** compress older conversation; details may be lost.
- **Trim/delete:** remove selected messages; preserve valid message sequences.
- **HITL:** pause a configured tool call for human review and resume with a decision.
- **Dynamic model:** choose which model handles a request.
- **Dynamic prompt:** choose instructions based on request or context.
- **Dynamic tools:** choose which tools the model is offered; enforce security separately.
