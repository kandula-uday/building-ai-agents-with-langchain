# Module 2 — Study Notes

These notes summarize the Module 2 notebooks: MCP, runtime context, state, multi-agent systems, the wedding planner, and the bonus SQL and RAG examples.

## 1. The big picture

1. **MCP** connects an AI application to tools and data through a standard protocol.
2. **Runtime context** supplies run-specific information, such as a user's preferences.
3. **State** holds working information that can change as the agent runs.
4. **Multi-agent systems** delegate work to specialized agents.
5. **SQL and RAG** give agents access to structured data and documents.

Use the simplest approach that fits the task. An agent does not need multiple specialists or an MCP server just to answer a simple question.

## 2. MCP: Model Context Protocol

### Subject

MCP is a standard protocol that lets applications discover and use capabilities provided by an **MCP server**. A client connects to the server; the server can expose:

- **Tools:** actions such as web searches or database lookups.
- **Resources:** data such as a document.
- **Prompts:** reusable instruction templates.

```text
User -> agent -> MCP client <-> MCP server -> API, database, or files
```

The server is not the language model. It provides capabilities that an agent application can use. MCP does not itself guarantee that information is current or that an action is authorized.

### Why use it?

With normal LangChain tools, you can pass Python tool objects directly to an agent. With MCP, tools can live in a separate server and be reused by different MCP-compatible clients.

### Server syntax

The local server in [2.1_mcp_server.py](./resources/2.1_mcp_server.py) uses `FastMCP`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("example_server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet the supplied person."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

`@mcp.tool()` exposes the function as an MCP tool. `stdio` lets a client start and communicate with the server as a local process.

### Client syntax

```python
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({
    "local_server": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["resources/2.1_mcp_server.py"],
    }
})

tools = await client.get_tools()
agent = create_agent(model="gpt-5-nano", tools=tools)

response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Search for LangChain information."}]
})
print(response["messages"][-1].content)
```

- `sys.executable` selects the Python running the notebook; that environment needs the server's dependencies.
- The relative server path depends on the notebook's working directory.
- `await` is appropriate in a Jupyter cell for these asynchronous calls.
- The example server's web search requires its configured Tavily credentials.

The notebook also obtains resources and prompts using `client.get_resources(...)` and `client.get_prompt(...)`. Retrieving a resource does not automatically put it into the model's messages; the application must use or pass the retrieved content.

### Local and remote connections

The time-server example uses `stdio`, even though it appears under an “Online MCP” heading. Its original command uses `uv`; `FileNotFoundError: 'uv'` means the notebook process cannot find that executable. Install `uv` and make it available on the notebook's PATH, or use the notebook's Python directly if `mcp_server_time` is installed in that environment.

The [travel-agent notebook](./2.1_travel_agent.ipynb) uses `streamable_http` to connect to a remote MCP endpoint. Remote services may require network access, authentication, and provider availability.

## 3. Runtime context

### Subject

Runtime context is information supplied by the application for a particular agent run, such as a user preference or authenticated user ID. It is normally treated as read-only during that run.

### Basic syntax

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

@dataclass
class ColourContext:
    favourite_colour: str = "blue"

@tool
def get_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the user's favourite colour."""
    return runtime.context.favourite_colour

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_favourite_colour],
    context_schema=ColourContext,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my favourite colour?"}]},
    context=ColourContext(favourite_colour="green"),
)
print(response["messages"][-1].content)
```

- `context_schema` describes the context supplied to the agent.
- `runtime.context` gives the tool access to it.
- The context is not automatically sent to the model. A tool, prompt, or other logic must use it if the model needs the information.
- For security-sensitive values, such as a user ID, use a trusted authentication source rather than model-generated input.

## 4. State: working data

### Subject

State contains information used and updated during the workflow. Messages are part of agent state; you can add custom fields.

```python
from langchain.agents import AgentState

class ColourState(AgentState):
    favourite_colour: str
```

`state_schema=ColourState` tells the agent about the additional field. Defining it does not automatically extract a value from the user's message.

### Read and update state in tools

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def save_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Save the user's favourite colour."""
    return Command(update={
        "favourite_colour": favourite_colour,
        "messages": [
            ToolMessage("Colour saved.", tool_call_id=runtime.tool_call_id)
        ],
    })

@tool
def read_colour(runtime: ToolRuntime) -> str:
    """Read the saved favourite colour."""
    return runtime.state.get("favourite_colour", "No colour saved.")
```

`runtime.state` reads the current state. `Command(update=...)` requests a state update; the `ToolMessage` records the result for the corresponding tool call.

### Keep state across calls

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5-nano",
    tools=[save_colour, read_colour],
    state_schema=ColourState,
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "colour-demo"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "My favourite colour is green."}]},
    config=config,
)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my favourite colour?"}]},
    config=config,
)
```

The checkpointer associates saved state with the thread ID. `InMemorySaver` is for demonstrations: its data is not durable across process restarts. The model must choose to call the relevant tools for this example to save and read the custom field.

### State versus runtime context

| | State | Runtime context |
|---|---|---|
| Main purpose | Working information that may change | Information supplied by the application for a run |
| Example | Conversation messages, current order | Authenticated user ID, user preference |
| Tool access | `runtime.state` | `runtime.context` |

A value can sometimes fit either place. Choose based on whether the workflow needs to update it and how the application supplies it.

## 5. Multi-agent systems

### Subject

A multi-agent system uses multiple agents with different responsibilities. A **coordinator** can delegate a task to a specialist and combine results.

```text
User -> coordinator -> specialist agent -> coordinator -> answer
```

### Basic syntax: a specialist as a tool

```python
from langchain.agents import create_agent
from langchain.tools import tool

specialist = create_agent(
    model="gpt-5-nano",
    system_prompt="You are a maths specialist. Explain your reasoning clearly.",
)

@tool
def ask_specialist(question: str) -> str:
    """Ask the maths specialist a question."""
    result = specialist.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content

coordinator = create_agent(
    model="gpt-5-nano",
    tools=[ask_specialist],
    system_prompt="Delegate maths questions to the specialist.",
)
```

The wrapper tool passes a focused message to the specialist and returns its answer. The specialist does not automatically receive the coordinator's entire state; pass the information it needs. Each agent can have its own tools, instructions, and message history.

The [multi-agent notebook](./2.3_multi_agent.ipynb) uses separate square-root and square specialists. The [wedding planner](./2.4_wedding_planners.ipynb) coordinates travel, venue, and playlist specialists. Its coordinator records wedding details in state, and wrapper tools read the relevant fields before invoking each specialist.

Model decisions are not guaranteed: prompts guide delegation but do not prove that the correct tools were used or that results are accurate.

## 6. Bonus: SQL tools

The [SQL notebook](./bonus_sql.ipynb) connects to a SQLite database and exposes a query tool:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")
print(db.run("SELECT * FROM Artist LIMIT 10"))
```

The notebook wraps `db.run(query)` in a tool so an agent can request queries. The path is relative to the working directory.

**Production caution:** Arbitrary model-generated SQL can read or change unintended data. Use a read-only connection where appropriate, least-privilege credentials, query restrictions, timeouts, and separate authorization. A prompt alone is not a security boundary.

## 7. Bonus: RAG and semantic search

**Retrieval-augmented generation (RAG)** retrieves relevant source material and gives it to a model to help answer a question.

```text
PDF -> document chunks -> embeddings -> vector store
Question -> similarity search -> relevant chunks -> agent answer
```

The [RAG notebook](./bonus_rag.ipynb) loads a PDF, splits it, embeds the chunks, and searches an in-memory vector store:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

documents = PyPDFLoader("resources/acmecorp-employee-handbook.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-large"))
store.add_documents(chunks)
results = store.similarity_search("How much vacation do new employees get?")
```

An agent tool can return retrieved content to the model. Retrieval improves access to source material but does not guarantee a correct answer; verify sources and handle empty results. An in-memory vector store is not durable storage.

## 8. Quick revision

- **MCP:** standard interface between applications and external capabilities.
- **Context:** run-specific information provided by the application.
- **State:** changing working information; a checkpointer can preserve it across calls.
- **Multi-agent:** specialized agents coordinated for a larger task.
- **SQL:** queries structured data; restrict access and validate operations.
- **RAG:** retrieves relevant documents to support an answer.
