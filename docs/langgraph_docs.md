# LangGraph — Building Stateful Multi-Actor Applications

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. It extends LangChain's capabilities by providing a graph-based workflow engine with cycles, controllability, and persistence.

---

## Core Concepts

### StateGraph

The `StateGraph` is the central abstraction. It represents a workflow as a directed graph where:
- **Nodes** are Python functions that process and transform state
- **Edges** define the transitions between nodes
- **State** is a typed dictionary shared across all nodes

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    messages: list[str]
    count: int

graph = StateGraph(MyState)
```

### Nodes

Nodes are regular Python functions that take state as input and return a partial state update:

```python
def my_node(state: MyState) -> dict:
    # Return only the keys you want to update
    return {"count": state["count"] + 1}

graph.add_node("my_node", my_node)
```

### Edges

Edges connect nodes:

```python
# Simple edge: always goes from A to B
graph.add_edge("node_a", "node_b")

# Entry point
graph.add_edge(START, "node_a")

# Exit
graph.add_edge("node_b", END)
```

---

## Conditional Edges — The Power of LangGraph

Conditional edges let you route to different nodes based on the current state:

```python
def router(state: MyState) -> str:
    if state["count"] > 5:
        return "done"
    else:
        return "continue"

graph.add_conditional_edges(
    "my_node",        # Source node
    router,           # Function that returns the route name
    {
        "done": END,          # Route name → target node
        "continue": "my_node" # Can loop back!
    }
)
```

---

## Compiling and Running the Graph

```python
compiled = graph.compile()

# Invoke synchronously
result = compiled.invoke({"messages": [], "count": 0})

# Stream step by step
for step in compiled.stream({"messages": [], "count": 0}):
    print(step)

# Async invocation
result = await compiled.ainvoke({"messages": [], "count": 0})
```

---

## Building a RAG Agent with LangGraph

Here's a complete self-corrective RAG pattern:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    relevant_docs: List[Document]
    answer: str
    retry_count: int

def retrieve(state: RAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"retrieved_docs": docs}

def grade_documents(state: RAGState) -> dict:
    relevant = [d for d in state["retrieved_docs"] if is_relevant(d, state["question"])]
    return {"relevant_docs": relevant}

def generate(state: RAGState) -> dict:
    answer = llm_chain.invoke({
        "context": state["relevant_docs"],
        "question": state["question"]
    })
    return {"answer": answer}

def rewrite_query(state: RAGState) -> dict:
    new_q = rewriter.invoke(state["question"])
    return {"question": new_q, "retry_count": state["retry_count"] + 1}

def route_after_grading(state: RAGState) -> str:
    if state["relevant_docs"]:
        return "generate"
    if state["retry_count"] < 2:
        return "rewrite"
    return END

# Build graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", route_after_grading, {
    "generate": "generate",
    "rewrite": "rewrite",
    END: END
})
graph.add_edge("rewrite", "retrieve")  # Loop
graph.add_edge("generate", END)

app = graph.compile()
```

---

## Persistence and Memory

LangGraph supports checkpointing to persist graph state:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
compiled = graph.compile(checkpointer=checkpointer)

# Each invocation with the same thread_id resumes from checkpoint
config = {"configurable": {"thread_id": "user-session-123"}}
result = compiled.invoke({"question": "Hello"}, config=config)

# Follow-up question — graph remembers previous state
result = compiled.invoke({"question": "And what about that?"}, config=config)
```

---

## Human-in-the-Loop

Interrupt the graph at specific nodes for human approval:

```python
compiled = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"]
)

# Graph runs until it hits "sensitive_action", then pauses
result = compiled.invoke(initial_state, config=config)

# Human reviews, then resumes
final = compiled.invoke(None, config=config)
```

---

## Subgraphs

Compose complex workflows by nesting graphs:

```python
sub_graph = build_sub_graph()
main_graph.add_node("sub_workflow", sub_graph)
```

---

## Visualizing the Graph

```python
from IPython.display import Image
Image(compiled.get_graph().draw_mermaid_png())
```

Or get the Mermaid definition:

```python
print(compiled.get_graph().draw_mermaid())
```

---

## Error Handling and Retries

```python
from langgraph.graph import StateGraph

# Add retry logic at the node level
graph.add_node("retrieve", retrieve, retry=RetryPolicy(max_attempts=3))
```

---

## State Reducers

Control how state updates are merged using reducers:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Default: last write wins
    question: str
    
    # Custom reducer: appends to list
    messages: Annotated[list, add_messages]
```

With `add_messages`, each node's update is appended to the list rather than replacing it — essential for conversation history.
