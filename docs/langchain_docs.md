# LangChain — Core Concepts Documentation

## What is LangChain?

LangChain is a framework for developing applications powered by large language models (LLMs). It enables applications that are context-aware (connect a language model to sources of context) and that reason (rely on the language model to reason about how to answer).

---

## Core Components

### 1. LLMs and Chat Models

LangChain provides a standard interface for interacting with many different LLMs:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
response = llm.invoke("What is LangChain?")
print(response.content)
```

### 2. Prompt Templates

Prompt templates help create structured prompts:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm
result = chain.invoke({"topic": "programming"})
```

### 3. Output Parsers

Parse LLM outputs into structured formats:

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# String output
chain = prompt | llm | StrOutputParser()

# JSON output
from pydantic import BaseModel
class Joke(BaseModel):
    setup: str
    punchline: str

parser = JsonOutputParser(pydantic_object=Joke)
chain = prompt | llm | parser
```

---

## LCEL — LangChain Expression Language

LCEL is the recommended way to compose chains using the pipe `|` operator:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_template("Answer the question: {question}")
llm = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

chain = prompt | llm | parser
answer = chain.invoke({"question": "What is LCEL?"})
```

### Streaming with LCEL

```python
for chunk in chain.stream({"question": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

### Async Support

```python
answer = await chain.ainvoke({"question": "What is async?"})
```

---

## Document Loaders

LangChain supports loading documents from many sources:

```python
# Text files
from langchain_community.document_loaders import TextLoader
loader = TextLoader("my_file.txt")
docs = loader.load()

# Markdown
from langchain_community.document_loaders import UnstructuredMarkdownLoader
loader = UnstructuredMarkdownLoader("README.md")
docs = loader.load()

# Web pages
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# PDFs
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()
```

Each document has `page_content` (str) and `metadata` (dict) attributes.

---

## Text Splitters

Split long documents into chunks for retrieval:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
```

### Why RecursiveCharacterTextSplitter?

It tries to keep semantically related pieces of text together by splitting on natural boundaries — paragraphs first, then sentences, then words — only falling to smaller units when necessary.

---

## Vector Stores

Store and retrieve embeddings for similarity search:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Similarity search
results = vectorstore.similarity_search("What is LCEL?", k=3)

# MMR search (more diverse results)
results = vectorstore.max_marginal_relevance_search("What is LCEL?", k=3)
```

---

## Retrievers

Retrievers provide a standard interface for fetching documents:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

docs = retriever.invoke("How do I use LangChain?")
```

### Search Types

- `similarity`: Standard cosine similarity search
- `mmr`: Maximal Marginal Relevance — balances relevance and diversity
- `similarity_score_threshold`: Only returns docs above a score threshold

---

## RAG — Retrieval Augmented Generation

The classic RAG pattern in LangChain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question using only the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LangChain?")
```

---

## Memory and Chat History

Add conversation memory to chains:

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

response = with_history.invoke(
    {"input": "What is LangChain?"},
    config={"configurable": {"session_id": "user-123"}}
)
```

---

## Agents and Tools

LangChain agents use LLMs to decide which tools to call:

```python
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 25°C"

tools = [get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What's the weather in Delhi?"})
```
