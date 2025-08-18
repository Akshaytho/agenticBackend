"""Agentic RAG implementation using LangGraph.

This module encapsulates all the logic for building a retrieval‑augmented
generation (RAG) agent with corrective behaviour. The agent performs the
following steps when answering a user query:

1. **Retrieve** relevant documents from a local vector database based on
   semantic similarity to the query.
2. **Grade** the retrieved documents for relevance using an LLM. If all
   documents are relevant, proceed directly to answer generation; otherwise
   rewrite the query for a web search.
3. **Rewrite** the query with an LLM to optimise it for a web search.
4. **Search** the web using DuckDuckGo to fetch additional context when the
   initial documents are insufficient.
5. **Generate** a final answer using the combined context from both the
   vector store and web results. Answers are grounded in the provided
   context to minimise hallucinations.

In addition to the core RAG flow, the agent maintains a conversational
history to provide long‑term memory across requests. The `AgenticRAG`
class exposes `ingest_files` to add new documents to the vector store and
`run` to answer queries.
"""

from __future__ import annotations

import os
import io
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph
from langgraph.graph import END
from duckduckgo_search import DDGS
import pdfplumber


class RAGState(TypedDict, total=False):
    """State object passed between graph nodes.

    Fields are optional so that nodes can add partial state as needed.
    """

    # Original user query
    query: str
    # Retrieved document texts from the vector store
    docs: List[str]
    # Grade values returned by the grader LLM ("Yes" or "No")
    grades: List[str]
    # Rewritten query for web search
    rewritten_query: str
    # Web search result snippets
    web_results: List[str]
    # Final answer from the generator LLM
    answer: str


class AgenticRAG:
    """High‑level wrapper around a LangGraph‑based RAG agent.

    The class handles persistence of the vector database, ingestion of
    documents, maintenance of conversation history and construction of the
    underlying graph for answering queries.
    """

    def __init__(self, persist_dir: Optional[str] = None, k: int = 4) -> None:
        """Initialise the agent.

        Args:
            persist_dir: Optional directory path where the vector store will be
                persisted. If omitted, defaults to ``./vector_store`` relative
                to this file. The directory will be created if it does not
                exist.
            k: Number of documents to retrieve from the vector store for each
                query.
        """
        self.persist_dir = Path(persist_dir or (Path(__file__).parent / "vector_store"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.k = k

        # Initialise embedding and vector store. If an existing store is
        # detected in ``persist_dir``, it will be loaded automatically.
        self.embedding = OpenAIEmbeddings()
        try:
            # Attempt to load an existing vector store
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedding,
            )
            # Ensure retrieval is configured even if the store is empty
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k}
            )
        except Exception:
            # Create an empty vector store if none exists
            self.vectorstore = None
            self.retriever = None

        # Initialise chat model with deterministic temperature for grading and
        # answering. Model name can be customised via environment variable.
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo-1106")
        self.chat_model = ChatOpenAI(model_name=model_name, temperature=0)

        # Maintain a persistent chat history across queries for long‑term memory
        self.chat_history: List[Dict[str, str]] = []

        # Build the underlying agent graph
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    def ingest_files(self, files: List[io.BytesIO], filenames: List[str]) -> int:
        """Ingest uploaded files into the vector store.

        Args:
            files: List of file‑like objects containing the raw bytes of the
                uploaded documents.
            filenames: List of filenames corresponding to ``files``. Used
                solely for metadata and file type detection.

        Returns:
            The number of document chunks added to the vector store.
        """
        documents: List[Document] = []
        for file_obj, filename in zip(files, filenames):
            # Determine file type from extension
            ext = Path(filename).suffix.lower()
            try:
                if ext in {".pdf"}:
                    # Use pdfplumber to extract text page by page
                    with pdfplumber.open(file_obj) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                else:
                    # Assume a text‑based file (txt, md, csv, etc.) and decode
                    raw_bytes = file_obj.read()
                    # Try UTF‑8 decoding, fall back to ISO‑8859‑1
                    try:
                        text = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_bytes.decode("latin-1")
                # Wrap the extracted text in a LangChain Document with metadata
                documents.append(Document(page_content=text, metadata={"source": filename}))
            finally:
                # Reset pointer for next read if necessary
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)

        if not documents:
            return 0

        # Split into smaller chunks for embedding
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=128, separators=["\n\n", "\n", " "]
        )
        docs = splitter.split_documents(documents)

        # Persist to vector store. Create a new store if it doesn't exist yet.
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                docs,
                self.embedding,
                persist_directory=str(self.persist_dir),
            )
        else:
            self.vectorstore.add_documents(docs)
        # Persist changes to disk
        self.vectorstore.persist()
        # Reconfigure retriever after adding documents
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return len(docs)

    # ------------------------------------------------------------------
    # Query handling
    # ------------------------------------------------------------------
    def run(self, query: str) -> str:
        """Answer a user query using the agent.

        The conversation history is updated with the user query and the
        assistant's answer for long‑term memory.

        Args:
            query: User's question.

        Returns:
            The generated answer.
        """
        state = {"query": query}
        result = self.graph.invoke(state)
        answer = result["answer"]  # type: ignore[index]
        # Append to history: user and assistant messages
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        """Construct the LangGraph for the RAG agent.

        Returns:
            A compiled LangGraph instance ready to invoke.
        """
        graph = StateGraph(RAGState)

        # Define all nodes
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("web_search", self._web_search)
        graph.add_node("generate", self._generate)

        # Define edges between nodes
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade")

        # Add conditional branching after grading
        def decide_next(state: RAGState) -> str:
            grades = state.get("grades", [])
            # If no documents were retrieved or any doc is not relevant, branch to rewrite
            if not grades or any(g.lower().startswith("no") for g in grades):
                return "rewrite_query"
            return "generate"

        graph.add_conditional_edges(
            "grade",
            decide_next,
            {
                "rewrite_query": "rewrite_query",
                "generate": "generate",
            },
        )

        # For the rewrite path
        graph.add_edge("rewrite_query", "web_search")
        graph.add_edge("web_search", "generate")

        # Set final node
        graph.add_edge("generate", END)
        # Compile and return executor
        return graph.compile()

    # --------------------------- Node functions ---------------------------
    def _retrieve(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve relevant documents from the vector store.

        Args:
            state: Current state containing at least the user query.

        Returns:
            A dictionary updating the state with a list of retrieved docs.
        """
        query = state["query"]
        docs_text: List[str] = []
        if self.retriever is not None:
            try:
                docs = self.retriever.get_relevant_documents(query)
                docs_text = [doc.page_content for doc in docs]
            except Exception:
                docs_text = []
        return {"docs": docs_text}

    def _grade(self, state: RAGState) -> Dict[str, Any]:
        """Grade the retrieved documents for relevance.

        Uses the chat model to classify each document as relevant (Yes) or
        irrelevant (No). When no documents are provided, returns an empty
        grades list.
        """
        docs = state.get("docs", []) or []
        query = state["query"]
        grades: List[str] = []
        if docs:
            for doc in docs:
                messages = [
                    {"role": "system", "content": (
                        "You are a document grader. Given a question and a document, "
                        "respond with 'Yes' if the document contains information "
                        "helpful to answer the question, otherwise respond with 'No'. "
                        "Only output 'Yes' or 'No'."
                    )},
                    {"role": "user", "content": (
                        f"Question: {query}\n\nDocument: {doc}\n\nIs this document relevant to answering the question?"
                    )},
                ]
                response = self.chat_model.invoke(messages)
                answer_text = response.content.strip() if hasattr(response, "content") else str(response)
                # Normalise answer to Yes/No
                if answer_text.lower().startswith("yes"):
                    grades.append("Yes")
                else:
                    grades.append("No")
        return {"grades": grades}

    def _rewrite_query(self, state: RAGState) -> Dict[str, Any]:
        """Rewrite the query for web search using an LLM.

        Returns a rewritten query that attempts to capture the core intent of
        the original question, optimised for retrieving information from the
        public web. If no documents exist, the original query is passed in.
        """
        query = state["query"]
        docs = state.get("docs", [])
        context_snippet = "\n\n".join(docs[:2]) if docs else ""
        messages = [
            {"role": "system", "content": (
                "You are a query rewriting assistant. Given a question and optional context, "
                "generate a concise search query that will return results relevant to the question. "
                "Focus on key nouns and phrases. Do not include any personal pronouns or filler words."
            )},
            {"role": "user", "content": (
                f"Question: {query}\n\nContext: {context_snippet}\n\nSearch query:"
            )},
        ]
        response = self.chat_model.invoke(messages)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response)
        # Fallback to original query if rewrite fails
        rewritten_query = rewritten or query
        return {"rewritten_query": rewritten_query}

    def _web_search(self, state: RAGState) -> Dict[str, Any]:
        """Search the web using DuckDuckGo when additional context is needed.

        Args:
            state: Contains the rewritten_query string.

        Returns:
            A dictionary with a list of web result snippets.
        """
        query = state.get("rewritten_query") or state.get("query")
        results: List[str] = []
        if query:
            try:
                
                ddg = DDGS()
                # Use duckduckgo_search to perform the query
                search_results = ddg(query, max_results=3)
                for item in search_results or []:
                    # Each item contains title, href, body (snippet)
                    snippet = f"{item.get('title', '')}: {item.get('body', '')} ({item.get('href', '')})"
                    results.append(snippet)
            except Exception:
                # On any failure return empty list
                results = []
        return {"web_results": results}

    def _generate(self, state: RAGState) -> Dict[str, Any]:
        """Generate a final answer using all available context.

        Combines retrieved documents and web search results to form a single
        context string, then invokes the chat model with the conversation
        history for a contextual answer. The answer is grounded in the
        provided context; if the model cannot answer from the context, it is
        instructed to say "I don't know".
        """
        query = state["query"]
        docs = state.get("docs", []) or []
        web_results = state.get("web_results", []) or []
        # Build context from both sources
        context_sections = []
        if docs:
            context_sections.append("\n\n".join(docs))
        if web_results:
            context_sections.append("\n\n".join(web_results))
        full_context = "\n\n".join(context_sections) if context_sections else ""
        # Build message history: include prior conversation for long‑term memory
        messages = []
        if self.chat_history:
            # Append previous exchanges to maintain continuity
            messages.extend(self.chat_history)
        # Add current system instruction and user request
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful AI assistant that answers user questions using only the provided context. "
                "If the context does not contain sufficient information, respond with \"I don't know\". "
                "Do not make up facts, and do not reference the context explicitly in your answer."
            )
        })
        user_content = f"Question: {query}\n\nContext:\n{full_context}" if full_context else f"Question: {query}"
        messages.append({"role": "user", "content": user_content})
        # Invoke the chat model
        response = self.chat_model.invoke(messages)
        answer_text = response.content.strip() if hasattr(response, "content") else str(response)
        return {"answer": answer_text}
