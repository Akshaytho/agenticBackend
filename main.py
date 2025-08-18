"""FastAPI server exposing endpoints for the Agentic RAG application.

This module provides a minimal REST API enabling clients to upload
documents and query the agent for answers. It delegates all heavy
lifting to the `AgenticRAG` class defined in `agent.py`.

Endpoints:

* **POST /upload** – Upload one or more files to the vector database. The
  payload must be multipart/form‑data with a `files` field containing
  the documents. Returns the number of chunks ingested.
* **POST /query** – Ask a question. The body must be JSON with a
  `query` key. Returns the generated answer as plain text.

The server expects an OpenAI API key to be available as an environment
variable named ``OPENAI_API_KEY``. Without it, the application will
raise an error on startup.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List

from agent import AgenticRAG

import io

# Ensure the OpenAI API key is available at startup
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set. Please set it before starting the server."
    )

# Instantiate the agent once at startup to share state across requests
rag_agent = AgenticRAG()

app = FastAPI(title="Agentic RAG Server", version="1.0")

# Allow browser front ends to access the API from any origin (for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Schema for incoming query requests."""

    query: str


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)) -> dict:
    """Endpoint to ingest uploaded files.

    Args:
        files: List of files uploaded by the client.

    Returns:
        A JSON object with the number of ingested chunks.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    file_objs = []
    filenames = []
    for upload in files:
        content = await upload.read()
        file_objs.append(io.BytesIO(content))
        filenames.append(upload.filename or "uploaded_file")
    num_chunks = rag_agent.ingest_files(file_objs, filenames)
    return {"ingested_chunks": num_chunks}


@app.post("/query")
async def ask_question(request: QueryRequest) -> dict:
    """Endpoint to answer a user question.

    Args:
        request: Body containing the user query.

    Returns:
        A JSON object with the answer string.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    answer = rag_agent.run(query)
    return {"answer": answer}
