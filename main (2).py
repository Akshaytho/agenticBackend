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

# Attempt to read the OpenAI API key from the environment.  Rather than raising
# an exception here (which would cause the entire application to fail and
# subsequently surface as a 500 error on the frontend), we defer error handling
# until runtime.  This allows the server to start up cleanly and return a
# descriptive error message from the endpoints if the API key is missing.
openai_api_key = os.getenv("OPENAI_API_KEY")

# Instantiate the agent once at startup to share state across requests.  If
# instantiation fails (for example, because the API key is missing or invalid),
# ``rag_agent`` will remain ``None`` and each endpoint will check for this
# condition and respond appropriately.  Wrapping the creation in a try/except
# prevents a hard crash during module import.
rag_agent: AgenticRAG | None = None
try:
    # Only attempt to instantiate the agent if the API key appears to be
    # configured.  Missing or empty values will still be passed through and
    # allowed here, because downstream components (LangChain/OpenAI) handle
    # authentication internally.  Should an error occur, the except block
    # captures it and logs a message without stopping application startup.
    rag_agent = AgenticRAG()
except Exception as e:
    # Log the error for debugging purposes.  In a production setting you might
    # use a proper logging framework rather than printing to stdout.
    import sys
    print(f"Error initialising AgenticRAG: {e}", file=sys.stderr)
    rag_agent = None

app = FastAPI(title="Agentic RAG Server", version="1.0")

# Allow browser front ends to access the API from any origin (for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Cache CORS preflight responses for one hour.  This reduces the number of
    # preflight requests the browser needs to make and can improve performance.
    max_age=3600,
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
    # Ensure the RAG agent has been initialised successfully.  If not, return
    # a clear error message instead of causing a server exception.  A missing
    # API key or other initialisation problem will result in ``rag_agent``
    # remaining ``None``.
    if rag_agent is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The server could not be initialised. Ensure that the OpenAI API key "
                "is set via the OPENAI_API_KEY environment variable and that all "
                "dependencies are installed correctly."
            ),
        )
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
    # As with the upload endpoint, verify that the agent is ready to serve
    # requests.  If initialisation failed earlier, inform the client rather
    # than triggering an internal error that would otherwise manifest as a
    # generic 500 response with a misleading CORS message in the browser.
    if rag_agent is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The server could not be initialised. Ensure that the OpenAI API key "
                "is set via the OPENAI_API_KEY environment variable and that all "
                "dependencies are installed correctly."
            ),
        )
    answer = rag_agent.run(query)
    return {"answer": answer}
