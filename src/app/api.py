import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QuestionRequest, 
    QAResponse, 
    ConversationalQARequest, 
    ConversationalQAResponse,
    ConversationTurn
)
from .services.qa_service import answer_question
from .services.indexing_service import index_pdf_file
from .services.session_service import SessionService
from .core.agents.graph import run_conversational_qa_flow

# Database is initialized lazily or via connection pool


app = FastAPI(
    title="Class 12 Multi-Agent RAG Demo",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    # Auto-cleanup sessions older than 7 days
    SessionService.cleanup_old_sessions(7)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover - simple demo handler
    """Catch-all handler for unexpected errors.

    FastAPI will still handle `HTTPException` instances and validation errors
    separately; this is only for truly unexpected failures so API consumers
    get a consistent 500 response body.
    """

    if isinstance(exc, HTTPException):
        # Let FastAPI handle HTTPException as usual.
        raise exc

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    """Submit a question about the vector databases paper.

    US-001 requirements:
    - Accept POST requests at `/qa` with JSON body containing a `question` field
    - Validate the request format and return 400 for invalid requests
    - Return 200 with `answer`, `draft_answer`, and `context` fields
    - Delegate to the multi-agent RAG service layer for processing
    """

    question = payload.question.strip()
    if not question:
        # Explicit validation beyond Pydantic's type checking to ensure
        # non-empty questions.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    # Delegate to the service layer which runs the multi-agent QA graph
    result = answer_question(question)

    return QAResponse(
        answer=result.get("answer", ""),
        context=result.get("context", ""),
    )


@app.post("/qa/conversation", response_model=ConversationalQAResponse, status_code=status.HTTP_200_OK)
async def conversational_qa(payload: ConversationalQARequest) -> ConversationalQAResponse:
    """Submit a question in a conversational context."""
    
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    session_id = payload.session_id
    if not session_id:
        session_id = SessionService.create_session()
    
    # Retrieve history
    history = SessionService.get_history(session_id)
    
    # Run conversational flow
    result = run_conversational_qa_flow(question, history, session_id)
    
    answer = result.get("answer", "")
    context = result.get("context", "") # Context might be missing or None
    
    # Update session history
    SessionService.add_turn(session_id, question, answer, context)
    
    # Return updated history
    updated_history = SessionService.get_history(session_id)
    
    return ConversationalQAResponse(
        answer=answer,
        context=context,
        session_id=session_id,
        history=updated_history
    )


@app.get("/qa/session/{session_id}/history", response_model=list[ConversationTurn], status_code=status.HTTP_200_OK)
async def get_conversation_history(session_id: str) -> list[ConversationTurn]:
    """Retrieve history for a specific session."""
    return SessionService.get_history(session_id)


@app.get("/qa/sessions", status_code=status.HTTP_200_OK)
async def list_sessions() -> list[dict]:
    """List all past chat sessions."""
    return SessionService.list_sessions()


@app.delete("/qa/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """Delete a specific chat session."""
    SessionService.delete_session(session_id)


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    """Upload a PDF and index it into the vector database.

    This endpoint:
    - Accepts a PDF file upload
    - Saves it to the local `data/uploads/` directory
    - Uses PyPDFLoader to load the document into LangChain `Document` objects
    - Indexes those documents into the configured Pinecone vector store
    """

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # upload_dir = Path("data/uploads")
    # upload_dir.mkdir(parents=True, exist_ok=True)

    # file_path = upload_dir / file.filename
    # contents = await file.read()
    # file_path.write_bytes(contents)

    # Create a temporary file to store the upload
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:        
        # Index the saved PDF
        #  chunks_indexed = index_pdf_file(file_path)
        chunks_indexed = index_pdf_file(tmp_path)
    finally:
        # Clean up the temporary file
        if tmp_path.exists():
            os.remove(tmp_path)


    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
        "message": "PDF indexed successfully.",
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("src.app.api:app", host="0.0.0.0", port=port, reload=True)

