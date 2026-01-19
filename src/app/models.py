from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint.

    From the API consumer's perspective we only expose the final,
    verified answer plus some metadata (e.g. context snippets).
    Internal draft answers remain inside the agent pipeline.
    """

    answer: str
    context: str

from typing import List, Optional

class ConversationalQARequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ConversationTurn(BaseModel):
    turn: int
    question: str
    answer: str
    context: Optional[str] = None
    timestamp: str

class ConversationalQAResponse(BaseModel):
    answer: str
    context: Optional[str]
    session_id: str
    history: List[ConversationTurn]
