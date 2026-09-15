from __future__ import annotations

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    chunk_id: int | str | None = None
    source: str = "unknown"
    page: int | str | None = None
    score: float | None = None
    content: str


class IngestResponse(BaseModel):
    request_id: str
    document_id: str
    collection_name: str
    filename: str
    file_type: str
    pdfs: int | None = None
    pages: int = 0
    chunks: int = 0
    passed_chunks: int = 0
    # NOT dropped from the index — every non-empty chunk is indexed. This
    # counts chunks that scored below the quality threshold and therefore
    # carry a 0.7 ranking multiplier at retrieval (apply_quality_penalty).
    penalised_chunks: int = 0
    indexed_chunks: int = 0
    # Which parser tier produced the text, so a user can see when a paid
    # OCR fallback was reached instead of free local extraction.
    parser_used: str | None = None


class IngestJobStatus(BaseModel):
    """Status of an async ingest. `result` is the full IngestResponse
    payload once status is "succeeded", None before that."""
    job_id: str
    status: str
    filename: str
    result: dict | None = None
    error: str | None = None
    created_at: float
    updated_at: float


class QueryRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    model: str
    collection_name: str


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

class OAuthCallbackRequest(BaseModel):
    code: str = Field(min_length=1)


class GoogleOAuthCallbackRequest(OAuthCallbackRequest):
    redirect_uri: str = Field(min_length=1)


class SignupRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)


class VerifyOtpRequest(BaseModel):
    email: str
    code: str = Field(min_length=6, max_length=6)


class ResendOtpRequest(BaseModel):
    email: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthTokenResponse(BaseModel):
    access_token: str
    user_id: str
    email: str
    # False until the user has agreed to the current PRIVACY_POLICY_VERSION —
    # the frontend uses this to gate the mandatory consent modal without a
    # separate round-trip right after signing in.
    privacy_policy_accepted: bool = False


class PrivacyPolicyAcceptanceResponse(BaseModel):
    detail: str
    privacy_policy_version: str


# --------------------------------------------------------------------------- #
# Documents / chat history
# --------------------------------------------------------------------------- #

class DocumentSummary(BaseModel):
    id: str
    filename: str
    ingested_at: str
    pages: int | None = None
    chunks: int | None = None
    indexed_chunks: int | None = None


class ChatTurnSummary(BaseModel):
    id: str
    question: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    created_at: str


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


class HealthStatus(BaseModel):
    status: str
    service: str
    qdrant: str
    openai: str
    collection_name: str | None = None
