from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import structlog
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from APP import auth, db, jobs
from APP.auth import email as otp_email
from APP.rag.service import RAGService
from APP.rag.vector_store import warm_reranker
from APP.auth import get_current_user
from APP.jobs import job_store
from APP.schemas import (
    AuthTokenResponse,
    ChatTurnSummary,
    DeleteDocumentResponse,
    DocumentSummary,
    GoogleOAuthCallbackRequest,
    HealthStatus,
    IngestJobStatus,
    IngestResponse,
    LoginRequest,
    OAuthCallbackRequest,
    PrivacyPolicyAcceptanceResponse,
    QueryRequest,
    QueryResponse,
    ResendOtpRequest,
    SignupRequest,
    VerifyOtpRequest,
)

# In-memory OTP resend cooldown — Phase 1's stopgap ahead of Phase 2's
# general slowapi rate limiting; OTP endpoints are a standing abuse target
# regardless of overall traffic, so this can't wait for Phase 2 to land.
_otp_last_sent: dict[str, float] = {}
OTP_RESEND_COOLDOWN_SECONDS = 60

INGEST_RATE_LIMIT = os.getenv("INGEST_RATE_LIMIT", "10/hour")
QUERY_RATE_LIMIT = os.getenv("QUERY_RATE_LIMIT", "60/hour")
LOGIN_RATE_LIMIT = os.getenv("LOGIN_RATE_LIMIT", "10/minute")

# Opt-in rather than environment-sniffed: there's no reliable signal in this
# codebase today for "this is the production deployment" (staging and prod
# run the same image), so defaulting to disabled would silently turn /docs
# off everywhere, including local dev, until someone noticed. Set
# DISABLE_API_DOCS=true on the production Container App to turn it off there.
DISABLE_API_DOCS = os.getenv("DISABLE_API_DOCS", "false").strip().lower() in ("1", "true", "yes")

# Starlette's UploadFile spools to disk past its threshold, so a huge upload
# doesn't blow up memory — but it still costs disk, network and embedding
# time. This is the same bound both /ingest routes read from raw_bytes.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

# Bump this to re-prompt every signed-in user for consent, even ones who
# already accepted an older version — db.has_accepted_privacy_policy compares
# against this exact string, not just "accepted at all".
PRIVACY_POLICY_VERSION = os.getenv("PRIVACY_POLICY_VERSION", "1.0")


def _rate_limit_key(request: Request) -> str:
    """Keys rate limits by the authenticated user, not IP — login is
    mandatory, so every request that reaches these routes has one. Decodes
    the JWT directly rather than depending on get_current_user's Depends
    result, since slowapi's limit check isn't guaranteed to run after
    FastAPI's own dependency resolution. Falls back to IP only for the
    (already-401-bound) case of a missing/invalid token.
    """
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        try:
            return auth.decode_access_token(authorization.removeprefix("Bearer ").strip())
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("rag_api")

# Serialises the check-and-set in /warmup so concurrent callers can't each
# start their own model load.
_warmup_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    service = RAGService()
    app.state.rag_service = service
    app.state.collection_exists = service.collection_exists()
    logger.info(
        "startup_complete",
        collection_name=service.settings.qdrant_collection,
        collection_exists=app.state.collection_exists,
    )
    try:
        yield
    finally:
        logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local RAG API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if DISABLE_API_DOCS else "/docs",
        redoc_url=None if DISABLE_API_DOCS else "/redoc",
        openapi_url=None if DISABLE_API_DOCS else "/openapi.json",
    )

    default_origins = "https://vaibhavgit5048.github.io,http://localhost:3000"
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", default_origins).split(",")
        if origin.strip()
    ]

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start = time.perf_counter()
        request.state.request_id = request_id
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            # Handled here rather than left to Starlette's ServerErrorMiddleware.
            # That one sits outside every user middleware, including CORS, so the
            # 500 it produces carries no Access-Control-Allow-Origin — the browser
            # then blocks the response and the caller sees a generic network
            # failure instead of the actual error. Returning it from inside means
            # CORS still wraps it and the real detail reaches the client.
            logger.exception("unhandled_exception", request_id=request_id)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            status_code = getattr(response, "status_code", 500)
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
            logger.info(
                "request_complete",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
            )

    # Added last on purpose: add_middleware prepends, so the last one registered
    # ends up outermost and therefore wraps the error handling above.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,  # no cookies used; wildcard+credentials was invalid anyway
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-OpenAI-Api-Key"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", str(uuid4()))
        logger.warning(
            "http_exception",
            request_id=request_id,
            status_code=exc.status_code,
            detail=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", str(uuid4()))
        logger.exception("unhandled_exception", request_id=request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )

    def get_service(request: Request) -> RAGService:
        service = getattr(request.app.state, "rag_service", None)
        if service is None:
            service = RAGService()
            request.app.state.rag_service = service
            request.app.state.collection_exists = service.collection_exists()
        return service

    @app.get("/health", response_model=HealthStatus)
    async def health(request: Request) -> HealthStatus:
        service = get_service(request)
        return service.health()

    @app.post("/warmup", status_code=202)
    async def warmup(request: Request) -> dict:
        """Pulls the embedding model and reranker into memory ahead of the first
        question.

        The app runs at minReplicas=0, so an idle period drops the replica and
        the next visitor pays a cold start. /health says nothing about that — it
        touches no model — so before this endpoint existed the entire model-load
        cost landed on whoever asked the first question. Called when the UI
        mounts, it spends that time while the page is being read instead.

        Returns immediately rather than blocking: the caller only needs the work
        started, and a request held open for a model load is exactly the kind of
        thing the ingress times out.
        """
        service = get_service(request)
        state = request.app.state

        # Unauthenticated by design — it has to be callable before sign-in, and
        # on the sign-in page is exactly when starting the wake is most useful.
        # That makes the guard necessary rather than tidy: without it every
        # request would spawn a thread, and a page refreshed a few times would
        # have several model loads racing each other.
        with _warmup_lock:
            if getattr(state, "warm", False):
                return {"status": "warm"}
            if getattr(state, "warming", False):
                return {"status": "warming"}
            state.warming = True

        def _warm() -> None:
            try:
                service.embedding_provider.embed_query("warmup")
                warm_reranker()
                state.warm = True
                logger.info("warmup_complete")
            except Exception:
                # Best-effort: a failed warmup must not affect real requests,
                # which load the same models lazily anyway.
                logger.exception("warmup_failed")
            finally:
                # Cleared either way, so a failed attempt can be retried rather
                # than latching the app into "warming" forever.
                state.warming = False

        threading.Thread(target=_warm, name="warmup", daemon=True).start()
        return {"status": "warming"}

    @app.get("/ready", response_model=HealthStatus)
    async def ready(request: Request) -> HealthStatus:
        service = get_service(request)
        status = service.health()
        if status.qdrant != "up" or status.openai != "up":
            raise HTTPException(status_code=503, detail=status.model_dump())
        return status

    @app.post("/ingest", response_model=IngestResponse)
    @limiter.limit(INGEST_RATE_LIMIT)
    async def ingest(
        request: Request,
        file: UploadFile = File(...),
        chunk_size: int = Form(1000),
        chunk_overlap: int = Form(150),
        quality_threshold: float = Form(4.0),
        current_user=Depends(get_current_user),
    ) -> IngestResponse:
        service = get_service(request)
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
            )

        try:
            response = await asyncio.to_thread(
                service.ingest,
                owner_id=current_user["id"],
                filename=file.filename or "upload",
                raw_bytes=payload,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                quality_threshold=quality_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return response.model_copy(update={"request_id": getattr(request.state, "request_id", response.request_id)})

    @app.post("/ingest/async", response_model=IngestJobStatus, status_code=202)
    @limiter.limit(INGEST_RATE_LIMIT)
    async def ingest_async(
        request: Request,
        file: UploadFile = File(...),
        chunk_size: int = Form(1000),
        chunk_overlap: int = Form(150),
        quality_threshold: float = Form(4.0),
        current_user=Depends(get_current_user),
    ) -> IngestJobStatus:
        """Non-blocking ingest: returns a job id immediately, poll
        /ingest/jobs/{id} for the outcome.

        POST /ingest stays synchronous and is still fine for ordinary
        documents (a 44-page PDF is ~93s). This exists because ingest time
        scales with document size while Azure Container Apps' ingress cuts
        requests at ~240s, and Phase 3's parser router adds per-file OCR calls
        on top — so large documents need a path that isn't bounded by an HTTP
        timeout at all.
        """
        service = get_service(request)
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
            )

        owner_id = current_user["id"]
        filename = file.filename or "upload"
        job = job_store.create(owner_id, filename)

        def _run() -> None:
            job_store.mark(job.id, jobs.RUNNING)
            try:
                result = service.ingest(
                    owner_id=owner_id,
                    filename=filename,
                    raw_bytes=payload,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    quality_threshold=quality_threshold,
                )
                job_store.mark(job.id, jobs.SUCCEEDED, result=result.model_dump())
            except Exception as exc:  # noqa: BLE001 — surfaced via job.error
                logger.exception("ingest_job_failed", job_id=job.id)
                job_store.mark(job.id, jobs.FAILED, error=str(exc))

        # Fire-and-forget on a worker thread. Not awaited: the whole point is
        # that the response returns before the work finishes.
        asyncio.create_task(asyncio.to_thread(_run))
        return IngestJobStatus(**job.to_dict())

    @app.get("/ingest/jobs/{job_id}", response_model=IngestJobStatus)
    async def ingest_job_status(job_id: str, current_user=Depends(get_current_user)) -> IngestJobStatus:
        job = job_store.get(job_id, current_user["id"])
        if job is None:
            # Same 404 whether the job never existed or belongs to someone
            # else, so a job id can't be probed to discover other users' work.
            raise HTTPException(status_code=404, detail="Job not found")
        return IngestJobStatus(**job.to_dict())

    @app.post("/query", response_model=QueryResponse)
    @limiter.limit(QUERY_RATE_LIMIT)
    async def query(
        request: Request,
        payload: QueryRequest,
        current_user=Depends(get_current_user),
        x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-Api-Key"),
    ) -> QueryResponse:
        service = get_service(request)
        try:
            response = await asyncio.to_thread(service.answer, current_user["id"], payload, x_openai_api_key)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return response.model_copy(update={"request_id": getattr(request.state, "request_id", response.request_id)})

    # ----------------------------------------------------------------- #
    # Auth — GitHub/Google OAuth, email+password+OTP. None of these
    # depend on get_current_user; they're what issues the JWT in the
    # first place.
    # ----------------------------------------------------------------- #

    @app.post("/auth/github/callback", response_model=AuthTokenResponse)
    async def github_callback(payload: OAuthCallbackRequest) -> AuthTokenResponse:
        try:
            result = await asyncio.to_thread(auth.exchange_github_code, payload.code)
        except auth.OAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"GitHub sign-in failed: {exc}") from exc

        user_id = db.get_or_create_user_for_oauth(result["email"], "github", result["provider_user_id"])
        token = auth.create_access_token(user_id)
        user = db.get_user_by_id(user_id)
        return AuthTokenResponse(
            access_token=token, user_id=user_id, email=result["email"],
            privacy_policy_accepted=db.has_accepted_privacy_policy(user, PRIVACY_POLICY_VERSION),
        )

    @app.post("/auth/google/callback", response_model=AuthTokenResponse)
    async def google_callback(payload: GoogleOAuthCallbackRequest) -> AuthTokenResponse:
        try:
            result = await asyncio.to_thread(auth.exchange_google_code, payload.code, payload.redirect_uri)
        except auth.OAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Google sign-in failed: {exc}") from exc

        user_id = db.get_or_create_user_for_oauth(result["email"], "google", result["provider_user_id"])
        token = auth.create_access_token(user_id)
        user = db.get_user_by_id(user_id)
        return AuthTokenResponse(
            access_token=token, user_id=user_id, email=result["email"],
            privacy_policy_accepted=db.has_accepted_privacy_policy(user, PRIVACY_POLICY_VERSION),
        )

    def _issue_and_send_otp(user_id: str, email_address: str) -> None:
        code = auth.generate_otp_code()
        db.create_otp(user_id, auth.hash_otp_code(code))
        otp_email.send_otp_email(email_address, code)
        _otp_last_sent[email_address] = time.monotonic()

    @app.post("/auth/signup", status_code=201)
    async def signup(payload: SignupRequest) -> dict:
        existing = db.get_user_by_email(payload.email)
        if existing is not None and existing["email_verified"]:
            # Same response as a genuine signup — do not let this endpoint be
            # used to enumerate which emails already have a verified account.
            # No code is sent for one that already exists and is verified.
            return {"detail": "Verification code sent"}

        if existing is None:
            user_id = db.create_user(payload.email, password_hash=auth.hash_password(payload.password))
        else:
            user_id = existing["id"]  # unverified retry — reuse the row, issue a fresh code

        await asyncio.to_thread(_issue_and_send_otp, user_id, payload.email)
        return {"detail": "Verification code sent"}

    @app.post("/auth/verify-otp", response_model=AuthTokenResponse)
    async def verify_otp(payload: VerifyOtpRequest) -> AuthTokenResponse:
        user = db.get_user_by_email(payload.email)
        if user is None:
            raise HTTPException(status_code=400, detail="Invalid code")

        otp = db.get_latest_otp(user["id"])
        if otp is None or auth.is_otp_expired(otp["expires_at"]) or not auth.verify_otp_code(payload.code, otp["code_hash"]):
            raise HTTPException(status_code=400, detail="Invalid or expired code")

        db.consume_otp(otp["id"])
        db.set_email_verified(user["id"])
        token = auth.create_access_token(user["id"])
        return AuthTokenResponse(
            access_token=token, user_id=user["id"], email=payload.email,
            privacy_policy_accepted=db.has_accepted_privacy_policy(user, PRIVACY_POLICY_VERSION),
        )

    @app.post("/auth/resend-otp", status_code=202)
    async def resend_otp(payload: ResendOtpRequest) -> dict:
        last_sent = _otp_last_sent.get(payload.email)
        if last_sent is not None and (time.monotonic() - last_sent) < OTP_RESEND_COOLDOWN_SECONDS:
            raise HTTPException(status_code=429, detail="Please wait before requesting another code")

        user = db.get_user_by_email(payload.email)
        if user is not None and not user["email_verified"]:
            await asyncio.to_thread(_issue_and_send_otp, user["id"], payload.email)
        # Same response whether or not the account exists/is already
        # verified — do not let this endpoint be used to enumerate emails.
        return {"detail": "If that account needs verification, a new code has been sent"}

    @app.post("/auth/login", response_model=AuthTokenResponse)
    @limiter.limit(LOGIN_RATE_LIMIT)
    async def login(request: Request, payload: LoginRequest) -> AuthTokenResponse:
        user = db.get_user_by_email(payload.email)
        if user is not None and db.is_locked(user):
            raise HTTPException(
                status_code=423, detail="Too many failed attempts. Try again in a few minutes."
            )
        if user is None or not user["password_hash"] or not auth.verify_password(payload.password, user["password_hash"]):
            if user is not None:
                db.record_failed_login(user["id"])
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        if not user["email_verified"]:
            raise HTTPException(status_code=403, detail="Email not verified")

        db.reset_failed_logins(user["id"])
        token = auth.create_access_token(user["id"])
        return AuthTokenResponse(
            access_token=token, user_id=user["id"], email=payload.email,
            privacy_policy_accepted=db.has_accepted_privacy_policy(user, PRIVACY_POLICY_VERSION),
        )

    @app.post("/auth/accept-privacy-policy", response_model=PrivacyPolicyAcceptanceResponse)
    async def accept_privacy_policy(current_user=Depends(get_current_user)) -> PrivacyPolicyAcceptanceResponse:
        db.record_privacy_policy_acceptance(current_user["id"], PRIVACY_POLICY_VERSION)
        return PrivacyPolicyAcceptanceResponse(
            detail="Privacy policy acceptance recorded", privacy_policy_version=PRIVACY_POLICY_VERSION
        )

    # ----------------------------------------------------------------- #
    # Documents — list/resume/delete, all scoped to the authenticated user.
    # ----------------------------------------------------------------- #

    @app.get("/documents", response_model=list[DocumentSummary])
    async def list_documents(request: Request, current_user=Depends(get_current_user)) -> list[DocumentSummary]:
        service = get_service(request)
        return service.list_documents(current_user["id"])

    @app.get("/documents/{document_id}/history", response_model=list[ChatTurnSummary])
    async def document_history(request: Request, document_id: str, current_user=Depends(get_current_user)) -> list[ChatTurnSummary]:
        service = get_service(request)
        return service.get_chat_history(document_id, current_user["id"])

    @app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
    async def delete_document(request: Request, document_id: str, current_user=Depends(get_current_user)) -> DeleteDocumentResponse:
        service = get_service(request)
        deleted = service.remove_document(document_id, current_user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return DeleteDocumentResponse(document_id=document_id, deleted=True)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("APP.main:app", host="0.0.0.0", port=8000, reload=False)
