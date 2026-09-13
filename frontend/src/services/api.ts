// The only place in the app that calls fetch. Every function is typed, every
// request has an AbortController timeout, and every failure surfaces the
// FastAPI `detail` string plus the X-Request-ID for debuggability.

import { API_BASE_URL, BYO_OPENAI_KEY_HEADER, ROUTES, STORAGE_KEYS, TIMEOUTS } from '@/config'
import type {
  AuthTokenResponse,
  ChatTurnSummary,
  CollectionInfo,
  DeleteCollectionResponse,
  DeleteDocumentResponse,
  DocumentSummary,
  HealthStatus,
  IngestJobStatus,
  IngestParams,
  IngestResponse,
  QueryResponse,
} from '@/types/api'

export class ApiError extends Error {
  readonly status: number | null
  readonly detail: string
  readonly requestId: string | null

  constructor(status: number | null, detail: string, requestId: string | null) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.requestId = requestId
  }

  /** 404 from /query means the document has no index yet — an empty state, not an error. */
  get isNotIngested() {
    return this.status === 404
  }

  /** 401 means the stored token is missing, expired, or was issued by another environment. */
  get isUnauthorized() {
    return this.status === 401
  }

  /** 429 from the per-user rate limiter. */
  get isRateLimited() {
    return this.status === 429
  }
}

interface RequestOptions {
  method?: string
  body?: BodyInit
  headers?: Record<string, string>
  timeout: number
  signal?: AbortSignal
  /** Auth endpoints must not trigger the global sign-out on 401 — a wrong
   *  password is a normal outcome there, not an expired session. */
  skipAuthRedirect?: boolean
}

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return localStorage.getItem(STORAGE_KEYS.jwt)
  } catch {
    return null
  }
}

/** Cleared on 401 so a stale token can't wedge the UI in a signed-in state
 *  that every request rejects. The listener lets useAuth react without this
 *  module importing React. */
function clearSession() {
  try {
    localStorage.removeItem(STORAGE_KEYS.jwt)
    localStorage.removeItem(STORAGE_KEYS.user)
  } catch {
    /* private browsing */
  }
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new Event('rag:signed-out'))
  }
}

async function request<T>(path: string, opts: RequestOptions): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), opts.timeout)

  // Respect an externally supplied signal (component unmount) as well as the timeout.
  const onExternalAbort = () => controller.abort()
  opts.signal?.addEventListener('abort', onExternalAbort)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: opts.method ?? 'GET',
      body: opts.body,
      headers: opts.headers,
      signal: controller.signal,
    })
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === 'AbortError'
    throw new ApiError(
      null,
      aborted
        ? `Request timed out after ${Math.round(opts.timeout / 1000)}s`
        : 'Could not reach the backend. It may be waking up — try again in a moment.',
      null,
    )
  } finally {
    clearTimeout(timer)
    opts.signal?.removeEventListener('abort', onExternalAbort)
  }

  const headerRequestId = response.headers.get('X-Request-ID')

  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    // Some error paths return an empty body; fall through with payload = null.
  }

  const bodyRecord = (payload ?? {}) as Record<string, unknown>
  const bodyRequestId = typeof bodyRecord.request_id === 'string' ? bodyRecord.request_id : null

  if (!response.ok) {
    if (response.status === 401 && !opts.skipAuthRedirect) clearSession()
    const detail =
      typeof bodyRecord.detail === 'string'
        ? bodyRecord.detail
        : `HTTP ${response.status} ${response.statusText}`.trim()
    throw new ApiError(response.status, detail, bodyRequestId ?? headerRequestId)
  }

  return payload as T
}

/** Bearer token on every authenticated route. Replaces the old X-Api-Key
 *  shared secret, which was a static string rather than real auth. */
function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra }
}

const jsonHeaders = { 'Content-Type': 'application/json' }

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/** `cold` gives the request the much longer budget a scale-to-zero wake needs;
 *  a warm probe should still fail fast so the UI notices a real outage. */
export function getHealth(signal?: AbortSignal, cold = false) {
  return request<HealthStatus>(ROUTES.health, {
    timeout: cold ? TIMEOUTS.healthCold : TIMEOUTS.health,
    signal,
  })
}

/** Fired when the app mounts, to absorb a scale-to-zero cold start during the
 *  seconds the user spends reading the page or choosing a file.
 *
 *  Hits /warmup, not /health: /health touches no model, so it wakes the
 *  container but leaves the embedding model and reranker to load on the first
 *  question — which is where the wait was actually being felt. /warmup starts
 *  that load in the background and returns immediately.
 */
export function warmup() {
  return request<{ status: string }>(ROUTES.warmup, {
    method: 'POST',
    timeout: TIMEOUTS.healthCold,
  }).catch(() => null)
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export function signUpEmail(email: string, password: string) {
  return request<{ detail: string }>(ROUTES.authSignup, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ email, password }),
    timeout: TIMEOUTS.auth,
    skipAuthRedirect: true,
  })
}

export function verifyOtp(email: string, code: string) {
  return request<AuthTokenResponse>(ROUTES.authVerifyOtp, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ email, code }),
    timeout: TIMEOUTS.auth,
    skipAuthRedirect: true,
  })
}

export function resendOtp(email: string) {
  return request<{ detail: string }>(ROUTES.authResendOtp, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ email }),
    timeout: TIMEOUTS.auth,
    skipAuthRedirect: true,
  })
}

export function signInEmail(email: string, password: string) {
  return request<AuthTokenResponse>(ROUTES.authLogin, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ email, password }),
    timeout: TIMEOUTS.auth,
    skipAuthRedirect: true,
  })
}

export function signInWithGithub(code: string) {
  return request<AuthTokenResponse>(ROUTES.authGithub, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ code }),
    timeout: TIMEOUTS.auth,
    skipAuthRedirect: true,
  })
}

export function signInWithGoogle(code: string, redirectUri: string) {
  return request<AuthTokenResponse>(ROUTES.authGoogle, {
    method: 'POST',
    headers: jsonHeaders,
    // Google validates redirect_uri again at token exchange, so the backend
    // must be told the exact value the browser used.
    body: JSON.stringify({ code, redirect_uri: redirectUri }),
    timeout: TIMEOUTS.auth,
    skipAuthRedirect: true,
  })
}

export function acceptPrivacyPolicy() {
  return request<{ detail: string; privacy_policy_version: string }>(ROUTES.authAcceptPrivacyPolicy, {
    method: 'POST',
    headers: authHeaders(),
    timeout: TIMEOUTS.auth,
  })
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export function listDocuments(signal?: AbortSignal) {
  return request<DocumentSummary[]>(ROUTES.documents, {
    headers: authHeaders(),
    timeout: TIMEOUTS.health,
    signal,
  })
}

export function getChatHistory(documentId: string, signal?: AbortSignal) {
  return request<ChatTurnSummary[]>(ROUTES.documentHistory(documentId), {
    headers: authHeaders(),
    timeout: TIMEOUTS.health,
    signal,
  })
}

export function deleteDocument(documentId: string) {
  return request<DeleteDocumentResponse>(ROUTES.document(documentId), {
    method: 'DELETE',
    headers: authHeaders(),
    timeout: TIMEOUTS.health,
  })
}

// ---------------------------------------------------------------------------
// Ingest + query
// ---------------------------------------------------------------------------

function ingestForm({ file, chunkSize, chunkOverlap, qualityThreshold }: IngestParams) {
  const form = new FormData()
  form.append('file', file)
  form.append('chunk_size', String(chunkSize))
  form.append('chunk_overlap', String(chunkOverlap))
  form.append('quality_threshold', String(qualityThreshold))
  return form
}

export function ingest(params: IngestParams) {
  // No Content-Type header: the browser must set the multipart boundary itself.
  return request<IngestResponse>(ROUTES.ingest, {
    method: 'POST',
    body: ingestForm(params),
    headers: authHeaders(),
    timeout: TIMEOUTS.ingest,
  })
}

/** Returns a job id immediately. Preferred for large files: a synchronous
 *  ingest is bounded by the hosting platform's request timeout, this isn't. */
export function ingestAsync(params: IngestParams) {
  return request<IngestJobStatus>(ROUTES.ingestAsync, {
    method: 'POST',
    body: ingestForm(params),
    headers: authHeaders(),
    timeout: TIMEOUTS.auth,
  })
}

export function getIngestJob(jobId: string) {
  return request<IngestJobStatus>(ROUTES.ingestJob(jobId), {
    headers: authHeaders(),
    timeout: TIMEOUTS.health,
  })
}

export function query(documentId: string, question: string, topK: number, byoOpenAiKey?: string) {
  const extra: Record<string, string> = { ...jsonHeaders }
  if (byoOpenAiKey) extra[BYO_OPENAI_KEY_HEADER] = byoOpenAiKey

  return request<QueryResponse>(ROUTES.query, {
    method: 'POST',
    headers: authHeaders(extra),
    body: JSON.stringify({ document_id: documentId, question, top_k: topK }),
    timeout: TIMEOUTS.query,
  })
}

// ---------------------------------------------------------------------------
// Collections (admin-ish, unchanged)
// ---------------------------------------------------------------------------

export function listCollections() {
  return request<CollectionInfo[]>(ROUTES.collections, { timeout: TIMEOUTS.health })
}

export function deleteCollection(name: string) {
  return request<DeleteCollectionResponse>(ROUTES.collection(name), {
    method: 'DELETE',
    headers: authHeaders(),
    timeout: TIMEOUTS.health,
  })
}
