// Mirrors the FastAPI response models exactly. Fields the backend declares as
// nullable are nullable here too — the UI must guard them.

export type SubsystemState = 'up' | 'down'

export interface HealthStatus {
  status: 'ok' | 'degraded'
  service: string
  qdrant: SubsystemState
  openai: SubsystemState
  collection_name: string | null
}

export interface IngestResponse {
  request_id: string
  document_id: string
  collection_name: string
  filename: string
  file_type: string
  pdfs: number | null
  pages: number
  chunks: number
  passed_chunks: number
  penalised_chunks: number
  indexed_chunks: number
  /** Which parser tier produced the text — 'local-pdf', 'azure-document-intelligence', etc. */
  parser_used: string | null
}

export interface SourceChunk {
  chunk_id: number | string | null
  source: string
  page: number | string | null
  /** Reciprocal Rank Fusion score — NOT a 0–1 similarity. Values are small (~0.04). */
  score: number | null
  content: string
}

export interface QueryResponse {
  request_id: string
  answer: string
  sources: SourceChunk[]
  model: string
  collection_name: string
}

export interface IngestParams {
  file: File
  chunkSize: number
  chunkOverlap: number
  qualityThreshold: number
}

/** Connectivity state derived from polling /health. */
export type HealthState = 'checking' | 'ok' | 'degraded' | 'offline'

/** Drives the reactive canvas background and other ambient surfaces. */
export type Activity = 'idle' | 'querying' | 'ingesting' | 'connected' | 'offline'

// ---------------------------------------------------------------------------
// Auth + documents (Phase 1 backend, surfaced in Phase 4)
// ---------------------------------------------------------------------------

export interface AuthTokenResponse {
  access_token: string
  user_id: string
  email: string
  privacy_policy_accepted: boolean
}

export interface User {
  id: string
  email: string
  privacyPolicyAccepted: boolean
}

export interface DocumentSummary {
  id: string
  filename: string
  ingested_at: string
  pages: number | null
  chunks: number | null
  indexed_chunks: number | null
}

export interface ChatTurnSummary {
  id: string
  question: string
  answer: string
  sources: SourceChunk[]
  created_at: string
}

export interface DeleteDocumentResponse {
  deleted: string
  request_id?: string
}

/** Async ingest: POST /ingest/async returns this immediately, then poll. */
export interface IngestJobStatus {
  job_id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  filename: string
  result: IngestResponse | null
  error: string | null
  created_at: number
  updated_at: number
}
