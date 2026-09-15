// Single source of truth for anything environment- or endpoint-shaped.
// Pointing the app at a different backend must only require changing
// NEXT_PUBLIC_API_BASE_URL — nothing here is duplicated elsewhere.

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, '') || 'http://localhost:8000'

export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/+$/, '') ||
  'https://vaibhavgit5048.github.io/private-rag-core'

export const REPO_URL =
  'https://github.com/VaibhavGIT5048/private-rag-core'

// OAuth client IDs are public by design — they identify the app to the
// provider and appear in the redirect URL anyway. The *secrets* live only on
// the backend, which is what performs the code-for-token exchange.
export const GITHUB_CLIENT_ID = process.env.NEXT_PUBLIC_GITHUB_CLIENT_ID || ''
export const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || ''

// Must match exactly what is registered with GitHub and Google — a trailing
// slash difference is enough to fail with redirect_uri_mismatch. Derived from
// SITE_URL so dev and prod each get their own without extra configuration.
export const AUTH_CALLBACK_URL = `${SITE_URL}/auth/callback/`

// Header for an optional user-supplied OpenAI key. Stored client-side only and
// sent per request — never persisted server-side. Swaps the CHAT model only;
// embeddings always stay on the backend's default, since a document's vectors
// are permanently tied to whichever model indexed them.
export const BYO_OPENAI_KEY_HEADER = 'X-OpenAI-Api-Key'

export const ROUTES = {
  health: '/health',
  ready: '/ready',
  warmup: '/warmup',
  ingest: '/ingest',
  ingestAsync: '/ingest/async',
  ingestJob: (id: string) => `/ingest/jobs/${encodeURIComponent(id)}`,
  query: '/query',
  collections: '/collections',
  collection: (name: string) => `/collections/${encodeURIComponent(name)}`,
  documents: '/documents',
  document: (id: string) => `/documents/${encodeURIComponent(id)}`,
  documentHistory: (id: string) => `/documents/${encodeURIComponent(id)}/history`,
  authGithub: '/auth/github/callback',
  authGoogle: '/auth/google/callback',
  authSignup: '/auth/signup',
  authVerifyOtp: '/auth/verify-otp',
  authResendOtp: '/auth/resend-otp',
  authLogin: '/auth/login',
  authAcceptPrivacyPolicy: '/auth/accept-privacy-policy',
} as const

// /health is cheap (server-side cached) but crosses the network; ingest runs one
// embedding call per chunk and legitimately takes minutes on a real PDF.
export const TIMEOUTS = {
  health: 8_000,
  // The backend scales to zero when idle, and Container Apps holds the first
  // request open while it activates a replica — which takes far longer than the
  // 8s a warm probe needs. Aborting at 8s is what made a normal wake render as
  // "Backend unavailable", so the probe that might be waking it gets its own,
  // much longer budget.
  healthCold: 60_000,
  // Every auth call (sign-in, sign-up, OTP, OAuth callback, privacy-policy
  // acceptance) can just as easily land on a cold container as /health can —
  // measured cold start on staging is ~59s. 30s aborted these mid-wake, which
  // is exactly what made the consent modal's "Agree and continue" hang and
  // then report a timeout right as the backend was about to answer.
  auth: 75_000,
  query: 120_000,
  ingest: 600_000,
} as const

export const POLL_MS = {
  other: 20_000,
  // While offline, check back quickly: the backend usually returns within a
  // minute of being woken, and waiting a full 20s to notice adds delay that
  // isn't the cold start's fault.
  offline: 3_000,
} as const

export const INGEST_DEFAULTS = {
  chunkSize: 1000,
  chunkOverlap: 150,
  qualityThreshold: 4.0,
  topK: 4,
} as const

// Mirrors the backend parser router's supported extensions.
export const ACCEPTED_EXTENSIONS = [
  '.pdf', '.txt', '.md', '.csv', '.json',
  '.docx', '.pptx', '.xlsx',
  '.png', '.jpg', '.jpeg', '.webp',
] as const

export const STORAGE_KEYS = {
  theme: 'rag.theme',
  pipelineOpen: 'rag.pipelineOpen',
  hasConnected: 'rag.hasConnected',
  byoOpenAiKey: 'rag.byoOpenAiKey',
  jwt: 'rag.jwt',
  user: 'rag.user',
  // Which provider started an OAuth round-trip, checked against the `state`
  // returned by the provider to defend against CSRF.
  oauthState: 'rag.oauthState',
} as const
