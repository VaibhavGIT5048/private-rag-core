// Builds the downloadable README.txt entirely in the browser — no network
// request. Commands here are the real ones from the project's docker-compose
// stack; keep them in sync with the repo's own README.

import { API_BASE_URL, REPO_URL } from '@/config'

const REPO_DIR = 'Semantic-Question-Answering-over-Large-Documents-using-RAG-LLaMA-3-Ollama'

const rule = '-'.repeat(68)

export function buildReadme(): string {
  return [
    'GROUNDED-RAG — Semantic Question Answering over Large Documents',
    'Ask questions of a PDF and get answers grounded in the document, with citations.',
    '',
    rule,
    'PREREQUISITES',
    rule,
    '  1. Docker Desktop installed AND running',
    '  2. An OpenAI API key',
    '',
    rule,
    'SETUP',
    rule,
    `  git clone ${REPO_URL}.git`,
    `  cd ${REPO_DIR}`,
    '  cp .env.example .env',
    '  docker compose up -d',
    '',
    'Open .env and set your key before starting the stack:',
    '',
    '  OPENAI_API_KEY=sk-your-key-here',
    '',
    rule,
    'VERIFY',
    rule,
    '  curl http://localhost:8000/health',
    '',
    'Expect: {"status":"ok","service":"rag-api","qdrant":"up","openai":"up",...}',
    'First boot pulls images and warms the reranker — allow a few minutes.',
    '',
    rule,
    'URLS',
    rule,
    '  http://localhost:8000   API (interactive docs at /docs)',
    '  http://localhost:6333   Qdrant dashboard',
    '  http://localhost:8501   legacy Streamlit UI (if the ui service is running)',
    '',
    rule,
    'USEFUL COMMANDS',
    rule,
    '  docker compose ps           # service status',
    '  docker compose logs -f api  # follow API logs',
    '  docker compose down         # stop everything (data is preserved)',
    '',
    rule,
    'BROWSER NOTE',
    rule,
    'The hosted frontend is served over HTTPS and talks to http://localhost:8000.',
    'Chrome, Edge and modern Firefox permit this. Safari blocks it — use Chrome,',
    'or run the frontend locally with `npm run dev`.',
    '',
    rule,
    'POINTING AT A DIFFERENT BACKEND',
    rule,
    'Set NEXT_PUBLIC_API_BASE_URL at build time to target a tunnelled or hosted',
    'API instead of localhost. A publicly reachable HTTPS backend makes the hosted',
    'site work in every browser with no local Docker at all.',
    '',
    `Currently configured backend: ${API_BASE_URL}`,
    '',
    `Repository: ${REPO_URL}`,
    '',
  ].join('\n')
}

export function downloadReadme() {
  const blob = new Blob([buildReadme()], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'README.txt'
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 4000)
}
