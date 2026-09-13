'use client'

// The explainer. Must stand entirely on its own with no backend running — this
// is what most visitors will see and judge the project by.

import Link from 'next/link'
import { ArrowRight, Github } from 'lucide-react'

import { REPO_URL } from '@/config'
import { useUiPrefs } from '@/hooks/useUiPrefs'
import { PipelineVisualiser, useLoopingStage } from '@/components/PipelineVisualiser'
import { Button, Eyebrow, Rule } from '@/components/ui'

const HERO_STATS = [
  { v: '2', k: 'retrieval indexes: dense + sparse' },
  { v: '0.65 / 0.35', k: 'RRF fusion weighting' },
  { v: '±1', k: 'neighbour chunks expanded' },
  { v: '100%', k: 'answers carry citations' },
]

const ARCHITECTURE = [
  {
    role: 'API',
    name: 'FastAPI',
    note: 'Ingest, query, health and collection endpoints. Every response carries an X-Request-ID.',
  },
  {
    role: 'Vectors',
    name: 'Qdrant',
    note: 'Qdrant Cloud stores document-scoped vectors with owner and document filters.',
  },
  {
    role: 'Keywords',
    name: 'BM25',
    note: 'Sparse index running alongside the vectors, catching exact terms embeddings miss.',
  },
  {
    role: 'Reranker',
    name: 'Flashrank',
    note: 'Re-scores the fused candidate set before anything reaches the model.',
  },
  {
    role: 'Models',
    name: 'Azure + BYO',
    note: 'Azure-hosted gpt-5-mini by default; an optional OpenAI key is used only for your request.',
  },
  {
    role: 'Runtime',
    name: 'Azure Container Apps',
    note: 'Scale-to-zero hosted runtime; Docker Compose remains available for local development.',
  },
]

const FEATURES = [
  { t: 'Structure-aware chunking', d: 'Splits around headings and sections, falling back to recursive chunks only when a section is oversized.' },
  {
    t: 'Chunk quality gate',
    d: 'Scores chunks and applies a soft retrieval penalty; every non-empty chunk remains indexed.',
  },
  { t: 'Hybrid retrieval', d: 'Dense vectors and sparse keywords searched in parallel.' },
  {
    t: 'RRF fusion',
    d: 'Reciprocal Rank Fusion merges both rankings (0.65 dense, 0.35 sparse).',
  },
  { t: 'Reranking', d: 'Flashrank re-orders candidates by true relevance to the question.' },
  { t: 'Neighbour expansion', d: 'Pulls adjacent chunks so answers keep their context.' },
  {
    t: 'Inline citations',
    d: 'Every claim tagged with its file and page, rendered as linked chips.',
  },
  { t: 'Parser router', d: 'Routes PDF, Office, image and CSV files through local and OCR adapters with fallback tiers.' },
  { t: 'JWT + guardrails', d: 'Every document route is account-scoped and retrieved text is treated as untrusted data.' },
  { t: 'RAGAS harness', d: 'Faithfulness and relevancy measured offline, not guessed at.' },
]

const FAQS = [
  {
    q: 'Is my data private?',
    a: 'Yes. Every document, embedding, and chat history is scoped to your account — retrieval is filtered by owner on every query, so another user can never see or search your documents.',
  },
  {
    q: 'Do you use my documents to train any model?',
    a: 'No. Documents are used only to answer your own questions about them. Nothing you upload is used for training.',
  },
  {
    q: 'What happens when I delete a document?',
    a: "Deleting a document removes its database record, its vectors from storage, and its retrieval index — it isn't just hidden from the list.",
  },
  {
    q: 'Which file types are supported?',
    a: 'PDF, TXT, Markdown, CSV, JSON, Word, PowerPoint, Excel, and common image formats (PNG, JPG, WEBP), routed through local parsing or an OCR fallback depending on the file.',
  },
  {
    q: 'Can I use my own OpenAI key?',
    a: 'Yes, from the workbench. It swaps the model that generates answers; retrieval always uses the same embedding model a document was indexed with. Your key is sent with the request and never stored on the backend.',
  },
]

const SECURITY = [
  ['Account-scoped access', 'JWT authentication is required for ingest, query, documents, and chat history.'],
  ['Document isolation', 'Every vector search is filtered by the signed-in owner and selected document.'],
  ['Untrusted-data boundary', 'Document text is context, never instructions; guardrails keep retrieved content from steering the system.'],
  ['Abuse controls', 'Per-user rate limits protect the expensive ingest and query paths.'],
  ['Key handling', 'The optional BYO OpenAI key is sent in a request header and is never persisted by the backend.'],
]

export function LandingView() {
  const { motionOff } = useUiPrefs()
  const { ref: pipeRef, stage } = useLoopingStage(!motionOff)

  return (
    <main className="relative z-10">
      <section className="mx-auto max-w-[1180px] px-[26px] pb-8 pt-[90px]">
        <Eyebrow className="mb-[26px]">Private, grounded document answers</Eyebrow>
        <h1 className="m-0 mb-[26px] max-w-[15ch] text-[clamp(44px,7vw,92px)] font-extrabold leading-[0.95] tracking-[-0.035em]">
          Ask questions of any PDF.
        </h1>
        <p className="m-0 mb-[38px] max-w-[56ch] text-[clamp(18px,2vw,23px)] leading-[1.45] opacity-70">
          Answers grounded in the document itself, with every claim cited back to the page it came
          from. Structure-aware chunking, hybrid retrieval, Flashrank reranking, and a quality gate
          keep the answer tied to the source.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <Link href="/signin">
            <Button variant="cta" breathe>
              Try it yourself <ArrowRight size={16} aria-hidden />
            </Button>
          </Link>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            <Button variant="secondary">
              <Github size={16} aria-hidden /> View the repository
            </Button>
          </a>
        </div>

        <Rule className="mt-16" />
        <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))]">
          {HERO_STATS.map((s) => (
            <div
              key={s.k}
              className="py-4 pr-[26px]"
              style={{ borderRight: 'var(--brd-w) solid var(--brd)' }}
            >
              <div className="tnum text-[32px] font-extrabold tracking-[-0.03em]">{s.v}</div>
              <div className="mt-1 text-[12px] opacity-60">{s.k}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-20">
        <Rule className="mb-12" />
        <div className="grid grid-cols-[repeat(auto-fit,minmax(320px,1fr))] gap-12">
          <div>
            <Eyebrow className="mb-[14px]">The problem</Eyebrow>
            <h3 className="mb-[14px] text-[27px] font-extrabold tracking-[-0.02em]">
              A plain LLM will confidently invent what your document says.
            </h3>
            <p className="m-0 leading-[1.6] opacity-70">
              It has never read your file. Ask it about a 200-page report and it produces something
              plausible, unattributable, and occasionally wrong in the exact places that matter.
            </p>
          </div>
          <div>
            <Eyebrow className="mb-[14px]">The approach</Eyebrow>
            <h3 className="mb-[14px] text-[27px] font-extrabold tracking-[-0.02em]">
              Retrieve first. Answer only from what was retrieved. Cite it.
            </h3>
            <p className="m-0 leading-[1.6] opacity-70">
              The document is split around its own structure, scored for quality, and indexed twice:
              dense vectors and sparse keywords. Every question is rewritten against recent chat,
              fused, reranked, and handed only the surviving chunks.
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-[90px]">
        <Rule className="mb-10" />
        <div className="mb-2 flex flex-wrap items-baseline gap-4">
          <h2 className="m-0 text-[38px] font-extrabold tracking-[-0.03em]">How it works</h2>
          <span
            className="text-[12px] opacity-55"
            title="Illustrative view of the backend pipeline"
          >
            Illustrative view of the backend pipeline, not measured telemetry.
          </span>
          <span className="text-[12px] font-extrabold opacity-65">Hover a stage for a plain-language explanation.</span>
        </div>
        <div ref={pipeRef} className="mt-7">
          <PipelineVisualiser stage={stage} />
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-[90px]">
        <Rule className="mb-10" />
        <div className="mb-7 flex flex-wrap items-baseline gap-4">
          <h2 className="m-0 text-[38px] font-extrabold tracking-[-0.03em]">Security by default</h2>
          <span className="text-[12px] opacity-55">The boundaries around the pipeline matter as much as the models.</span>
        </div>
        <div
          className="grid grid-cols-[repeat(auto-fit,minmax(230px,1fr))]"
          style={{ gap: 'var(--brd-w)', background: 'var(--brd)' }}
        >
          {SECURITY.map(([title, detail]) => (
            <div key={title} className="p-6" style={{ background: 'var(--panel)', backdropFilter: 'var(--blur)' }}>
              <div className="mb-2 text-[16px] font-extrabold">{title}</div>
              <div className="text-[13px] leading-[1.55] opacity-65">{detail}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-[90px]">
        <Rule className="mb-10" />
        <h2 className="m-0 mb-7 text-[38px] font-extrabold tracking-[-0.03em]">Architecture</h2>
        <div
          className="grid grid-cols-[repeat(auto-fit,minmax(230px,1fr))]"
          style={{ gap: 'var(--brd-w)', background: 'var(--brd)' }}
        >
          {ARCHITECTURE.map((a) => (
            <div
              key={a.name}
              className="p-6"
              style={{ background: 'var(--panel)', backdropFilter: 'var(--blur)' }}
            >
              <div
                className="mb-[10px] text-[11px] font-extrabold uppercase tracking-[0.12em]"
                style={{ color: 'var(--accent)' }}
              >
                {a.role}
              </div>
              <div className="mb-[6px] text-[19px] font-extrabold tracking-[-0.02em]">{a.name}</div>
              <div className="text-[13px] leading-[1.5] opacity-65">{a.note}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-[90px]">
        <Rule className="mb-10" />
        <h2 className="m-0 mb-7 text-[38px] font-extrabold tracking-[-0.03em]">
          What&apos;s in the pipeline
        </h2>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-x-10 gap-y-[26px]">
          {FEATURES.map((f) => (
            <div key={f.t} style={{ borderTop: 'var(--brd-w) solid var(--brd)' }} className="pt-[14px]">
              <div className="mb-[5px] text-[16px] font-extrabold tracking-[-0.01em]">{f.t}</div>
              <div className="text-[13px] leading-[1.5] opacity-65">{f.d}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-[90px]">
        <div className="p-11" style={{ background: 'var(--accent)', color: 'var(--on-accent)' }}>
          <div className="mb-4 text-[11px] font-extrabold uppercase tracking-[0.14em] opacity-75">
            Where your data goes
          </div>
          <p className="m-0 mb-4 max-w-[44ch] text-[clamp(20px,2.4vw,30px)] font-extrabold leading-[1.28] tracking-[-0.02em]">
            Your account keeps its own documents and chat history; the default model runs through our Azure deployment.
          </p>
          <p className="m-0 max-w-[60ch] text-[15px] leading-[1.55] opacity-85">
            Retrieval uses Qdrant Cloud and self-hosted bge-m3 embeddings. Generation uses Azure-hosted
            gpt-5-mini by default. If you provide an OpenAI key, it is sent per request and never stored
            on the backend.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-[1180px] px-[26px] pb-[90px]">
        <Rule className="mb-10" />
        <h2 className="m-0 mb-7 text-[38px] font-extrabold tracking-[-0.03em]">Frequently asked</h2>
        <div className="grid max-w-[760px] gap-[2px]" style={{ background: 'var(--brd)' }}>
          {FAQS.map((item) => (
            <details key={item.q} className="group p-5" style={{ background: 'var(--panel)', backdropFilter: 'var(--blur)' }}>
              <summary className="cursor-pointer list-none text-[16px] font-extrabold tracking-[-0.01em] marker:content-none">
                {item.q}
              </summary>
              <p className="m-0 mt-3 text-[14px] leading-[1.6] opacity-70">{item.a}</p>
            </details>
          ))}
        </div>
      </section>

      <footer style={{ borderTop: 'var(--brd-w) solid var(--brd)' }}>
        <div className="mx-auto flex max-w-[1180px] flex-wrap items-center gap-6 px-[26px] py-7 text-[13px]">
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="font-extrabold">
            Repository →
          </a>
          <span className="opacity-55">MIT licence</span>
          <span className="opacity-55">Vaibhav · Semantic Q&amp;A over Large Documents</span>
          <Link href="/privacy" className="font-extrabold opacity-70 hover:opacity-100">
            Privacy Policy
          </Link>
          <div className="flex-1" />
          <Link href="/signin">
            <Button variant="ghost">Sign in →</Button>
          </Link>
        </div>
      </footer>
    </main>
  )
}
