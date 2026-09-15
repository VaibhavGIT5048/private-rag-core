'use client'

// The live workbench. Ingest, query, source inspection and collection
// management stay in one route so the user can move from upload to answers
// without losing context.

import { useCallback, useEffect, useState, useRef } from 'react'
import Link from 'next/link'
import { ChevronRight, Search } from 'lucide-react'
import { useRouter, useSearchParams } from 'next/navigation'
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'

import { API_BASE_URL, STORAGE_KEYS } from '@/config'
import { useActivity } from '@/hooks/useActivity'
import { useRequireAuth } from '@/hooks/useAuth'
import { useHealth } from '@/hooks/useHealth'
import { useToast } from '@/hooks/useToast'
import { useUiPrefs } from '@/hooks/useUiPrefs'
import {
  CITE_SCHEME,
  formatScore,
  maxScore,
  matchSourceIndices,
  parseCitationHref,
  scoreBarPercent,
  toMarkdownWithCitationLinks,
} from '@/lib/citations'
import { MAX_PIPE_STAGE, QUERY_STAGES } from '@/lib/pipeline'
import { ApiError, getChatHistory, query } from '@/services/api'
import type { ChatTurnSummary, IngestResponse, SourceChunk } from '@/types/api'
import { CollectionsPanel } from '@/components/CollectionsPanel'
import { IngestPanel } from '@/components/IngestPanel'
import { PipelineVisualiser } from '@/components/PipelineVisualiser'
import { Button, Eyebrow, Mono, Panel, PanelHeader, Spinner } from '@/components/ui'

interface Turn {
  id: string
  question: string
  topK: number
  pending: boolean
  answer?: string
  sources: SourceChunk[]
  model?: string
  requestId?: string
  error?: string
  errorTitle?: string
  /** Collapsed by default — several expanded source cards per turn drowns
   *  out the answer itself. Clicking a citation forces this open. */
  sourcesOpen?: boolean
}

interface CitationFocus {
  turnId: string
  source: string
  page: string
}

const DEFAULT_TOP_K = 4

function historyTurn(turn: ChatTurnSummary): Turn {
  return {
    id: turn.id,
    question: turn.question,
    topK: DEFAULT_TOP_K,
    pending: false,
    answer: turn.answer,
    sources: turn.sources ?? [],
  }
}

export function WorkbenchView() {
  const router = useRouter()
  const params = useSearchParams()
  const { ready } = useRequireAuth()
  const documentId = params.get('doc')
  const { health, isConnected, waking } = useHealth()
  const { setBusy } = useActivity()
  const { flash, announce } = useToast()
  const { motionOff } = useUiPrefs()

  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(DEFAULT_TOP_K)
  const [transcript, setTranscript] = useState<Turn[]>([])
  const [querying, setQuerying] = useState(false)
  const [thinkingIdx, setThinkingIdx] = useState(0)
  const [typed, setTyped] = useState(0)
  const [pipeStage, setPipeStage] = useState(-1)
  // Which turn the reveal animation is currently walking through, if any.
  // Deliberately explicit: inferring it from "is this the newest turn" made
  // every restored conversation render its most recent answer as an empty
  // string, because `typed` starts at 0 and no timer ever runs for history.
  const [typingTurnId, setTypingTurnId] = useState<string | null>(null)
  // Which turn the main answer area is showing. null means "the newest",
  // rather than pinning an id, so a new answer takes over automatically
  // without needing to be written here too.
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null)
  const [focus, setFocus] = useState<CitationFocus | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)
  const [byoKey, setByoKey] = useState('')
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.byoOpenAiKey)
      if (stored) setByoKey(stored)
    } catch {
      /* private browsing — falls back to session-only, unpersisted */
    }
  }, [])

  const updateByoKey = useCallback((value: string) => {
    setByoKey(value)
    try {
      if (value) localStorage.setItem(STORAGE_KEYS.byoOpenAiKey, value)
      else localStorage.removeItem(STORAGE_KEYS.byoOpenAiKey)
    } catch {
      /* ignore */
    }
  }, [])

  const stageTimer = useRef<ReturnType<typeof setInterval> | null>(null)
  const typeTimer = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearTimers = useCallback(() => {
    if (stageTimer.current) clearInterval(stageTimer.current)
    if (typeTimer.current) clearInterval(typeTimer.current)
    stageTimer.current = null
    typeTimer.current = null
  }, [])

  useEffect(() => () => clearTimers(), [clearTimers])

  // A workbench without a document is still useful for the first upload, but
  // a resumed workbench must load the saved conversation before accepting a
  // new question.
  useEffect(() => {
    if (!ready || !documentId) {
      if (ready && !documentId) setTranscript([])
      return
    }
    const controller = new AbortController()
    setHistoryLoading(true)
    setHistoryError(null)
    void getChatHistory(documentId, controller.signal)
      .then((turns) => {
        setTranscript(turns.map(historyTurn))
      })
      .catch((err) => {
        if (controller.signal.aborted) return
        setHistoryError(err instanceof ApiError ? err.detail : 'Could not load chat history.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setHistoryLoading(false)
      })
    return () => controller.abort()
  }, [documentId, ready])

  const startTyping = useCallback(
    (turnId: string, answer: string) => {
      if (motionOff) {
        setTypingTurnId(null)
        setTyped(answer.length)
        return
      }

      if (typeTimer.current) clearInterval(typeTimer.current)
      setTyped(0)
      setTypingTurnId(turnId)
      const step = Math.max(2, Math.round(answer.length / 90))
      typeTimer.current = setInterval(() => {
        setTyped((current) => {
          const next = current + step
          if (next >= answer.length) {
            if (typeTimer.current) clearInterval(typeTimer.current)
            typeTimer.current = null
            // Released here rather than left to the render: once the reveal is
            // complete the turn is just another finished answer.
            setTypingTurnId(null)
            return answer.length
          }
          return next
        })
      }, 26)
    },
    [motionOff],
  )

  const patchTurn = useCallback((id: string, patch: Partial<Turn>) => {
    setTranscript((current) => current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)))
  }, [])

  // Focusing a citation highlights one source card, so the dropdown holding
  // that card has to open too — otherwise the highlight lands inside a
  // collapsed panel and the click appears to do nothing.
  const focusCitation = useCallback(
    (turnId: string, source: string, page: string) => {
      const alreadyFocused =
        focus?.turnId === turnId && focus.source === source && focus.page === page
      setFocus(alreadyFocused ? null : { turnId, source, page })
      if (!alreadyFocused) patchTurn(turnId, { sourcesOpen: true })
    },
    [focus, patchTurn],
  )

  const submitQuery = useCallback(async () => {
    const trimmed = question.trim()
    if (!trimmed || querying || !isConnected) return

    const id = `q-${Date.now()}`
    setTranscript((current) => current.concat([{ id, question: trimmed, topK, pending: true, sources: [] }]))
    setQuestion('')
    setQuerying(true)
    setThinkingIdx(0)
    setTyped(0)
    // Resetting `typed` without releasing the previous turn would blank an
    // answer that is still mid-reveal until the new one arrives.
    setTypingTurnId(null)
    // Snap back to the newest turn, so an answer asked for while reading an
    // older one still appears rather than arriving off-screen in the rail.
    setSelectedTurnId(null)
    setPipeStage(0)
    setFocus(null)
    setBusy('querying')
    clearTimers()

    stageTimer.current = setInterval(() => {
      setThinkingIdx((current) => (current + 1) % QUERY_STAGES.length)
      setPipeStage((current) => Math.min(current + 1, MAX_PIPE_STAGE))
    }, 620)

    try {
      if (!documentId) {
        setTranscript((current) => current.filter((turn) => turn.id !== id))
        flash('Ingest a document first, then ask a question.')
        return
      }
      const result = await query(documentId, trimmed, topK, byoKey || undefined)
      const sources = result.sources ?? []
      patchTurn(id, {
        pending: false,
        answer: result.answer,
        sources,
        model: result.model,
        requestId: result.request_id,
      })
      setPipeStage(MAX_PIPE_STAGE)
      announce(`Answer ready with ${sources.length} sources.`)
      startTyping(id, result.answer)
    } catch (err) {
      const apiErr = err instanceof ApiError ? err : null
      const notIngested = apiErr?.isNotIngested ?? false
      const detail =
        notIngested
          ? 'Nothing is indexed yet, so there is nothing to ground an answer in. Ingest a document and ask again.'
          : apiErr?.detail ?? (err instanceof Error ? err.message : 'Unknown error')
      patchTurn(id, {
        pending: false,
        error: detail,
        errorTitle: notIngested
          ? 'No document indexed yet'
          : `Request failed${apiErr?.status ? ` — HTTP ${apiErr.status}` : ''}`,
        requestId: apiErr?.requestId ?? 'none returned',
      })
      flash(notIngested ? 'Ingest a document first' : `Query failed — ${detail}`)
    } finally {
      if (stageTimer.current) clearInterval(stageTimer.current)
      stageTimer.current = null
      setQuerying(false)
      setBusy(null)
    }
  }, [announce, byoKey, clearTimers, documentId, flash, isConnected, patchTurn, question, querying, setBusy, startTyping, topK])

  const connected = isConnected
  const hasExistingIndex = Boolean(documentId)
  const emptyTranscript = transcript.length === 0
  // Only the newest turn gets the full main-column treatment; everything
  // before it is history and lives in the side rail instead. transcript is
  // chronological (oldest first) whether it came from getChatHistory or from
  // appending a fresh question, so the last element is always "newest".
  const latestTurn = transcript.at(-1) ?? null
  // The rail SELECTS which turn the main area shows; it does not expand turns
  // in place. Expanding inline meant rendering a full answer into a ~260px
  // column — text wrapping every three words — and, worse, it made the rail
  // tall enough to stretch the whole grid row, leaving a large void beside the
  // short panels next to it. This is the pattern a conversation sidebar
  // normally uses: the list navigates, the main pane displays.
  const selectedTurn = transcript.find((turn) => turn.id === selectedTurnId) ?? latestTurn
  const viewingOlder = Boolean(selectedTurn && latestTurn && selectedTurn.id !== latestTurn.id)
  // Drives both the column template and the bottom row's span. They have to
  // agree: a span wider than the template silently creates an implicit extra
  // column, which is a different broken layout rather than an error.
  const hasRail = transcript.length > 1

  const renderAnswer = (turn: Turn) => {
    const full = turn.answer ?? ''
    const stillTyping = turn.id === typingTurnId && Boolean(full) && typed < full.length
    const visibleAnswer = stillTyping ? full.slice(0, typed) : full

    return (
      <div className="space-y-3">
        {/* Restored turns carry no model or request id — those exist only on a
            live response — so the line is dropped rather than filled with
            "unknown model · request —". */}
        {turn.model && (
          <div className="flex flex-wrap items-center gap-2 text-[12px] opacity-55">
            <span className="font-extrabold uppercase tracking-[0.12em]">{turn.model}</span>
            {turn.requestId && (
              <>
                <span>·</span>
                <Mono className="text-[11.5px]">request {turn.requestId.slice(0, 8)}</Mono>
              </>
            )}
          </div>
        )}

        {/* Answers are Markdown. Citations are rewritten to links on a `cite:`
            scheme first, so one parse covers the whole answer — splitting the
            string around citations breaks any list item or bold run they sit
            inside. The `a` override swaps those links back for chips. */}
        {/* Capped at a readable measure. Now that the answer panel spans the
            full page width, uncapped prose would run to ~200 characters a
            line, which is where reading accuracy falls apart — the spare
            width goes to the sources column instead. */}
        <div
          className="answer-prose max-w-[76ch] text-[15px] leading-[1.7]"
          data-typing={stillTyping ? 'true' : 'false'}
        >
          <ReactMarkdown
            // react-markdown strips any protocol outside its safe list, which
            // would blank out every `cite:` href and lose the citations. Pass
            // those through untouched; everything else keeps the default
            // sanitising so a link in a document can't smuggle in javascript:.
            urlTransform={(url) => (url.startsWith(CITE_SCHEME) ? url : defaultUrlTransform(url))}
            components={{
              // `node` is react-markdown's AST handle — destructured out so it
              // never reaches the DOM, which would warn on an unknown attribute.
              a: ({ href, children, node, ...props }) => {
                const cite = href ? parseCitationHref(href) : null
                if (!cite) {
                  return (
                    <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                      {children}
                    </a>
                  )
                }

                const active =
                  focus?.turnId === turn.id && focus.source === cite.source && focus.page === cite.page
                return (
                  <button
                    type="button"
                    data-citation=""
                    aria-pressed={active}
                    onClick={() => focusCitation(turn.id, cite.source, cite.page)}
                    className="mx-[3px] inline-flex items-center rounded-[var(--r-sm)] border px-2 py-[2px] text-[12.5px] font-extrabold transition-transform"
                    style={{
                      verticalAlign: '1px',
                      borderColor: 'var(--accent)',
                      background: active
                        ? 'var(--accent)'
                        : 'color-mix(in srgb, var(--accent) 12%, transparent)',
                      color: active ? 'var(--on-accent)' : 'var(--accent-hi)',
                    }}
                  >
                    {cite.source} · p{cite.page}
                  </button>
                )
              },
            }}
          >
            {toMarkdownWithCitationLinks(visibleAnswer)}
          </ReactMarkdown>
        </div>
      </div>
    )
  }

  // Full content for one turn — question, answer, sources, request id. Used
  // both for the newest turn (always shown, in the main column) and for a
  // history-rail row once it's expanded; identical either way, so citation
  // focus and the sources dropdown behave the same regardless of where a turn
  // happens to be rendered.
  const renderTurnBody = (turn: Turn) => {
    const sourceMax = maxScore(turn.sources)
    const focusedIndices =
      focus?.turnId === turn.id ? matchSourceIndices(turn.sources, focus.source, focus.page) : []
    // Sources sit beside the answer rather than under it, so starting them
    // open costs no vertical space and fills a column that would otherwise be
    // blank. `??` keeps an explicit toggle winning over that default.
    const sourcesOpen = turn.sourcesOpen ?? true

    return (
      <div className="grid gap-4 p-5">
        <div className="flex flex-wrap items-baseline gap-3">
          <div className="text-[12px] font-extrabold uppercase tracking-[0.1em] opacity-55">Question</div>
          <div className="text-[16px] font-extrabold tracking-[-0.01em]">{turn.question}</div>
        </div>

        {turn.pending && (
          <div className="flex items-center gap-3 text-[13px] opacity-70">
            <Spinner size={16} />
            <span>{QUERY_STAGES[thinkingIdx]}</span>
          </div>
        )}

        {turn.error && (
          <div role="alert" className="grid gap-2 p-[14px] text-[13px] leading-[1.55]" style={{ border: 'var(--brd-w) solid var(--accent)' }}>
            <div className="font-extrabold">{turn.errorTitle ?? 'Request failed'}</div>
            <div className="opacity-75">{turn.error}</div>
            <Mono className="text-[11.5px] opacity-50">X-Request-ID {turn.requestId ?? 'none returned'}</Mono>
          </div>
        )}

        {turn.answer && !turn.error && (
          // Prose and sources side by side. The prose column is capped at a
          // readable measure (see renderAnswer), so the leftover width goes to
          // the sources rather than stretching lines to 200 characters.
          //
          // `auto` rather than a fixed fr: with the cap in play a fixed ratio
          // would leave dead space between the two columns whenever it bit.
          // Sizing the prose track to its content and giving the remainder to
          // the sources means the full width is used at any answer length.
          <div className="grid items-start gap-6 xl:grid-cols-[auto_minmax(0,1fr)]">
            <div className="min-w-0">{renderAnswer(turn)}</div>

            <div className="min-w-0">
              {turn.sources.length === 0 ? (
                <>
                  <div className="mb-3 text-[11px] font-extrabold uppercase tracking-[0.12em] opacity-55">
                    Sources
                  </div>
                  <div className="text-[13px] opacity-60">The backend returned no sources for this answer.</div>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => patchTurn(turn.id, { sourcesOpen: !sourcesOpen })}
                    aria-expanded={sourcesOpen}
                    aria-controls={`sources-${turn.id}`}
                    className="mb-3 flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.12em] opacity-55 hover:opacity-90"
                  >
                    <ChevronRight
                      size={13}
                      aria-hidden
                      style={{
                        transform: sourcesOpen ? 'rotate(90deg)' : 'none',
                        transition: 'transform .18s ease',
                      }}
                    />
                    Sources ({turn.sources.length})
                  </button>

                  {/* Toggled by class, not the `hidden` attribute: Tailwind's
                      `.grid` utility sits in a later layer than preflight's
                      `[hidden]` rule at equal specificity, so it would win and
                      the collapsed panel would stay on screen.
                      Capped height once side by side, so a ten-source answer
                      scrolls its own column instead of setting the height of
                      the whole panel. */}
                  <div
                    id={`sources-${turn.id}`}
                    className={
                      sourcesOpen ? 'grid gap-3 xl:max-h-[68vh] xl:overflow-y-auto xl:pr-1' : 'hidden'
                    }
                  >
                    {turn.sources.map((source, index) => {
                      const focused = focusedIndices.includes(index)
                      return (
                        <button
                          key={`${turn.id}-${index}`}
                          type="button"
                          onClick={() =>
                            setFocus((current) =>
                              current?.turnId === turn.id && current.source === source.source && current.page === String(source.page ?? '—')
                                ? null
                                : { turnId: turn.id, source: source.source, page: String(source.page ?? '—') },
                            )
                          }
                          className="grid gap-3 text-left transition-transform"
                          style={{
                            padding: '16px',
                            background: focused ? 'color-mix(in srgb, var(--accent) 10%, var(--panel-solid))' : 'var(--panel-solid)',
                            border: `var(--brd-w) solid ${focused ? 'var(--accent)' : 'var(--brd)'}`,
                            transform: focused ? 'translateY(-1px)' : 'none',
                          }}
                        >
                          <div className="flex flex-wrap items-baseline justify-between gap-3">
                            <div className="text-[14px] font-extrabold">{source.source}</div>
                            <div className="flex flex-wrap gap-3 text-[11px] font-extrabold uppercase tracking-[0.08em] opacity-55">
                              <span>{source.page ?? 'page —'}</span>
                              <span>chunk {source.chunk_id ?? '—'}</span>
                              <span>score {formatScore(source.score)}</span>
                            </div>
                          </div>

                          <div className="h-[6px] overflow-hidden" style={{ background: 'var(--brd)' }}>
                            <span
                              style={{
                                width: scoreBarPercent(source.score, sourceMax),
                                background: 'var(--accent)',
                                transition: 'width .9s cubic-bezier(.2,.8,.2,1)',
                              }}
                              className="block h-full"
                            />
                          </div>

                          <div className="text-[13px] leading-[1.6] opacity-78">
                            {source.content || 'No excerpt returned.'}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {!turn.pending && !turn.error && !turn.answer && (
          <div className="text-[13px] opacity-60">No answer returned.</div>
        )}

        {turn.requestId && (
          <div className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-45">
            request {turn.requestId.slice(0, 8)}
          </div>
        )}
      </div>
    )
  }

  const onIngested = (stats: IngestResponse) => {
    router.replace(`/workbench?doc=${encodeURIComponent(stats.document_id)}`)
    setRefreshToken((token) => token + 1)
  }

  if (!ready) {
    return (
      <main className="mx-auto w-full max-w-[1100px] px-6 py-24">
        <Spinner size={18} />
      </main>
    )
  }

  return (
    <main className="relative z-10 mx-auto max-w-[1320px] px-[26px] pb-20 pt-8">
      {/* Deliberately restrained for a page people return to and work in: a
          58px three-line hero plus a paragraph pushed the query box and the
          answer below the fold on every visit, to say something already read
          on the first one. */}
      <section className="mb-6 flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-[62ch]">
          <Eyebrow className="mb-2">Workbench</Eyebrow>
          <h1 className="m-0 text-[clamp(24px,2.4vw,32px)] font-extrabold leading-[1.05] tracking-[-0.03em]">
            Ingest a PDF, ask a question, inspect the sources.
          </h1>
          <p className="m-0 mt-2 text-[14px] leading-[1.5] opacity-65">
            Every answer is grounded, every citation is clickable, and every collection delete is
            typed-confirmed.
          </p>
        </div>

        {!connected && (
          <Panel className="max-w-[440px]">
            <div className="p-5">
              {/* A cold start is the ordinary case here, not a fault: the
                  backend scales to zero when idle to keep it free to run.
                  Calling that "unavailable" reads as broken, so it only says so
                  once the wake has taken longer than one plausibly can. */}
              <div className="mb-2 flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.12em] opacity-55">
                {waking && <Spinner size={12} />}
                {waking ? 'Waking the backend' : 'Backend unavailable'}
              </div>
              <div className="text-[14px] leading-[1.55] opacity-75">
                {waking ? (
                  <>
                    It scales to zero when idle, so the first visit after a quiet
                    spell takes about half a minute to start. This page connects
                    itself the moment it is ready — nothing to do.
                  </>
                ) : (
                  <>
                    Still no response from <span className="font-extrabold">{API_BASE_URL}</span>.
                    The workbench activates automatically once /health reports ok.
                  </>
                )}
              </div>
              {!waking && (
                <div className="mt-4 flex flex-wrap gap-3">
                  <Link href="/home">
                    <Button variant="ghost">Back to home</Button>
                  </Link>
                  <Button variant="chip" onClick={() => flash('This reconnects automatically once the backend responds')}>
                    What now?
                  </Button>
                </div>
              )}
            </div>
          </Panel>
        )}
      </section>

      {/* Every track is minmax(0, …) rather than a px minimum. Fixed minimums
          summed wider than the container at the breakpoint that enabled them,
          so the grid overflowed and columns bled into each other; a 0 minimum
          can always shrink to fit. The third column only appears at xl, where
          there is genuinely room for it.
          items-start stops the columns stretching to match the tallest one. */}
      <div
        className={`grid items-start gap-7 ${
          hasRail
            ? 'lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.7fr)] xl:grid-cols-[minmax(0,0.75fr)_minmax(0,1.7fr)_minmax(0,0.85fr)]'
            : 'lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.7fr)]'
        }`}
      >
        {/* min-w-0 on the column itself: a grid item defaults to
            min-width:auto, so one long unbreakable string (a document UUID)
            would otherwise push the whole track wider than its share. */}
        <div className="grid min-w-0 content-start gap-7">
          <IngestPanel
            connected={connected}
            hasExistingIndex={hasExistingIndex}
            onIngested={onIngested}
          />

          {/* One slim row instead of a header + three stacked rows — the same
              three facts, without a card's worth of padding around each. */}
          <Panel>
            <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 px-4 py-3 text-[12.5px]">
              <span className="flex shrink-0 items-center gap-[6px] font-extrabold">
                <span
                  className="h-2 w-2 rounded-full"
                  aria-hidden
                  style={{ background: connected ? 'var(--accent)' : 'var(--brd)' }}
                />
                {connected ? 'Connected' : 'Offline'}
              </span>
              <span className="shrink-0 opacity-35">·</span>
              {/* The document id is a 36-char unbreakable UUID. It needs both a
                  0 minimum and its own overflow rule, or it sets the width of
                  everything upstream of it. */}
              <span
                className="min-w-0 flex-1 truncate font-mono text-[11.5px] opacity-70"
                title={documentId ?? undefined}
              >
                {documentId ?? 'No document selected'}
              </span>
              <span className="shrink-0 opacity-35">·</span>
              <span className="shrink-0 opacity-70">{querying ? QUERY_STAGES[thinkingIdx] : 'Ready'}</span>
            </div>
          </Panel>
        </div>

        <div className="grid min-w-0 content-start gap-7">
          <Panel>
            <PanelHeader
              title="Query"
              right={
                <div className="flex items-center gap-3">
                  <label htmlFor="top-k" className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-55">
                    top-k
                  </label>
                  <select
                    id="top-k"
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                    className="px-[7px] py-[5px] text-[12px] font-extrabold"
                    style={{
                      color: 'var(--ink)',
                      background: 'var(--chip-bg)',
                      border: 'var(--brd-w) solid var(--brd)',
                      borderRadius: 'var(--r-sm)',
                    }}
                  >
                    {[3, 4, 5, 6, 8, 10].map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </div>
              }
            />

            <div className="p-5">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <label htmlFor="byo-openai-key" className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-55">
                  Your OpenAI key (optional)
                </label>
                <input
                  id="byo-openai-key"
                  type="password"
                  autoComplete="off"
                  value={byoKey}
                  onChange={(e) => updateByoKey(e.target.value)}
                  placeholder="sk-… uses your own OpenAI billing for this answer"
                  className="min-w-[260px] flex-1 px-[10px] py-[6px] text-[12.5px]"
                  style={{
                    color: 'var(--ink)',
                    background: 'var(--chip-bg)',
                    border: 'var(--brd-w) solid var(--brd)',
                    borderRadius: 'var(--r-sm)',
                  }}
                />
              </div>
              <p className="mb-3 text-[11.5px] opacity-55">
                Stored only in this browser and sent with this request — never saved on the backend. Leave blank to
                use the default (Azure-hosted, billed to us).
              </p>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    e.preventDefault()
                    void submitQuery()
                  }
                }}
                placeholder="What is the carbon intensity reduction goal, and by when?"
                rows={3}
                className="w-full resize-y px-[14px] py-[14px] text-[15.5px] leading-[1.55]"
                style={{
                  fontFamily: 'var(--font-archivo)',
                  color: 'var(--ink)',
                  background: 'var(--chip-bg)',
                  border: 'var(--brd-w) solid var(--brd)',
                  borderRadius: 'var(--r-sm)',
                }}
              />

              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button variant="solid" onClick={() => void submitQuery()} disabled={!question.trim() || querying || !connected}>
                  {querying ? <Spinner size={14} /> : <Search size={14} aria-hidden />}
                  {querying ? 'Retrieving…' : 'Ask'}
                </Button>
                <div className="text-[12px] opacity-60">
                  {querying ? QUERY_STAGES[thinkingIdx] : connected ? 'Cmd/Ctrl + Enter to submit' : 'Connect the backend first'}
                </div>
              </div>
            </div>
          </Panel>
        </div>

        {hasRail && (
          // Between lg and xl there are only 2 explicit columns (see the
          // comment above), so without an explicit span this wraps into
          // column 1 alone and leaves a dead gap beside it in column 2 —
          // col-span-2 makes it take the full row there instead; xl:col-span-1
          // hands it back its own dedicated third column once one exists.
          <div className="grid min-w-0 content-start gap-4 lg:col-span-2 xl:col-span-1">
            <Panel>
              <PanelHeader
                title="History"
                right={
                  <span className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-50">
                    {transcript.length} turns
                  </span>
                }
              />
              {/* Bounded and scrolled internally. An unbounded rail grows with
                  the conversation, and because a grid row is as tall as its
                  tallest item, that stretched the entire row and left a void
                  beside the shorter panels. A fixed px cap rather than a vh
                  one, so browser zoom cannot turn it back into a full-page
                  column. */}
              <div
                className="max-h-[420px] overflow-y-auto"
                style={{ display: 'grid', gap: 'var(--brd-w)', background: 'var(--brd)' }}
              >
                {transcript
                  .slice()
                  .reverse()
                  .map((turn) => {
                    const active = turn.id === selectedTurn?.id
                    const isLatest = turn.id === latestTurn?.id
                    return (
                      <button
                        key={turn.id}
                        type="button"
                        onClick={() => setSelectedTurnId(isLatest ? null : turn.id)}
                        aria-current={active ? 'true' : undefined}
                        className="flex w-full min-w-0 items-start gap-2 px-4 py-3 text-left transition-colors"
                        style={{
                          background: active
                            ? 'color-mix(in srgb, var(--accent) 10%, var(--panel-solid))'
                            : 'var(--panel-solid)',
                          // A left rule rather than a full border, so switching
                          // selection does not shift the text by a pixel.
                          boxShadow: active ? 'inset 3px 0 0 0 var(--accent)' : 'none',
                        }}
                      >
                        <span className="min-w-0 flex-1">
                          <span
                            className="block text-[13px] font-extrabold leading-[1.4]"
                            style={{
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                              // A pasted question can be one very long word;
                              // without this it would widen the rail rather
                              // than wrap inside it.
                              overflowWrap: 'anywhere',
                            }}
                          >
                            {turn.question}
                          </span>
                          <span className="mt-[3px] block text-[10.5px] font-extrabold uppercase tracking-[0.08em] opacity-45">
                            {isLatest ? 'Latest' : `${turn.sources.length} sources`}
                          </span>
                        </span>
                      </button>
                    )
                  })}
              </div>
            </Panel>
          </div>
        )}

        {/* The answer gets the full page width on its own row, rather than
            being confined to the query column with the rest of the viewport
            left blank beside it. Its internal split (prose | sources) is what
            actually consumes that width — see renderTurnBody. */}
        <div className={`grid min-w-0 gap-7 lg:col-span-2 ${hasRail ? 'xl:col-span-3' : ''}`}>
          {historyError && (
            <Panel>
              <div role="alert" className="p-5 text-[13px]" style={{ border: 'var(--brd-w) solid var(--accent)' }}>
                {historyError}
              </div>
            </Panel>
          )}

          {historyLoading ? (
            <Panel>
              <div className="flex items-center gap-3 p-8 text-[13px] opacity-70"><Spinner size={16} /> Loading saved conversation…</div>
            </Panel>
          ) : emptyTranscript ? (
            <Panel>
              <div className="grid gap-3 p-11">
                <div className="flex gap-1">
                  <span className="h-2 w-2 rounded-full bg-[var(--accent)] opacity-80" />
                  <span className="h-2 w-2 rounded-full bg-[var(--accent)] opacity-50" />
                  <span className="h-2 w-2 rounded-full bg-[var(--accent)] opacity-30" />
                </div>
                <div className="text-[19px] font-extrabold tracking-[-0.02em]">
                  {hasExistingIndex ? 'Ask something specific.' : 'Ingest a document first.'}
                </div>
                <div className="max-w-[52ch] text-[13.5px] leading-[1.55] opacity-65">
                  {hasExistingIndex
                    ? 'Questions are answered only from this document. Anything it cannot support, it will say so rather than invent.'
                    : 'Nothing is selected yet. Drop a document into the ingest panel and the transcript starts here.'}
                </div>
              </div>
            </Panel>
          ) : (
            // One turn at a time, always at full width — whichever the rail
            // has selected, defaulting to the newest. Older turns are reached
            // by selecting them, not by stacking them on top of the answer
            // someone just asked for.
            selectedTurn && (
              <Panel key={selectedTurn.id}>
                {viewingOlder && (
                  <div
                    className="flex flex-wrap items-center justify-between gap-3 px-5 py-3"
                    style={{ borderBottom: 'var(--brd-w) solid var(--brd)' }}
                  >
                    <span className="text-[11px] font-extrabold uppercase tracking-[0.12em] opacity-55">
                      Viewing an earlier turn
                    </span>
                    <Button variant="chip" onClick={() => setSelectedTurnId(null)}>
                      Back to latest
                    </Button>
                  </div>
                )}
                {renderTurnBody(selectedTurn)}
              </Panel>
            )
          )}
        </div>

        {/* Pipeline and Collections are reference material, not part of the
            ask-and-read loop — full width underneath, so neither pushes the
            answer down the page. */}
        <div className={`grid min-w-0 gap-7 lg:col-span-2 ${hasRail ? 'xl:col-span-3' : ''}`}>
          <Panel>
            <PanelHeader
              title="Pipeline"
              right={<span className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-50">illustrative</span>}
            />
            <div className="overflow-x-auto p-6">
              <div className="min-w-[720px] max-w-[940px]">
                <PipelineVisualiser stage={pipeStage} />
              </div>
            </div>
          </Panel>

          <CollectionsPanel refreshToken={refreshToken} />
        </div>
      </div>
    </main>
  )
}
