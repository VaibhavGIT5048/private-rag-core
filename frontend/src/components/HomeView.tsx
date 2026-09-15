'use client'

// Every document this account has ingested. Resuming one carries its chat
// history with it, which is the point: before multi-document support, a second
// upload destroyed the first and there was nothing to come back to.

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { FileText, Plus, Trash2 } from 'lucide-react'

import { useAuth, useRequireAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/useToast'
import { formatApiError } from '@/lib/apiError'
import { deleteDocument, listDocuments } from '@/services/api'
import type { DocumentSummary } from '@/types/api'
import { Button, Eyebrow, Mono, Panel, PanelHeader, Spinner } from '@/components/ui'

function formatDate(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

export function HomeView() {
  const { ready } = useRequireAuth()
  const { user } = useAuth()
  const { flash } = useToast()
  const router = useRouter()

  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  // Warmup now fires once app-wide from HealthProvider, which happens earlier
  // than reaching this page — so there is deliberately no call here.

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      setDocuments(await listDocuments(signal))
      setError(null)
    } catch (err) {
      if (signal?.aborted) return
      setError(formatApiError(err, 'Could not load your documents.'))
    }
  }, [])

  useEffect(() => {
    if (!ready) return
    const controller = new AbortController()
    void refresh(controller.signal)
    return () => controller.abort()
  }, [ready, refresh])

  const remove = useCallback(
    async (doc: DocumentSummary) => {
      if (!window.confirm(`Delete "${doc.filename}"? Its chat history goes too. This cannot be undone.`)) {
        return
      }
      setDeleting(doc.id)
      try {
        await deleteDocument(doc.id)
        setDocuments((current) => (current ?? []).filter((d) => d.id !== doc.id))
        flash('Document deleted.')
      } catch (err) {
        flash(formatApiError(err, 'Delete failed.'))
      } finally {
        setDeleting(null)
      }
    },
    [flash],
  )

  if (!ready) {
    return (
      <main className="mx-auto w-full max-w-[1100px] px-6 py-24">
        <Spinner size={18} />
      </main>
    )
  }

  return (
    <main className="mx-auto w-full max-w-[1100px] px-6 py-14">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>Your documents</Eyebrow>
          <h1
            className="mt-3 text-[38px] font-extrabold leading-[1.05] tracking-[-0.03em]"
            style={{ color: 'var(--ink)' }}
          >
            Pick up where you left off.
          </h1>
          {user && <p className="mt-2 text-[13px] opacity-60">Signed in as {user.email}</p>}
        </div>
        <Button variant="solid" onClick={() => router.push('/workbench')}>
          <Plus size={14} aria-hidden />
          Ingest a document
        </Button>
      </div>

      {error && (
        <div
          role="alert"
          className="mt-6 px-[14px] py-[12px] text-[13px]"
          style={{ background: 'var(--chip-bg)', border: 'var(--brd-w) solid var(--accent)', borderRadius: 'var(--r-sm)' }}
        >
          {error}
        </div>
      )}

      <Panel className="mt-8">
        <PanelHeader
          title="Ingested documents"
          right={
            <span className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-50">
              {documents ? `${documents.length} total` : '—'}
            </span>
          }
        />

        {documents === null ? (
          <div className="flex items-center gap-3 p-6 text-[14px] opacity-70">
            <Spinner size={15} /> Loading…
          </div>
        ) : documents.length === 0 ? (
          <div className="grid gap-3 p-6">
            <p className="text-[15px] leading-[1.6]">
              Nothing ingested yet. Upload a document and it will appear here, with its chat history,
              ready to resume any time.
            </p>
            <div>
              <Button variant="solid" onClick={() => router.push('/workbench')}>
                <Plus size={14} aria-hidden />
                Ingest your first document
              </Button>
            </div>
          </div>
        ) : (
          <ul className="grid" style={{ gap: 'var(--brd-w)', background: 'var(--brd)' }}>
            {documents.map((doc) => (
              <li
                key={doc.id}
                className="flex flex-wrap items-center justify-between gap-4 px-5 py-4"
                style={{ background: 'var(--panel-solid)' }}
              >
                <div className="min-w-0 flex items-start gap-3">
                  <FileText size={16} aria-hidden className="mt-[3px] shrink-0 opacity-60" />
                  <div className="min-w-0">
                    <div className="truncate text-[15px] font-extrabold">{doc.filename}</div>
                    <div className="mt-[3px] text-[12px] opacity-60">
                      {formatDate(doc.ingested_at)}
                      {doc.pages ? ` · ${doc.pages} pages` : ''}
                      {doc.indexed_chunks ? ` · ${doc.indexed_chunks} chunks indexed` : ''}
                    </div>
                    <Mono className="mt-[3px] block text-[10.5px] opacity-40">{doc.id}</Mono>
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  <Button variant="solid" onClick={() => router.push(`/workbench?doc=${encodeURIComponent(doc.id)}`)}>
                    Resume
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={() => void remove(doc)}
                    disabled={deleting === doc.id}
                    aria-label={`Delete ${doc.filename}`}
                  >
                    {deleting === doc.id ? <Spinner size={13} /> : <Trash2 size={14} aria-hidden />}
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </main>
  )
}
