'use client'

// Mandatory gate shown once per signed-in account per policy version — the
// user must scroll the policy text to its end before the checkbox unlocks,
// and acceptance is recorded server-side (APP.main's /auth/accept-privacy-policy)
// so it survives across devices and sessions, not just this browser.

import { useCallback, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

import { useAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/useToast'
import { PRIVACY_POLICY_SECTIONS } from '@/lib/privacyPolicy'
import { ApiError, acceptPrivacyPolicy } from '@/services/api'
import { Button } from '@/components/ui'

// A few px of slack: some browsers never report scrollTop+clientHeight as
// exactly equal to scrollHeight due to subpixel rounding.
const BOTTOM_THRESHOLD_PX = 24

export function PrivacyConsentModal() {
  const router = useRouter()
  const { user, isAuthenticated, hydrated, markPrivacyPolicyAccepted, signOut } = useAuth()
  const { flash } = useToast()

  const [scrolledToBottom, setScrolledToBottom] = useState(false)
  const [agreed, setAgreed] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  const onScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - BOTTOM_THRESHOLD_PX) {
      setScrolledToBottom(true)
    }
  }, [])

  // A short viewport-height policy (tall screen, few sections) may never fire
  // a scroll event at all — nothing to scroll past, so nothing should block.
  useEffect(() => {
    const el = scrollRef.current
    if (el && el.scrollHeight <= el.clientHeight + BOTTOM_THRESHOLD_PX) {
      setScrolledToBottom(true)
    }
  }, [isAuthenticated, user])

  const onAgree = useCallback(async () => {
    setSubmitting(true)
    try {
      await acceptPrivacyPolicy()
      markPrivacyPolicyAccepted()
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail : 'Could not reach the backend.'
      flash(`Could not record your acceptance — ${detail}`)
    } finally {
      setSubmitting(false)
    }
  }, [flash, markPrivacyPolicyAccepted])

  const onDecline = useCallback(() => {
    signOut()
    router.replace('/signin')
  }, [router, signOut])

  if (!hydrated || !isAuthenticated || !user || user.privacyPolicyAccepted) return null

  return (
    <div
      role="presentation"
      className="fixed inset-0 z-[70] grid place-items-center p-6"
      style={{ background: 'color-mix(in srgb, #000 60%, transparent)', backdropFilter: 'blur(4px)' }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="privacy-consent-title"
        // No onClick-outside / Escape dismissal on purpose — this gate is
        // mandatory, not a convenience dialog the user can click past.
        className="anim-rise grid w-full max-w-[560px] grid-rows-[auto_minmax(0,1fr)_auto] gap-0 overflow-hidden"
        style={{
          background: 'var(--panel-solid)',
          border: 'var(--brd-w) solid var(--brd)',
          boxShadow: 'var(--shadow)',
          maxHeight: '86vh',
        }}
      >
        <div className="p-6 pb-4">
          <h2 id="privacy-consent-title" className="m-0 mb-2 text-[22px] font-extrabold tracking-[-0.02em]">
            Before you continue
          </h2>
          <p className="m-0 text-[13.5px] leading-[1.55] opacity-70">
            Please read our Privacy Policy and confirm you agree to it. Scroll to the end to
            unlock the checkbox below.
          </p>
        </div>

        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="grid gap-6 overflow-y-auto px-6 py-4"
          style={{ borderTop: 'var(--brd-w) solid var(--brd)', borderBottom: 'var(--brd-w) solid var(--brd)' }}
        >
          {PRIVACY_POLICY_SECTIONS.map((section) => (
            <section key={section.heading}>
              <h3 className="m-0 mb-[6px] text-[14px] font-extrabold tracking-[-0.01em]">{section.heading}</h3>
              <div className="grid gap-2">
                {section.body.map((paragraph, i) => (
                  <p key={i} className="m-0 text-[13px] leading-[1.6] opacity-70">
                    {paragraph}
                  </p>
                ))}
              </div>
            </section>
          ))}
          <div className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-40">
            — end of policy —
          </div>
        </div>

        <div className="grid gap-4 p-6 pt-4">
          <Link
            href="/privacy"
            target="_blank"
            rel="noreferrer"
            className="text-[13px] font-extrabold underline"
            style={{ color: 'var(--accent-hi)' }}
          >
            View more — full Privacy Policy &amp; Terms
          </Link>

          <label className="flex items-start gap-3 text-[13.5px] leading-[1.5]" style={{ opacity: scrolledToBottom ? 1 : 0.45 }}>
            <input
              type="checkbox"
              checked={agreed}
              disabled={!scrolledToBottom}
              onChange={(e) => setAgreed(e.target.checked)}
              className="mt-[3px]"
            />
            <span>
              I have read and agree to the Privacy Policy.
              {!scrolledToBottom && (
                <span className="block text-[11.5px] font-extrabold opacity-70">
                  Scroll to the end of the policy above to enable this.
                </span>
              )}
            </span>
          </label>

          <div className="flex flex-wrap items-center gap-3">
            <Button variant="cta" disabled={!agreed || submitting} onClick={() => void onAgree()}>
              {submitting ? 'Saving…' : 'Agree and continue'}
            </Button>
            <Button variant="ghost" onClick={onDecline}>
              Sign out instead
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
