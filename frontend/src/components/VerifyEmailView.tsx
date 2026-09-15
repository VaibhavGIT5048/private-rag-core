'use client'

// OTP entry after email signup. The code is emailed via Azure Communication
// Services; verifying it marks the account verified and returns the JWT.

import { useCallback, useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { AlertTriangle } from 'lucide-react'

import { useAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/useToast'
import { formatApiError } from '@/lib/apiError'
import { ApiError, resendOtp, verifyOtp } from '@/services/api'
import { Button, Eyebrow, Panel, PanelHeader, Spinner } from '@/components/ui'

// Matches the backend's own 60s cooldown, so the button is disabled for
// exactly as long as a resend would be silently swallowed anyway.
const RESEND_COOLDOWN_SECONDS = 60

export function VerifyEmailView() {
  const router = useRouter()
  const params = useSearchParams()
  const { signIn } = useAuth()
  const { flash } = useToast()

  const emailParam = params.get('email') ?? ''
  const [email, setEmail] = useState(emailParam)
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cooldown, setCooldown] = useState(0)

  useEffect(() => {
    if (cooldown <= 0) return
    const timer = setInterval(() => setCooldown((n) => Math.max(0, n - 1)), 1000)
    return () => clearInterval(timer)
  }, [cooldown])

  const submit = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault()
      if (busy) return
      setError(null)
      setBusy(true)
      try {
        const result = await verifyOtp(email.trim(), code.trim())
        signIn(result)
        flash('Email verified.')
        router.replace('/home')
      } catch (err) {
        const apiErr = err instanceof ApiError ? err : null
        setError(apiErr?.detail ?? 'That code was not accepted.')
      } finally {
        setBusy(false)
      }
    },
    [busy, code, email, flash, router, signIn],
  )

  const resend = useCallback(async () => {
    if (cooldown > 0 || !email) return
    setError(null)
    try {
      await resendOtp(email.trim())
      setCooldown(RESEND_COOLDOWN_SECONDS)
      // Deliberately vague, mirroring the backend: it returns the same
      // response whether or not the account exists, so the UI must not imply
      // one either — that would turn this into an email-enumeration oracle.
      flash('If that account needs verification, a new code is on its way.')
    } catch (err) {
      setError(formatApiError(err, 'Could not resend the code.'))
    }
  }, [cooldown, email, flash])

  return (
    <main className="mx-auto w-full max-w-[520px] px-6 py-16">
      <Eyebrow>Verify email</Eyebrow>
      <h1
        className="mt-3 text-[38px] font-extrabold leading-[1.05] tracking-[-0.03em]"
        style={{ color: 'var(--ink)' }}
      >
        Enter your code.
      </h1>
      <p className="mt-3 text-[15px] leading-[1.6] opacity-70">
        We sent a six-digit code to {email ? <strong>{email}</strong> : 'your email address'}. It
        expires in 10 minutes.
      </p>

      {/* Not decorative: the sender is an Azure-managed *.azurecomm.net domain
          with no reputation of its own, and these reliably land in spam. Saying
          so up front is the difference between a working signup and a dead end. */}
      <div
        className="mt-5 flex gap-3 px-[14px] py-[12px] text-[13px] leading-[1.55]"
        style={{ background: 'var(--chip-bg)', border: 'var(--brd-w) solid var(--brd)', borderRadius: 'var(--r-sm)' }}
      >
        <AlertTriangle size={16} aria-hidden className="mt-[2px] shrink-0" style={{ color: 'var(--accent)' }} />
        <span>
          <strong>Check your spam folder.</strong> The code is sent from an automated address that
          most mail providers treat as unfamiliar, so it often lands there rather than your inbox.
        </span>
      </div>

      <Panel className="mt-6">
        <PanelHeader title="Verification code" />
        <form onSubmit={submit} className="grid gap-3 p-5">
          {!emailParam && (
            <>
              <label htmlFor="v-email" className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-55">
                Email
              </label>
              <input
                id="v-email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-[12px] py-[10px] text-[14px]"
                style={{ color: 'var(--ink)', background: 'var(--chip-bg)', border: 'var(--brd-w) solid var(--brd)', borderRadius: 'var(--r-sm)' }}
              />
            </>
          )}

          <label htmlFor="code" className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-55">
            Six-digit code
          </label>
          <input
            id="code"
            inputMode="numeric"
            pattern="[0-9]*"
            maxLength={6}
            required
            autoFocus
            value={code}
            onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))}
            placeholder="000000"
            className="w-full px-[14px] py-[12px] text-[24px] font-extrabold tracking-[0.4em]"
            style={{ color: 'var(--ink)', background: 'var(--chip-bg)', border: 'var(--brd-w) solid var(--brd)', borderRadius: 'var(--r-sm)' }}
          />

          {error && (
            <div
              role="alert"
              className="px-[12px] py-[10px] text-[13px]"
              style={{ background: 'var(--chip-bg)', border: 'var(--brd-w) solid var(--accent)', borderRadius: 'var(--r-sm)' }}
            >
              {error}
            </div>
          )}

          <div className="mt-1 flex flex-wrap items-center gap-3">
            <Button type="submit" variant="solid" disabled={busy || code.length !== 6 || !email}>
              {busy ? <Spinner size={14} /> : null}
              Verify and continue
            </Button>
            <Button type="button" variant="ghost" onClick={() => void resend()} disabled={cooldown > 0 || !email}>
              {cooldown > 0 ? `Resend in ${cooldown}s` : 'Resend code'}
            </Button>
          </div>
        </form>
      </Panel>
    </main>
  )
}
