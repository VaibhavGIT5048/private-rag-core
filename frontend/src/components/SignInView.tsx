'use client'

// One entry point for all three sign-in methods. GitHub and Google redirect to
// the provider; email/password posts straight to the backend. All three end at
// the same place — a JWT in useAuth — so nothing downstream cares which was used.

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Eye, EyeOff, Github, Mail } from 'lucide-react'

import { AUTH_CALLBACK_URL, GITHUB_CLIENT_ID, GOOGLE_CLIENT_ID, STORAGE_KEYS } from '@/config'
import { useAuth } from '@/hooks/useAuth'
import { useToast } from '@/hooks/useToast'
import { ApiError, signInEmail, signUpEmail } from '@/services/api'
import { Button, Eyebrow, Panel, PanelHeader, Spinner } from '@/components/ui'

type Mode = 'signin' | 'signup'

const fieldStyle = {
  color: 'var(--ink)',
  background: 'var(--chip-bg)',
  border: 'var(--brd-w) solid var(--brd)',
  borderRadius: 'var(--r-sm)',
} as const

export function SignInView() {
  const router = useRouter()
  const { isAuthenticated, hydrated, signIn } = useAuth()
  const { flash } = useToast()

  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showPassword, setShowPassword] = useState(false)

  // Already signed in — don't make them do it again.
  useEffect(() => {
    if (hydrated && isAuthenticated) router.replace('/home')
  }, [hydrated, isAuthenticated, router])

  const startOauth = useCallback((provider: 'github' | 'google') => {
    const clientId = provider === 'github' ? GITHUB_CLIENT_ID : GOOGLE_CLIENT_ID
    if (!clientId) {
      setError(
        `${provider === 'github' ? 'GitHub' : 'Google'} sign-in isn't configured for this deployment.`,
      )
      return
    }

    // `state` does double duty: CSRF defence, and it tells the shared callback
    // page which provider to exchange the code with.
    const state = `${provider}:${crypto.randomUUID()}`
    try {
      sessionStorage.setItem(STORAGE_KEYS.oauthState, state)
    } catch {
      /* private browsing — the callback falls back to parsing the prefix */
    }

    const url =
      provider === 'github'
        ? `https://github.com/login/oauth/authorize?client_id=${encodeURIComponent(clientId)}` +
          `&redirect_uri=${encodeURIComponent(AUTH_CALLBACK_URL)}&scope=read:user%20user:email&state=${encodeURIComponent(state)}`
        : `https://accounts.google.com/o/oauth2/v2/auth?client_id=${encodeURIComponent(clientId)}` +
          `&redirect_uri=${encodeURIComponent(AUTH_CALLBACK_URL)}&response_type=code` +
          `&scope=${encodeURIComponent('openid email profile')}&state=${encodeURIComponent(state)}`

    window.location.assign(url)
  }, [])

  const submitEmail = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault()
      if (busy) return
      setError(null)
      setBusy(true)
      try {
        if (mode === 'signup') {
          await signUpEmail(email.trim(), password)
          // The account exists but is unverified until the emailed code is
          // entered, so there is no token yet — hand off to /verify-email.
          router.push(`/verify-email?email=${encodeURIComponent(email.trim())}`)
          return
        }
        const result = await signInEmail(email.trim(), password)
        signIn(result)
        flash('Signed in.')
        router.replace('/home')
      } catch (err) {
        const apiErr = err instanceof ApiError ? err : null
        // An unverified account can't log in — send them to finish, rather
        // than leaving them re-typing a password that was never the problem.
        if (apiErr?.detail?.toLowerCase().includes('not verified')) {
          router.push(`/verify-email?email=${encodeURIComponent(email.trim())}`)
          return
        }
        setError(apiErr?.detail ?? (err instanceof Error ? err.message : 'Sign-in failed'))
      } finally {
        setBusy(false)
      }
    },
    [busy, email, flash, mode, password, router, signIn],
  )

  return (
    <main className="mx-auto w-full max-w-[520px] px-6 py-16">
      <Eyebrow>Sign in</Eyebrow>
      <h1
        className="mt-3 text-[38px] font-extrabold leading-[1.05] tracking-[-0.03em]"
        style={{ color: 'var(--ink)' }}
      >
        Your documents, your account.
      </h1>
      <p className="mt-3 text-[15px] leading-[1.6] opacity-70">
        Documents and chat history are private to your account. Sign in with GitHub, Google, or an
        email address.
      </p>

      <div className="mt-6 flex gap-2" role="tablist" aria-label="Account access">
        <Button
          variant={mode === 'signin' ? 'solid' : 'ghost'}
          onClick={() => { setMode('signin'); setError(null) }}
          aria-selected={mode === 'signin'}
          role="tab"
        >
          Sign in
        </Button>
        <Button
          variant={mode === 'signup' ? 'solid' : 'ghost'}
          onClick={() => { setMode('signup'); setError(null) }}
          aria-selected={mode === 'signup'}
          role="tab"
        >
          Create account
        </Button>
      </div>

      <Panel className="mt-8">
        <PanelHeader title={mode === 'signin' ? 'Sign in' : 'Create account'} />
        <div className="grid gap-4 p-5">
          <Button variant="ghost" onClick={() => startOauth('github')} disabled={busy}>
            <Github size={15} aria-hidden />
            Continue with GitHub
          </Button>
          <Button variant="ghost" onClick={() => startOauth('google')} disabled={busy}>
            <Mail size={15} aria-hidden />
            Continue with Google
          </Button>

          <div className="flex items-center gap-3 py-1">
            <span className="h-px flex-1" style={{ background: 'var(--brd)' }} />
            <span className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-50">or</span>
            <span className="h-px flex-1" style={{ background: 'var(--brd)' }} />
          </div>

          <form onSubmit={submitEmail} className="grid gap-3">
            <label htmlFor="email" className="text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-55">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-[12px] py-[10px] text-[14px]"
              style={fieldStyle}
            />

            <label htmlFor="password" className="mt-1 text-[11px] font-extrabold uppercase tracking-[0.1em] opacity-55">
              Password
            </label>
            <div className="relative">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                required
                minLength={8}
                autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === 'signup' ? 'At least 8 characters' : '••••••••'}
                className="w-full px-[12px] py-[10px] pr-10 text-[14px]"
                style={fieldStyle}
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                aria-pressed={showPassword}
                className="absolute right-0 top-0 flex h-full w-10 cursor-pointer items-center justify-center border-0 bg-transparent opacity-60 hover:opacity-100"
              >
                {showPassword ? <EyeOff size={16} aria-hidden /> : <Eye size={16} aria-hidden />}
              </button>
            </div>

            {error && (
              <div
                role="alert"
                className="px-[12px] py-[10px] text-[13px]"
                style={{ background: 'var(--chip-bg)', border: 'var(--brd-w) solid var(--accent)', borderRadius: 'var(--r-sm)' }}
              >
                {error}
              </div>
            )}

            <Button type="submit" variant="solid" disabled={busy || !email || !password}>
              {busy ? <Spinner size={14} /> : null}
              {mode === 'signin' ? 'Sign in' : 'Create account'}
            </Button>
          </form>

          <button
            type="button"
            onClick={() => {
              setMode(mode === 'signin' ? 'signup' : 'signin')
              setError(null)
            }}
            className="text-left text-[13px] underline opacity-70 hover:opacity-100"
          >
            {mode === 'signin' ? "No account? Create one" : 'Already have an account? Sign in'}
          </button>
        </div>
      </Panel>
    </main>
  )
}
