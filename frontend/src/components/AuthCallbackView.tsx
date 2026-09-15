'use client'

// Shared OAuth landing page for GitHub and Google.
//
// One page for both because the provider only ever redirects to a URL that was
// registered in advance, and GitHub allows exactly one per OAuth App. The
// `state` parameter carries which provider started the round-trip, so the
// single registered URL can serve both.

import { useEffect, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { AUTH_CALLBACK_URL, STORAGE_KEYS } from '@/config'
import { useAuth } from '@/hooks/useAuth'
import { formatApiError } from '@/lib/apiError'
import { signInWithGithub, signInWithGoogle } from '@/services/api'
import { Eyebrow, Panel, Spinner } from '@/components/ui'

export function AuthCallbackView() {
  const router = useRouter()
  const params = useSearchParams()
  const { signIn } = useAuth()
  const [error, setError] = useState<string | null>(null)
  // React 18 StrictMode mounts effects twice in development; an OAuth code is
  // single-use, so a second exchange would fail and surface a bogus error.
  const exchanged = useRef(false)

  useEffect(() => {
    if (exchanged.current) return
    exchanged.current = true

    const code = params.get('code')
    const state = params.get('state') ?? ''
    const providerError = params.get('error_description') ?? params.get('error')

    if (providerError) {
      setError(providerError)
      return
    }
    if (!code) {
      setError('No authorization code was returned.')
      return
    }

    let stored: string | null = null
    try {
      stored = sessionStorage.getItem(STORAGE_KEYS.oauthState)
      sessionStorage.removeItem(STORAGE_KEYS.oauthState)
    } catch {
      /* private browsing */
    }

    // CSRF check: a code delivered with a state we never issued did not come
    // from a sign-in this browser started. Skipped only when sessionStorage is
    // unavailable, where there is nothing to compare against.
    if (stored && stored !== state) {
      setError('Sign-in state did not match. Please start again.')
      return
    }

    const provider = state.split(':')[0] === 'google' ? 'google' : 'github'

    const run = async () => {
      try {
        const result =
          provider === 'google'
            ? await signInWithGoogle(code, AUTH_CALLBACK_URL)
            : await signInWithGithub(code)
        signIn(result)
        router.replace('/home')
      } catch (err) {
        setError(formatApiError(err, 'Sign-in failed. Please try again.'))
      }
    }
    void run()
  }, [params, router, signIn])

  return (
    <main className="mx-auto w-full max-w-[520px] px-6 py-24">
      <Eyebrow>{error ? 'Sign-in failed' : 'Signing you in'}</Eyebrow>
      <Panel className="mt-5">
        <div className="grid gap-4 p-6">
          {error ? (
            <>
              <p className="text-[15px] leading-[1.6]">{error}</p>
              <button
                type="button"
                onClick={() => router.replace('/signin')}
                className="text-left text-[13px] underline opacity-70 hover:opacity-100"
              >
                Back to sign in
              </button>
            </>
          ) : (
            <div className="flex items-center gap-3 text-[15px]">
              <Spinner size={16} />
              Completing sign-in…
            </div>
          )}
        </div>
      </Panel>
    </main>
  )
}
