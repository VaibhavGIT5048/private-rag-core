'use client'

// JWT + current user, held in localStorage so a reload doesn't sign you out.
// The token is issued by whichever of the three methods was used (GitHub,
// Google, or email/OTP) — from here on they are indistinguishable.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { useRouter } from 'next/navigation'

import { STORAGE_KEYS } from '@/config'
import type { AuthTokenResponse, User } from '@/types/api'

interface AuthValue {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  /** False until localStorage has been read. Route guards must wait for this,
   *  or they redirect a signed-in user to /signin on every refresh. */
  hydrated: boolean
  signIn: (payload: AuthTokenResponse) => void
  signOut: () => void
  /** Marks the current user as having accepted the privacy policy, once the
   *  backend confirms it — updates local state so the consent modal closes
   *  without waiting for a fresh sign-in. */
  markPrivacyPolicyAccepted: () => void
}

const AuthContext = createContext<AuthValue | null>(null)

function readStored(): { token: string | null; user: User | null } {
  try {
    const token = localStorage.getItem(STORAGE_KEYS.jwt)
    const raw = localStorage.getItem(STORAGE_KEYS.user)
    return { token, user: raw ? (JSON.parse(raw) as User) : null }
  } catch {
    return { token: null, user: null }
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null)
  const [user, setUser] = useState<User | null>(null)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    const stored = readStored()
    setToken(stored.token)
    setUser(stored.user)
    setHydrated(true)

    // services/api.ts clears storage and fires this when any request 401s, so
    // an expired token updates the UI without every component polling for it.
    const onSignedOut = () => {
      setToken(null)
      setUser(null)
    }
    window.addEventListener('rag:signed-out', onSignedOut)
    return () => window.removeEventListener('rag:signed-out', onSignedOut)
  }, [])

  const signIn = useCallback((payload: AuthTokenResponse) => {
    const nextUser: User = {
      id: payload.user_id,
      email: payload.email,
      privacyPolicyAccepted: payload.privacy_policy_accepted,
    }
    setToken(payload.access_token)
    setUser(nextUser)
    try {
      localStorage.setItem(STORAGE_KEYS.jwt, payload.access_token)
      localStorage.setItem(STORAGE_KEYS.user, JSON.stringify(nextUser))
    } catch {
      // Private browsing: the session still works, it just won't survive a reload.
    }
  }, [])

  const markPrivacyPolicyAccepted = useCallback(() => {
    setUser((current) => {
      if (!current) return current
      const next = { ...current, privacyPolicyAccepted: true }
      try {
        localStorage.setItem(STORAGE_KEYS.user, JSON.stringify(next))
      } catch {
        /* ignore */
      }
      return next
    })
  }, [])

  const signOut = useCallback(() => {
    setToken(null)
    setUser(null)
    try {
      localStorage.removeItem(STORAGE_KEYS.jwt)
      localStorage.removeItem(STORAGE_KEYS.user)
    } catch {
      /* ignore */
    }
  }, [])

  const value = useMemo<AuthValue>(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(token),
      hydrated,
      signIn,
      signOut,
      markPrivacyPolicyAccepted,
    }),
    [user, token, hydrated, signIn, signOut, markPrivacyPolicyAccepted],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}

/** Route guard for pages that touch a user's documents. Returns `ready` so the
 *  caller can render a placeholder instead of flashing protected content
 *  during the redirect. */
export function useRequireAuth() {
  const { isAuthenticated, hydrated } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (hydrated && !isAuthenticated) router.replace('/signin')
  }, [hydrated, isAuthenticated, router])

  return { ready: hydrated && isAuthenticated }
}
