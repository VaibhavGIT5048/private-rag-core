'use client'

// ONE health poller for the whole app, published via context. The status badge,
// the reactive background, and the /workbench guard all read from here — no
// per-component polling.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import { POLL_MS, STORAGE_KEYS } from '@/config'
import { getHealth, warmup } from '@/services/api'
import type { HealthState, HealthStatus } from '@/types/api'

interface HealthContextValue {
  health: HealthStatus | null
  state: HealthState
  attempts: number
  lastCheckedAt: number | null
  /** True once /health has returned ok at any point this session or previously. */
  everConnected: boolean
  /** Backend was connected, then stopped responding. */
  reconnecting: boolean
  /** Offline, but recently enough that a scale-to-zero wake is the likely
   *  explanation. Drives "waking up" copy instead of "unavailable" — after the
   *  window passes it flips back, because a wake that never finishes is an
   *  outage and saying otherwise would be misleading. */
  waking: boolean
  isConnected: boolean
  checkNow: () => void
}

/** A cold start is typically well under a minute; past this it is not a wake. */
const WAKING_WINDOW_MS = 90_000

const HealthContext = createContext<HealthContextValue | null>(null)

export function HealthProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [state, setState] = useState<HealthState>('checking')
  const [attempts, setAttempts] = useState(0)
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null)
  const [everConnected, setEverConnected] = useState(false)
  const [offlineSince, setOfflineSince] = useState<number | null>(null)

  const intervalRef = useRef(POLL_MS.other)

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)
  // `poll` is memoised with no deps so the loop is never torn down; reading the
  // current state through a ref keeps the cold/warm and cadence decisions
  // correct without putting `state` in that dependency list.
  const stateRef = useRef<HealthState>('checking')
  stateRef.current = state

  useEffect(() => {
    try {
      if (localStorage.getItem(STORAGE_KEYS.hasConnected) === '1') setEverConnected(true)
    } catch {
      // localStorage can throw in private browsing modes; non-fatal.
    }
  }, [])

  // Start the wake as early as the app loads, not on reaching /home. The models
  // take longer to load than the container takes to start, so the sooner this
  // is kicked off the more of it overlaps with the user signing in or reading
  // the page — which is the whole point. Fire-and-forget: it returns 202
  // immediately and failure is handled by the poller below.
  useEffect(() => {
    void warmup()
  }, [])

  const poll = useCallback(async () => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setAttempts((n) => n + 1)
    try {
      // Treat "not currently connected" as possibly-waking and give the request
      // the long budget. Only a probe against a backend we know is up gets the
      // short one, so a genuine outage is still noticed quickly.
      const body = await getHealth(controller.signal, stateRef.current !== 'ok')
      if (!mountedRef.current) return
      setHealth(body)
      setState(body.status === 'ok' ? 'ok' : 'degraded')
      if (body.status === 'ok') setOfflineSince(null)
      setLastCheckedAt(Date.now())
      if (body.status === 'ok') {
        setEverConnected(true)
        try {
          localStorage.setItem(STORAGE_KEYS.hasConnected, '1')
        } catch {
          /* ignore */
        }
      }
    } catch {
      if (!mountedRef.current) return
      setHealth(null)
      setState('offline')
      // Stamp only the transition into offline, so the "waking" window is
      // measured from when contact was lost rather than from each retry.
      setOfflineSince((current) => current ?? Date.now())
      setLastCheckedAt(Date.now())
    }
  }, [])

  // Self-rescheduling loop so a cadence change takes effect on the next tick
  // without cancelling an in-flight request.
  useEffect(() => {
    mountedRef.current = true
    let stopped = false

    const tick = async () => {
      await poll()
      if (stopped) return
      // Offline: check back quickly so a finished wake is picked up promptly
      // rather than sitting behind a 20s tick.
      const delay = stateRef.current === 'ok' ? intervalRef.current : POLL_MS.offline
      timerRef.current = setTimeout(tick, delay)
    }
    tick()

    return () => {
      stopped = true
      mountedRef.current = false
      if (timerRef.current) clearTimeout(timerRef.current)
      abortRef.current?.abort()
    }
  }, [poll])

  const checkNow = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    void poll().then(() => {
      timerRef.current = setTimeout(function again() {
        void poll().then(() => {
          timerRef.current = setTimeout(again, intervalRef.current)
        })
      }, intervalRef.current)
    })
  }, [poll])

  const value = useMemo<HealthContextValue>(
    () => ({
      health,
      state,
      attempts,
      lastCheckedAt,
      everConnected,
      reconnecting: state === 'offline' && everConnected,
      waking:
        state === 'offline' &&
        offlineSince !== null &&
        (lastCheckedAt ?? 0) - offlineSince < WAKING_WINDOW_MS,
      isConnected: state === 'ok',
      checkNow,
    }),
    [health, state, attempts, lastCheckedAt, everConnected, offlineSince, checkNow],
  )

  return <HealthContext.Provider value={value}>{children}</HealthContext.Provider>
}

export function useHealth() {
  const ctx = useContext(HealthContext)
  if (!ctx) throw new Error('useHealth must be used inside <HealthProvider>')
  return ctx
}
