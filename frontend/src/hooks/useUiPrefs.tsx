'use client'

// Theme (Light / Dark) + motion quality (full / reduced / off), both
// persisted. Motion is forced to 'off' when the OS asks for reduced motion,
// so every animation in the app is a progressive enhancement.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { STORAGE_KEYS } from '@/config'

export type Theme = 'light' | 'dark'
export type Motion = 'full' | 'reduced' | 'off'

interface UiPrefsValue {
  theme: Theme
  motion: Motion
  /** True when motion is 'off' — either chosen or forced by the OS setting. */
  motionOff: boolean
  /** True only in 'full' — gates the most expensive effects. */
  motionFull: boolean
  setTheme: (t: Theme) => void
  setMotion: (m: Motion) => void
  /** False until the client has read localStorage, so SSR markup stays stable. */
  hydrated: boolean
}

const UiPrefsContext = createContext<UiPrefsValue | null>(null)

export function UiPrefsProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('dark')
  const [motion, setMotionState] = useState<Motion>('full')
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    try {
      const storedTheme = localStorage.getItem(STORAGE_KEYS.theme)
      if (storedTheme === 'light' || storedTheme === 'dark') setThemeState(storedTheme)

      const storedMotion = localStorage.getItem(STORAGE_KEYS.motion)
      if (storedMotion === 'full' || storedMotion === 'reduced' || storedMotion === 'off') {
        setMotionState(storedMotion)
      }
    } catch {
      /* private browsing — fall back to defaults */
    }

    // OS preference wins over the stored value.
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (mq.matches) setMotionState('off')
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) setMotionState('off')
    }
    mq.addEventListener('change', onChange)
    setHydrated(true)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  // The theme attribute drives every CSS custom property in globals.css.
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    document.documentElement.dataset.motion = motion
  }, [motion])

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem(STORAGE_KEYS.theme, t)
    } catch {
      /* ignore */
    }
  }, [])

  const setMotion = useCallback((m: Motion) => {
    setMotionState(m)
    try {
      localStorage.setItem(STORAGE_KEYS.motion, m)
    } catch {
      /* ignore */
    }
  }, [])

  const value = useMemo<UiPrefsValue>(
    () => ({
      theme,
      motion,
      motionOff: motion === 'off',
      motionFull: motion === 'full',
      setTheme,
      setMotion,
      hydrated,
    }),
    [theme, motion, setTheme, setMotion, hydrated],
  )

  return <UiPrefsContext.Provider value={value}>{children}</UiPrefsContext.Provider>
}

export function useUiPrefs() {
  const ctx = useContext(UiPrefsContext)
  if (!ctx) throw new Error('useUiPrefs must be used inside <UiPrefsProvider>')
  return ctx
}
