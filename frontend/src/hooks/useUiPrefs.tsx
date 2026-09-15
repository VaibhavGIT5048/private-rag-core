'use client'

// Theme (Light / Dark), persisted. Motion always runs full unless the OS
// itself asks for reduced motion — there is no manual override for it, so
// every animation in the app is a progressive enhancement over that one
// accessibility signal rather than a user-facing setting to maintain.

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

interface UiPrefsValue {
  theme: Theme
  /** True only when the OS's reduced-motion preference is set. */
  motionOff: boolean
  /** True whenever motion isn't off — gates the most expensive effects. */
  motionFull: boolean
  setTheme: (t: Theme) => void
  /** False until the client has read localStorage, so SSR markup stays stable. */
  hydrated: boolean
}

const UiPrefsContext = createContext<UiPrefsValue | null>(null)

export function UiPrefsProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('dark')
  const [motionOff, setMotionOff] = useState(false)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    try {
      const storedTheme = localStorage.getItem(STORAGE_KEYS.theme)
      if (storedTheme === 'light' || storedTheme === 'dark') setThemeState(storedTheme)
    } catch {
      /* private browsing — fall back to defaults */
    }

    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    setMotionOff(mq.matches)
    const onChange = (e: MediaQueryListEvent) => setMotionOff(e.matches)
    mq.addEventListener('change', onChange)
    setHydrated(true)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  // The theme attribute drives every CSS custom property in globals.css.
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    document.documentElement.dataset.motion = motionOff ? 'off' : 'full'
  }, [motionOff])

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem(STORAGE_KEYS.theme, t)
    } catch {
      /* ignore */
    }
  }, [])

  const value = useMemo<UiPrefsValue>(
    () => ({
      theme,
      motionOff,
      motionFull: !motionOff,
      setTheme,
      hydrated,
    }),
    [theme, motionOff, setTheme, hydrated],
  )

  return <UiPrefsContext.Provider value={value}>{children}</UiPrefsContext.Provider>
}

export function useUiPrefs() {
  const ctx = useContext(UiPrefsContext)
  if (!ctx) throw new Error('useUiPrefs must be used inside <UiPrefsProvider>')
  return ctx
}
