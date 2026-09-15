'use client'

// Sticky chrome shown on every route: brand, nav, connectivity badge, theme
// toggle, repo link. Also hosts the two global banners (backend-connected on
// landing, reconnecting on workbench).

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { Github } from 'lucide-react'

import { API_BASE_URL, REPO_URL } from '@/config'
import { useHealth } from '@/hooks/useHealth'
import { useAuth } from '@/hooks/useAuth'
import { useUiPrefs } from '@/hooks/useUiPrefs'
import { Button, Spinner } from '@/components/ui'
import { StatusBadge } from '@/components/StatusBadge'
import { BrandMark } from '@/components/BrandMark'

const NAV = [
  { href: '/', label: 'Overview' },
  { href: '/home', label: 'Home' },
  { href: '/workbench', label: 'Workbench' },
]

export function Header() {
  const router = useRouter()
  const pathname = usePathname() ?? '/'
  const { theme, setTheme } = useUiPrefs()
  const { isConnected, reconnecting } = useHealth()
  const { user, hydrated, isAuthenticated, signOut } = useAuth()

  const isActive = (href: string) =>
    href === '/' ? pathname === '/' : pathname.startsWith(href)

  const dark = theme === 'dark'
  const themeBtn = (active: boolean): React.CSSProperties => ({
    background: active ? 'var(--accent)' : 'transparent',
    color: active ? 'var(--on-accent)' : 'var(--ink)',
    opacity: active ? 1 : 0.55,
  })

  return (
    <>
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[100] focus:px-4 focus:py-2 focus:text-[13px] focus:font-extrabold"
        style={{ background: 'var(--accent)', color: 'var(--on-accent)', borderRadius: 'var(--r-sm)' }}
      >
        Skip to content
      </a>
      <header
        className="sticky top-0 z-40 flex flex-wrap items-center gap-5 px-[26px] py-3"
        style={{
          background: 'var(--head-bg)',
          backdropFilter: 'var(--blur)',
          borderBottom: 'var(--brd-w) solid var(--brd)',
          boxShadow: 'var(--head-shadow)',
        }}
      >
        <Link href="/" className="flex items-center gap-[10px]" style={{ color: 'var(--ink)' }}>
          <BrandMark size={28} />
          <span className="text-[15px] font-extrabold tracking-[-0.02em]">
            GROUNDED<span style={{ color: 'var(--accent)' }}>·</span>RAG
          </span>
        </Link>

        <nav className="flex gap-[2px]" aria-label="Sections">
          {NAV.map((item) => {
            const active = isActive(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className="px-3 py-[7px] text-[12.5px] font-extrabold tracking-[0.02em] transition-opacity hover:opacity-100"
                style={{
                  borderRadius: 'var(--r-sm)',
                  background: active ? 'var(--accent)' : 'transparent',
                  color: active ? 'var(--on-accent)' : 'var(--ink)',
                  opacity: active ? 1 : 0.6,
                }}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="flex-1" />

        <StatusBadge />

        <div
          className="flex overflow-hidden"
          style={{ border: 'var(--brd-w) solid var(--brd)', borderRadius: 'var(--r-sm)' }}
          role="group"
          aria-label="Theme"
        >
          <button
            onClick={() => setTheme('light')}
            aria-pressed={!dark}
            className="cursor-pointer border-0 px-[10px] py-[6px] text-[11.5px] font-extrabold"
            style={themeBtn(!dark)}
          >
            Light
          </button>
          <button
            onClick={() => setTheme('dark')}
            aria-pressed={dark}
            className="cursor-pointer border-0 px-[10px] py-[6px] text-[11.5px] font-extrabold"
            style={themeBtn(dark)}
          >
            Dark
          </button>
        </div>

        {hydrated && (isAuthenticated ? (
          <div className="flex items-center gap-2">
            <span className="hidden max-w-[180px] truncate text-[11.5px] opacity-60 sm:inline">{user?.email}</span>
            <Button variant="ghost" onClick={() => { signOut(); router.replace('/signin') }}>Sign out</Button>
          </div>
        ) : (
          <Link href="/signin"><Button variant="solid">Sign in</Button></Link>
        ))}

        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-[6px] text-[12px] font-extrabold"
        >
          <Github size={14} aria-hidden /> GitHub
        </a>
      </header>

      {/* Landing: surface the connection without hijacking someone mid-read. */}
      {isConnected && pathname === '/' && (
        <div
          className="anim-rise relative z-30 flex flex-wrap items-center gap-4 px-[26px] py-[11px]"
          style={{ background: 'var(--accent)', color: 'var(--on-accent)' }}
        >
          <span className="text-[13px] font-extrabold tracking-[0.02em]">
            Backend connected on {API_BASE_URL} — the workbench is live.
          </span>
          <Link
            href="/workbench"
            className="ambient text-[13px] font-extrabold"
            style={{
              background: 'var(--on-accent)',
              color: 'var(--accent)',
              padding: '7px 14px',
              borderRadius: 'var(--r-sm)',
              animation: 'breathe 3.4s ease-in-out infinite',
            }}
          >
            Launch workbench →
          </Link>
        </div>
      )}

      {/* Lost the backend mid-session: never destructive, never wipes results. */}
      {reconnecting && pathname.startsWith('/workbench') && (
        <div
          role="alert"
          className="relative z-30 flex items-center gap-3 px-[26px] py-[10px] text-[13px]"
          style={{
            background: 'var(--warn-bg)',
            borderBottom: 'var(--brd-w) solid var(--brd)',
          }}
        >
          <Spinner size={13} />
          <span>
            <span className="font-extrabold">Reconnecting.</span> The backend stopped responding —
            your transcript is intact and submits are paused until it returns.
          </span>
        </div>
      )}
    </>
  )
}
