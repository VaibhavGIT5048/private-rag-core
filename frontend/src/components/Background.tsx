'use client'

// Ambient layers: drifting aurora and a particle constellation that reacts to
// what the app is doing.
//
// Performance discipline (this runs on every route, so it must be cheap):
//  - only transform/opacity are animated in CSS; the canvas draws to a bitmap
//  - particle count scales with viewport area and is hard-capped
//  - devicePixelRatio clamped to 2
//  - rAF loop stops entirely when the tab is hidden or motion is off

import { useEffect, useRef } from 'react'

import { useActivity } from '@/hooks/useActivity'
import { useUiPrefs } from '@/hooks/useUiPrefs'
import type { Activity } from '@/types/api'

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  r: number
}

interface ActivityProfile {
  speed: number
  alpha: number
  link: number
}

function profileFor(activity: Activity, dark: boolean): ActivityProfile {
  switch (activity) {
    case 'querying':
      // Accelerate and brighten — the UI visibly "thinks".
      return { speed: 2.6, alpha: dark ? 0.85 : 0.5, link: 150 }
    case 'ingesting':
      // Slow convergence, suggesting material being drawn in and indexed.
      return { speed: 0.6, alpha: dark ? 0.8 : 0.48, link: 130 }
    case 'connected':
      // One-shot celebratory outward burst.
      return { speed: 3.4, alpha: 1, link: 170 }
    case 'offline':
      return { speed: 0.3, alpha: dark ? 0.22 : 0.16, link: 80 }
    default:
      return { speed: 1, alpha: dark ? 0.5 : 0.34, link: 120 }
  }
}

export function Background() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const { activity } = useActivity()
  const { theme, motionOff, motionFull } = useUiPrefs()

  // Read the latest values inside the rAF loop without restarting it.
  const activityRef = useRef(activity)
  activityRef.current = activity
  const themeRef = useRef(theme)
  themeRef.current = theme
  const motionOffRef = useRef(motionOff)
  motionOffRef.current = motionOff

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let particles: Particle[] = []
    let dpr = 1
    let raf: number | null = null

    const size = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.floor(window.innerWidth * dpr)
      canvas.height = Math.floor(window.innerHeight * dpr)
      const area = window.innerWidth * window.innerHeight
      const count = Math.max(28, Math.min(110, Math.round(area / 17000)))
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.28,
        vy: (Math.random() - 0.5) * 0.28,
        r: Math.random() * 1.6 + 0.5,
      }))
    }

    const frame = () => {
      raf = requestAnimationFrame(frame)

      const w = window.innerWidth
      const h = window.innerHeight
      const dark = themeRef.current === 'dark'
      const still = motionOffRef.current
      const { speed, alpha, link } = profileFor(activityRef.current, dark)
      const rgb = dark ? '124,131,255' : '236,48,19'

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const cx = w / 2
      const cy = h / 2
      const t = performance.now() / 1000
      const effSpeed = still ? 0 : speed

      for (const p of particles) {
        if (effSpeed > 0) {
          p.x += p.vx * effSpeed
          p.y += p.vy * effSpeed

          if (activityRef.current === 'ingesting') {
            p.x += (cx - p.x) * 0.004
            p.y += (cy - p.y) * 0.004
          } else if (activityRef.current === 'querying') {
            p.y -= 0.5
          } else if (activityRef.current === 'connected') {
            p.x += (p.x - cx) * 0.012
            p.y += (p.y - cy) * 0.012
          }

          if (p.x < -20) p.x = w + 20
          if (p.x > w + 20) p.x = -20
          if (p.y < -20) p.y = h + 20
          if (p.y > h + 20) p.y = -20
        }

        const breathe = still ? 1 : 1 + Math.sin(t + p.x * 0.01) * 0.18
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r * breathe, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${rgb},${alpha})`
        ctx.fill()
      }

      // Constellation links. O(n²) over a capped n (~110 max) stays well inside
      // frame budget; skipped entirely outside 'full' motion.
      if (motionFull) {
        ctx.lineWidth = 1
        for (let i = 0; i < particles.length; i++) {
          for (let j = i + 1; j < particles.length; j++) {
            const a = particles[i]
            const b = particles[j]
            const dist = Math.hypot(a.x - b.x, a.y - b.y)
            if (dist < link) {
              ctx.strokeStyle = `rgba(${rgb},${alpha * 0.28 * (1 - dist / link)})`
              ctx.beginPath()
              ctx.moveTo(a.x, a.y)
              ctx.lineTo(b.x, b.y)
              ctx.stroke()
            }
          }
        }
      }
    }

    const start = () => {
      if (raf === null) raf = requestAnimationFrame(frame)
    }
    const stop = () => {
      if (raf !== null) {
        cancelAnimationFrame(raf)
        raf = null
      }
    }

    // Zero CPU in a hidden tab.
    const onVisibility = () => (document.hidden ? stop() : start())

    size()
    start()
    window.addEventListener('resize', size)
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      stop()
      window.removeEventListener('resize', size)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [motionFull])

  return (
    <>
      <div
        className="ambient pointer-events-none fixed z-0"
        aria-hidden
        style={{
          inset: '-25%',
          opacity: 'var(--aurora)',
          background: 'conic-gradient(from 0deg, #4f46e5, #06b6d4, #d946ef, #4f46e5)',
          filter: 'blur(120px)',
          animation: motionOff ? 'none' : 'drift 46s linear infinite',
        }}
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none fixed inset-0 z-0 h-full w-full"
        aria-hidden
      />
    </>
  )
}
