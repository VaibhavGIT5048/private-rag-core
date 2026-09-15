'use client'

// Thin fixed bar tracking how far down the current page the visitor has
// scrolled. Purely cosmetic — reads scrollTop directly rather than through
// useActivity/useHealth, since it has nothing to do with backend state.

import { useEffect, useState } from 'react'

export function ScrollProgressBar() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const update = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight
      setProgress(scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0)
    }
    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
    }
  }, [])

  return (
    <div className="pointer-events-none fixed left-0 top-0 z-50 h-[3px] w-full" aria-hidden>
      <div
        className="h-full"
        style={{ width: `${progress * 100}%`, background: 'var(--accent)', transition: 'width 0.1s linear' }}
      />
    </div>
  )
}
