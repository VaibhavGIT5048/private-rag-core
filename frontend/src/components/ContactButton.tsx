'use client'

// A floating contact affordance every route can reach. There's no support
// inbox behind this app, so "contact" means the same place bug reports and
// questions already go: a new GitHub issue on the repo.

import { MessageCircle } from 'lucide-react'

import { REPO_URL } from '@/config'

export function ContactButton() {
  return (
    <a
      href={`${REPO_URL}/issues/new`}
      target="_blank"
      rel="noreferrer"
      aria-label="Report an issue or ask a question on GitHub"
      title="Report an issue or ask a question"
      className="fixed bottom-6 right-6 z-40 flex h-12 w-12 items-center justify-center transition-transform hover:scale-105"
      style={{
        background: 'var(--accent)',
        color: 'var(--on-accent)',
        borderRadius: '999px',
        boxShadow: 'var(--shadow)',
      }}
    >
      <MessageCircle size={20} aria-hidden />
    </a>
  )
}
