import type { Metadata } from 'next'

import './globals.css'
import { Providers } from './providers'

const fontStack = 'ui-sans-serif, system-ui, sans-serif'

export const metadata: Metadata = {
  title: 'Grounded RAG — Ask questions of any PDF',
  description:
    'Private, cited answers over your documents with structure-aware chunking, hybrid retrieval, reranking and JWT-protected chat.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // data-theme is set to dark up-front so first paint matches the
    // default; UiPrefsProvider corrects it from localStorage on mount.
    <html
      lang="en"
      data-theme="dark"
      data-motion="full"
      style={{ ['--font-archivo' as never]: fontStack }}
    >
      <body className="font-sans antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
