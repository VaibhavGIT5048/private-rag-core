import type { Metadata } from 'next'

import { PrivacyPolicyView } from '@/components/PrivacyPolicyView'

export const metadata: Metadata = {
  title: 'Privacy Policy · Grounded RAG',
  description: 'What Grounded RAG collects, how it is used, and how to remove it.',
}

export default function PrivacyPage() {
  return <PrivacyPolicyView />
}
