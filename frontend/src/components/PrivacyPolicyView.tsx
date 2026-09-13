'use client'

import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

import { PRIVACY_POLICY_EFFECTIVE_DATE, PRIVACY_POLICY_SECTIONS } from '@/lib/privacyPolicy'
import { Eyebrow, Rule } from '@/components/ui'

export function PrivacyPolicyView() {
  return (
    <main className="relative z-10 mx-auto max-w-[760px] px-[26px] pb-[110px] pt-16">
      <Link
        href="/"
        className="mb-8 inline-flex items-center gap-2 text-[13px] font-extrabold opacity-70 hover:opacity-100"
      >
        <ArrowLeft size={15} aria-hidden /> Back
      </Link>

      <Eyebrow className="mb-[14px]">Legal</Eyebrow>
      <h1 className="m-0 mb-3 text-[clamp(32px,5vw,48px)] font-extrabold leading-[1.02] tracking-[-0.03em]">
        Privacy Policy
      </h1>
      <p className="m-0 mb-10 text-[13px] opacity-55">Effective {PRIVACY_POLICY_EFFECTIVE_DATE}</p>

      <Rule className="mb-10" />

      <div className="grid gap-10">
        {PRIVACY_POLICY_SECTIONS.map((section) => (
          <section key={section.heading}>
            <h2 className="m-0 mb-3 text-[20px] font-extrabold tracking-[-0.02em]">{section.heading}</h2>
            <div className="grid gap-3">
              {section.body.map((paragraph, i) => (
                <p key={i} className="m-0 text-[15px] leading-[1.65] opacity-75">
                  {paragraph}
                </p>
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  )
}
