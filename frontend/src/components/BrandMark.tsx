// Grounded RAG mark: an ungrounded answer settles onto a source line and
// receives its citation. Kept as SVG so it remains crisp in the header and at
// favicon-sized scales without shipping a raster asset.
export function BrandMark({ size = 28 }: { size?: number }) {
  const line = 'var(--ink)'
  const node = 'var(--accent)'
  const float = 'var(--brd)'

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 120 120"
      role="img"
      aria-label="Grounded RAG mark"
      className="brand-mark shrink-0"
    >
      <line x1="24" y1="96" x2="96" y2="96" stroke={line} strokeWidth="4" />
      <line
        x1="60"
        y1="49"
        x2="60"
        y2="96"
        stroke={line}
        strokeWidth="3"
        strokeDasharray="4 5"
        className="brand-mark__connector"
      />
      <circle cx="60" cy="43" r="24" fill="none" stroke={node} strokeWidth="1.5" strokeDasharray="3 4" className="brand-mark__halo" />
      <circle cx="60" cy="43" r="13" fill={float} className="brand-mark__answer" />
      <g className="brand-mark__citation">
        <rect x="74" y="85" width="22" height="16" fill={node} />
        <text x="85" y="97" fill="var(--on-accent)" fontSize="10" fontWeight="800" fontFamily="ui-sans-serif, system-ui, sans-serif" textAnchor="middle">1</text>
      </g>
    </svg>
  )
}
