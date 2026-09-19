/**
 * How strong a recommendation is, stated plainly. A number people can read at a
 * glance beats a coloured bar they have to decode — and it keeps the palette to
 * black, scarlet and white.
 */
export default function MatchScore({ score, note }: { score: number; note?: string }) {
  const pct = Math.round(Math.min(Math.max(score, 0), 1) * 100);
  const strong = pct >= 80;
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-[.78rem] font-semibold tabular-nums"
            style={{ color: strong ? 'var(--lamp-hi)' : 'var(--halide-mid)' }}>
        {pct}%
      </span>
      <span className="text-[.7rem]" style={{ color: 'var(--halide-dim)' }}>match</span>
      {note && (
        <>
          <span aria-hidden="true" style={{ color: 'var(--ink-edge)' }}>·</span>
          <span className="truncate text-[.7rem]" style={{ color: 'var(--halide-dim)' }}>{note}</span>
        </>
      )}
    </div>
  );
}
